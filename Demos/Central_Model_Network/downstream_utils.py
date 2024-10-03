import numpy as np
import pandas as pd
import prophet
import pykalman as pk
from neuralprophet import NeuralProphet

REF_TIME = pd.Timestamp("2023-06-01 00:00:00")


def forecast_using_prophet(data, test, setup=None):
    if setup == "Ideal":
        model = prophet.Prophet(daily_seasonality=True, changepoint_prior_scale=.75, interval_width=.95)
    elif setup == "Baseline":
        model = prophet.Prophet(daily_seasonality=True, changepoint_prior_scale=.25)
    else:
        model = prophet.Prophet(daily_seasonality=True)
    model.fit(data)

    # make future df that matches test.csv timestamps
    new_times = test["timestamp_(min)"].apply(lambda x: pd.Timedelta(x, "min") + REF_TIME)
    future = pd.DataFrame()
    future["ds"] = new_times

    # make predictions using learnt model
    forecast = model.predict(future)
    forecast.to_csv(f"forecast_prophet_{setup}.csv")

    return forecast


def forecast_using_neural_prophet(data, test, setup=None):
    data["ds"] = pd.date_range(start=data["ds"].iloc[0], periods=len(data), freq="min")

    confidence_level = 0.8
    boundaries = round((1 - confidence_level) / 2, 2)
    quantiles = [boundaries, confidence_level + boundaries]
    model = NeuralProphet(daily_seasonality=True, quantiles=quantiles)
    _ = model.fit(data)

    # make future df that matches test.csv timestamps
    new_times = test["timestamp_(min)"].apply(lambda x: pd.Timedelta(x, "min") + REF_TIME)
    future = pd.DataFrame()
    future["ds"] = new_times
    future["y"] = [None] * len(new_times)

    # make predictions using learnt model
    forecast = model.predict(future)
    forecast = forecast.rename(
        columns={
            "yhat1": "yhat",
            "yhat1 10.0%": "yhat_lower",
            "yhat1 90.0%": "yhat_upper",
        }
    )
    forecast.to_csv(f"forecast_neural_prophet_{setup}.csv")

    return forecast


def forecast_using_window(data, test, k=100, setup=None):
    new_times = test["timestamp_(min)"].apply(lambda x: pd.Timedelta(x, "min") + REF_TIME)
    test["ds"] = new_times

    # take last k values for y
    window = data["y"].iloc[-k:]

    row_list = []
    for ds in test["ds"]:
        y_mean = np.mean(window)
        ci = 3 * np.std(window)

        row = {
            "ds": ds,
            "yhat": y_mean,
            "yhat_lower": y_mean - ci,
            "yhat_upper": y_mean + ci,
        }
        row_list.append(row)

        # update window
        window = np.append(window[1:], y_mean)

    forecast = pd.DataFrame(row_list)
    forecast.to_csv(f"forecast_window_{setup}.csv")

    return forecast


def forecast_using_kalman(data, test, feature, setup=None):
    new_times = test["timestamp_(min)"].apply(
        lambda x: pd.Timedelta(x, "min") + REF_TIME
    )
    test["ds"] = new_times

    model = pk.KalmanFilter()
    model.em(data["y"])
    (filtered_state_means, filtered_state_covariances) = model.filter(data["y"][:100])

    row_list = []
    for i, ds in enumerate(test["ds"]):
        y_mean, var = model.filter_update(
            filtered_state_means[i], filtered_state_covariances[i], test[feature][i + 1]
        )
        y_mean = y_mean[0]
        ci = 3 * var[0][0]

        row = {
            "ds": ds,
            "yhat": y_mean,
            "yhat_lower": y_mean - ci,
            "yhat_upper": y_mean + ci,
        }
        row_list.append(row)

    forecast = pd.DataFrame(row_list)
    forecast.to_csv(f"forecast_kalman_{setup}.csv")

    return forecast
