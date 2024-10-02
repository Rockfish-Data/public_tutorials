import asyncio
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import prophet
import pyarrow as pa
import pykalman as pk
import rockfish as rf
import rockfish.actions as ra
from neuralprophet import NeuralProphet
from sklearn.metrics import f1_score, confusion_matrix


REF_TIME = pd.Timestamp("2023-06-01 00:00:00")


async def get_synthetic_data(model_to_gen_conf):
    # connect to Rockfish platform
    conn = rf.Connection.from_config()

    syn_datasets = []
    for model_label, gen_params in model_to_gen_conf.items():
        print(f"generating from model {model_label} with params = {gen_params}")
        generate_conf = pickle.load(open("generate_conf.pkl", "rb"))

        # USUALLY IN THE DEMO WE WOULD SHOW GENERATION LIVE, SO PUT A WORKFLOW_ID HERE WITH ALREADY TRAINED MODELS
        model = await conn.list_models(labels={"kind": model_label}).last()

        builder = rf.WorkflowBuilder()
        builder.add_path(model, generate_conf, ra.DatasetSave(name="synthetic"))
        workflow = await builder.start(conn)
        syn_datasets.append((await workflow.datasets().concat(conn)).table)

    return pa.concat_tables(syn_datasets)


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


def evaluate_model_performance(
    data,
    feature="feature_15",
    test_nrows=5000,
    model="prophet",
    setup="Baseline",
    mark_tp_fp=False,
):
    data = pd.concat(data)

    data = data[[feature, "timestamp"]].rename(columns={"timestamp": "ds", feature: "y"})
    data["ds"] = pd.to_datetime(data["ds"], format="%Y-%m-%d %H:%M:%S", exact=False)
    data = data.dropna()
    data = data.drop_duplicates(subset=["ds"], keep="first").sort_values(by="ds")

    # load test features and labels
    test = pd.read_csv("test.csv", nrows=test_nrows)
    test_labels = pd.read_csv("test_label.csv", nrows=test_nrows)

    if model == "prophet":
        forecast = forecast_using_prophet(data, test, setup)
    elif model == "neural_prophet":
        forecast = forecast_using_neural_prophet(data, test, setup)
    elif model == "window":
        forecast = forecast_using_window(data, test, k=7500, setup=setup)
    elif model == "kalman":
        forecast = forecast_using_kalman(data, test, feature, setup)

    # get anomaly labels
    pred_labels = np.where(test[feature] <= forecast["yhat_upper"], 0, 1)

    # plot
    fig, ax = plt.subplots()
    x = pd.to_datetime(forecast["ds"])  # plot timestamps on x axis
    ax.plot(x, test[feature], "g", label="True Value")
    ax.plot(x, forecast["yhat"], "b", label="Predicted Value")
    ax.fill_between(x, forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.1)
    ax.set_ylim(0.3, 0.65)

    # mark true and false positives
    if mark_tp_fp:
        tp_idxs = np.where((test_labels["label"] == 1) & (pred_labels == 1))[0]  # get idxs for true positives
        fp_idxs = np.where((test_labels["label"] == 0) & (pred_labels == 1))[0]  # get idxs for false positives
        ax.plot(x.iloc[tp_idxs], test[feature].iloc[tp_idxs], "r.", label="True Anomaly")
        ax.plot(x.iloc[fp_idxs], test[feature].iloc[fp_idxs], "k.", label="False Anomaly")

    ax.legend()
    plt.title(f"{model}, {setup}")
    plt.show()

    # compute and return f1 score
    print(f"Setup: {setup}")
    print(f"F1 Score: {f1_score(y_true=test_labels['label'], y_pred=pred_labels):.2f}")
    tn, fp, fn, tp = confusion_matrix(y_true=test_labels["label"], y_pred=pred_labels).ravel()
    print(f"TP: {tp}, FP: {fp}")
    print(f"True Positive Rate: {((tp / (tp + fn)) * 100):.2f}%")


async def generate():
    # TODO: make this match template
    model_label_to_gen_conf = {
        "location3.csv": {
            "sessions": 1500,
        },
        # EXAMPLE:
        # "jan": {
        #     "sessions": 500,
        # }
        # "feb": {
        #     "sessions": 500,
        # }
    }
    # syn_data = await get_synthetic_data(model_label_to_gen_conf)

    loc1_data = pd.read_csv("location1.csv")
    loc2_data = pd.read_csv("location2.csv")
    loc3_syn_data = pd.read_csv("syn_data.csv")
    loc3_real_data = pd.read_csv("location3.csv")
    loc3_hack_data = None  # TODO: competing approach

    model = "prophet"

    # baseline: missing location3 data
    evaluate_model_performance(
        [loc1_data, loc2_data], model=model, setup="Baseline", mark_tp_fp=True
    )

    # rf: use synthetic location3
    evaluate_model_performance(
        [loc1_data, loc2_data, loc3_syn_data],
        model=model,
        setup="Rockfish",
        mark_tp_fp=True
    )

    # ideal: use real location3
    evaluate_model_performance(
        [loc1_data, loc2_data, loc3_real_data],
        model=model,
        setup="Ideal",
        mark_tp_fp=True
    )


asyncio.run(generate())
