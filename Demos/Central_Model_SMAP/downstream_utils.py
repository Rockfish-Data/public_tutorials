import numpy as np
import pandas as pd
import prophet

REF_TIME = pd.Timestamp("2023-06-01 00:00:00")

def forecast_using_prophet(data, test, setup=None):
    model = prophet.Prophet(daily_seasonality=True)
    np.random.seed(500)
    model.fit(data)

    # make future df that matches test.csv timestamps
    new_times = test["timestamp_(min)"].apply(lambda x: pd.Timedelta(x, "min") + REF_TIME)
    future = pd.DataFrame()
    future["ds"] = new_times

    # make predictions using learnt model
    forecast = model.predict(future)
    forecast.to_csv(f"datafiles/downstream model files/forecast_prophet_{setup}.csv")

    return forecast