import numpy as np
import pandas as pd
import prophet
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt

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

def evaluate_model_performance(
        data,
        feature="feature_9",
        test_nrows=5000,
        setup=None,
        mark_tp_fp=False,
):
    data = pd.concat(data)

    # preprocess data for model training
    data = data[[feature, "timestamp"]].rename(columns={"timestamp": "ds", feature: "y"})
    data["ds"] = pd.to_datetime(data["ds"], format="%Y-%m-%d %H:%M:%S", exact=False)
    data = data.dropna()
    data = data.drop_duplicates(subset=["ds"], keep="first").sort_values(by="ds")

    # load test features and labels
    test = pd.read_csv("datafiles/test.csv", nrows=test_nrows)
    test_labels = pd.read_csv("datafiles/test_label.csv", nrows=test_nrows)

    forecast = forecast_using_prophet(data, test, setup)

    # get anomaly labels
    pred_labels = np.where(test[feature] <= forecast["yhat_upper"], 0, 1)

    # plot
    fig, ax = plt.subplots()
    x = pd.to_datetime(forecast["ds"])  # plot timestamps on x axis
    ax.plot(x, test[feature], "g", label="True Value")
    ax.plot(x, forecast["yhat"], "b", label="Predicted Value")
    ax.fill_between(x, forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.1)
    ax.set_ylim(0.4, 1.03)

    # mark true and false positives
    if mark_tp_fp:
        tp_idxs = np.where((test_labels["label"] == 1) & (pred_labels == 1))[0]  # get idxs for true positives
        fp_idxs = np.where((test_labels["label"] == 0) & (pred_labels == 1))[0]  # get idxs for false positives
        ax.plot(x.iloc[tp_idxs], test[feature].iloc[tp_idxs], "r.", label="True Anomaly")
        ax.plot(x.iloc[fp_idxs], test[feature].iloc[fp_idxs], "k.", label="False Anomaly")

    ax.legend()
    plt.xticks([])
    plt.xlabel("Time")
    plt.ylabel("Normalized Feature_9")
    plt.title(f"{setup}")
    plt.show()

    # compute and return f1 score
    print(f"Setup: {setup}")
    print(f"F1 Score: {f1_score(y_true=test_labels['label'], y_pred=pred_labels):.2f}")
    tn, fp, fn, tp = confusion_matrix(y_true=test_labels["label"], y_pred=pred_labels).ravel()
    print(f"TP: {tp}, FP: {fp}")
    print(f"True Positive Rate: {((tp / (tp + fn)) * 100):.2f}%")
    print(f"False Positive Rate: {((fp / (tn + fp)) * 100):.2f}%")
    print(f'Accuracy: {((tp + tn) / (tp + tn + fp + fn) * 100):.2f}%')

def compare_models(no_sharing = True,
                   ideal_case = True,
                   rockfish = True,
                   naive_syn = True):
    loc1_data = pd.read_csv("datafiles/location1.csv")
    loc2_data = pd.read_csv("datafiles/location2.csv")
    loc3_syn_data = pd.read_csv("datafiles/new_syn.csv")
    loc3_real_data = pd.read_csv("datafiles/location3.csv")
    loc3_naive_data = pd.read_csv('datafiles/naive_syn_data.csv')

    if no_sharing:
        # baseline: missing location3 data
        evaluate_model_performance(
            [loc1_data, loc2_data], setup="No Sharing", mark_tp_fp=True
        )

    if ideal_case:
        # ideal: use real location3
        evaluate_model_performance(
            [loc1_data, loc2_data, loc3_real_data],
            setup="Ideal Case",
            mark_tp_fp=True,
        )

    if rockfish:
        # rf: use synthetic location3
        evaluate_model_performance(
            [loc1_data, loc2_data, loc3_syn_data],
            setup="Rockfish",
            mark_tp_fp=True,
        )

    if naive_syn:
        # naive: use naive_syn
        evaluate_model_performance(
            [loc1_data, loc2_data, loc3_naive_data],
            setup='Naive Synthetic',
            mark_tp_fp=True,
        )