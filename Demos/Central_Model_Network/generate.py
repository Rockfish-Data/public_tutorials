import asyncio
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import rockfish as rf
import rockfish.actions as ra
from sklearn.metrics import f1_score, confusion_matrix

from downstream_utils import (
    forecast_using_prophet,
    forecast_using_neural_prophet,
    forecast_using_window,
    forecast_using_kalman,
)


async def get_synthetic_data(generate_conf):
    # connect to Rockfish platform
    conn = rf.Connection.from_config()

    syn_datasets = []
    for source, params in generate_conf.items():
        model_label = params["model"]
        print(f"Generating from {model_label}")
        generate_conf = pickle.load(open("generate_conf.pkl", "rb"))

        model = await conn.list_models(
            labels={"kind": model_label, "workflow_id": "5CCxuX4EGLJvG3qSYhPC9p"}
        ).last()

        builder = rf.WorkflowBuilder()
        builder.add_path(model, generate_conf, ra.DatasetSave(name="synthetic"))
        workflow = await builder.start(conn)

        # save syn data for quality checks
        syn_dataset = (await workflow.datasets().concat(conn)).table
        filename = model_label[6:]
        syn_dataset.to_pandas().to_csv(f"syn_location3_hours/{filename}.csv", index=False)

        # add timestamps to syn data
        timestamps = pd.read_csv(f"location3_hours/{filename}_timestamp.csv")[
            "timestamp"
        ].to_list()
        syn_dataset = syn_dataset.append_column("timestamp", [timestamps])
        syn_datasets.append(syn_dataset)

    return pa.concat_tables(syn_datasets)


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
    generate_conf = {
        "source1": {"model": "model_location3_2023-08-06_hour00"},
        "source2": {"model": "model_location3_2023-08-06_hour01"},
    }
    syn_data = await get_synthetic_data(generate_conf)
    syn_data.to_pandas().to_csv("test_tab_gan.csv", index=False)

    exit(0)

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
        mark_tp_fp=True,
    )

    # ideal: use real location3
    evaluate_model_performance(
        [loc1_data, loc2_data, loc3_real_data],
        model=model,
        setup="Ideal",
        mark_tp_fp=True,
    )


asyncio.run(generate())
