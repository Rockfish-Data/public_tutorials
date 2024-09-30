import pandas as pd
import rockfish as rf
import rockfish.actions as ra
import pyarrow as pa
import pickle
import asyncio
import prophet
import matplotlib.pyplot as plt
import numpy as np
from utils import prophet_fit, get_outliers, prophet_plot


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


def evaluate_model_performance(data, feature="feature_15", title=None):
    data = pd.concat(data)

    data = data[[feature, 'timestamp']].rename(columns={'timestamp':'ds', feature:'y'})
    data['ds'] = pd.to_datetime(data['ds'], format="%Y-%m-%d %H:%M:%S", exact=False)
    data = data.dropna()

    model = prophet.Prophet()
    model.fit(data)

    # load test features and labels
    test = pd.read_csv('test.csv', nrows=5000)
    test_labels = pd.read_csv('test_label.csv', nrows=5000)

    # make future df that matches test.csv timestamps
    reference_time = pd.Timestamp('2023-06-01 00:00:00')
    new_times = test['timestamp_(min)'].apply(lambda x: pd.Timedelta(x,'min') + reference_time)
    future = pd.DataFrame()
    future['ds'] = new_times

    # make predictions using learnt model
    forecast = model.predict(future)
    forecast.to_csv(f"forecast_{title}.csv")

    # get anomaly labels
    pred_labels = np.where(test[feature].between(forecast['yhat_lower'], forecast['yhat_upper']), 0, 1)

    # plot
    fig, ax = plt.subplots()
    x = pd.to_datetime(forecast['ds'])  # plot timestamps on x axis
    ax.plot(x, test[feature], 'g', label="True Value")
    ax.plot(x, forecast['yhat'], 'b', label="Predicted Value")
    ax.fill_between(x, forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.1)

    # mark true and false positives
    tp_idxs = np.where((test_labels['label'] == 1) & (pred_labels == 1))[0]  # get idxs for true positives
    fp_idxs = np.where((test_labels['label'] == 0) & (pred_labels == 1))[0]  # get idxs for false positives
    ax.plot(x.iloc[tp_idxs], test[feature].iloc[tp_idxs], "r.", label="True Anomaly")
    ax.plot(x.iloc[fp_idxs], test[feature].iloc[fp_idxs], "k.", label="False Anomaly")

    ax.legend()
    plt.title(title)
    plt.show()


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
    locx_data = pd.read_csv("locationx.csv")
    loc3_hack_data = None  # TODO: competing approach

    # TODO: debug baseline
    # evaluate_model_performance([loc1_data, loc2_data], title="Baseline")  # baseline: missing location3 data
    # evaluate_model_performance([loc1_data, loc2_data, loc3_syn_data], title="Rockfish")  # rf: use synthetic location3
    # evaluate_model_performance([loc1_data, loc2_data, loc3_hack_data], title="Hack")  # rf: use hack location3
    evaluate_model_performance([loc1_data, loc2_data, loc3_real_data], title="Ideal")  # ideal: use real location3



asyncio.run(generate())