import pandas as pd
import numpy as np
import rockfish as rf
import rockfish.actions as ra
import pyarrow as pa
import pickle
import asyncio
import prophet
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


def evaluate_model_performance(data, feature="feature_13"):
    data = pd.concat(data)

    data = data[[feature, 'timestamp']].rename(columns={'timestamp':'ds', feature:'y'})
    data['ds'] = pd.to_datetime(data['ds'], format="%Y-%m-%d %H:%M:%S", exact=False)
    data = data.dropna()

    model = prophet.Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=60, freq='min')
    forecast = model.predict(future)
    fig = model.plot(forecast)
    fig.show()


async def generate():
    # ONLY CHANGE THIS PER DEMO USE CASE
    # e.g. for product demo, we want to show blending and amplification
    #      for AI model training, we want to show generating missing location data
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

    # DOWNSTREAM CODE THAT USES SYN DATA GOES HERE
    # e.g. for product demo, save syn_data to file
    #      for AI model training, add xgboost/ransyncoder code here
    # pa.csv.write_csv(syn_data, "location3_synthetic.csv")

    loc1_data = pd.read_csv("location1.csv")
    loc2_data = pd.read_csv("location2.csv")
    loc3_data = pd.read_csv("syn_data.csv")
    loc3_real_data = pd.read_csv("location3.csv")
    locx_data = pd.read_csv("locationx.csv")

    evaluate_model_performance([locx_data])  # baseline: missing location3 data
    evaluate_model_performance([locx_data, loc3_data])  # rf: use synthetic location3
    evaluate_model_performance([locx_data, loc3_real_data])  # ideal: use real location3



asyncio.run(generate())