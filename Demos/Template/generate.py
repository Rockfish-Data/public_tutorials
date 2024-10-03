import rockfish as rf
import rockfish.actions as ra
import pyarrow as pa
import pickle
import asyncio

async def get_synthetic_data(generate_conf):
    # connect to Rockfish platform
    conn = rf.Connection.from_config()

    syn_datasets = []
    for source, params in generate_conf.items():
        model_label = params["model"]
        print(f"Generating from {model_label}")
        generate_conf = pickle.load(open("generate_conf.pkl", "rb"))

        # USUALLY IN THE DEMO WE WOULD SHOW GENERATION LIVE, SO PUT A WORKFLOW_ID HERE WITH ALREADY TRAINED MODELS
        model = await conn.list_models(labels={"kind": model_label, "workflow_id": "3HnOxCXK5OO7MpHEzYRee5"}).last()
        print(model)

        builder = rf.WorkflowBuilder()
        builder.add_path(model, generate_conf, ra.DatasetSave(name="synthetic"))
        workflow = await builder.start(conn)
        syn_datasets.append((await workflow.datasets().concat(conn)).table)

    return pa.concat_tables(syn_datasets)

async def generate():
    # ONLY CHANGE THIS PER DEMO USE CASE
    # e.g. for product demo, we want to show blending and amplification
    #      for AI model training, we want to show generating missing location data
    generate_conf = {
        "source1": {
            "start_time": "",
            "end_time": "",
            "model": "model1",
            "sessions": 1500,
        },
        # EXAMPLE:
        # "source1": {
        #     "start_time": "",
        #     "end_time": "",
        #     "model": "model1",
        #     "sessions": 500,
        # }
        # "source2": {
        #     "start_time": "",
        #     "end_time": "",
        #     "model": "model2",
        #     "sessions": 1500,
        # }
    }
    syn_data = await get_synthetic_data(generate_conf)

    # DOWNSTREAM CODE THAT USES SYN DATA GOES HERE
    # e.g. for product demo, save syn_data to file
    #      for AI model training, add xgboost/ransyncoder code here
    pa.csv.write_csv(syn_data, "synthetic.csv")

asyncio.run(generate())