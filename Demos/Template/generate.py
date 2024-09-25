import rockfish as rf
import rockfish.actions as ra
import pyarrow as pa
import pickle
import asyncio

async def get_synthetic_data(model_to_gen_conf):
    # connect to Rockfish platform
    conn = rf.Connection.from_config()

    syn_datasets = []
    for model_label, gen_params in model_to_gen_conf.items():
        print(f"generating from model {model_label} with params = {gen_params}")
        generate_conf = pickle.load(open("generate_conf.pkl", "rb"))

        # USUALLY IN THE DEMO WE WOULD SHOW GENERATION LIVE, SO PUT A WORKFLOW_ID HERE WITH ALREADY TRAINED MODELS
        model = await conn.list_models(labels={"kind": model_label, "workflow_id": "3HnOxCXK5OO7MpHEzYRee5"}).last()

        builder = rf.WorkflowBuilder()
        builder.add_path(model, generate_conf, ra.DatasetSave(name="synthetic"))
        workflow = await builder.start(conn)
        syn_datasets.append((await workflow.datasets().concat(conn)).table)

    return pa.concat_tables(syn_datasets)

async def generate():
    # ONLY CHANGE THIS PER DEMO USE CASE
    # e.g. for product demo, we want to show blending and amplification
    #      for AI model training, we want to show generating missing location data
    model_label_to_gen_conf = {
        "test.csv": {
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
    syn_data = await get_synthetic_data(model_label_to_gen_conf)

    # DOWNSTREAM CODE THAT USES SYN DATA GOES HERE
    # e.g. for product demo, save syn_data to file
    #      for AI model training, add xgboost/ransyncoder code here
    pa.csv.write_csv(syn_data, "synthetic.csv")

asyncio.run(generate())