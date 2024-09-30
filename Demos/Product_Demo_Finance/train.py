import rockfish as rf
import asyncio
import pickle

async def runtime():
    # connect to Rockfish platform
    conn = rf.Connection.from_config()

    # load runtime_conf (obtained after onboarding)
    runtime_conf = pickle.load(open("runtime_conf.pkl", "rb"))
    datastream = runtime_conf.actions["datastream-load"]

    # start runtime
    runtime_workflow = await runtime_conf.start(conn)
    print(f'Runtime ID [for debugging now, not shown in demo]: {runtime_workflow.id()}')

    # stream datasets to model
    # TODO: use connector
    dataset_paths = ["transactions_week1.csv", "transactions_week2.csv"]
    for i, path in enumerate(dataset_paths):
        dataset = rf.Dataset.from_csv("train", path)
        await runtime_workflow.write_datastream(datastream, dataset)
        print(f"Training model {i} on {path}")

    # optional: add labels
    # TODO: show path to model in store / show objects in store, model size
    for i, path in enumerate(dataset_paths):
        model = await runtime_workflow.models().nth(i)
        await model.add_labels(conn, kind=path)
        print(f"Finished training model {i} on {path}")

asyncio.run(runtime())