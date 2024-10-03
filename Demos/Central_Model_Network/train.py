import time
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
    print(f'❗️Runtime ID [for debugging now, not shown in demo]: {runtime_workflow.id()}')

    # stream datasets to model
    # data files location https://drive.google.com/drive/folders/1PqESQgLIrz-ztBc9UoH5kZpqIwu9GFyd?usp=sharing
    dataset_paths = [
        "location3_hours/location3_2023-08-06_hour00.csv",
        "location3_hours/location3_2023-08-06_hour01.csv"
    ]
    for i, path in enumerate(dataset_paths):
        dataset = rf.Dataset.from_csv("train", path)
        await runtime_workflow.write_datastream(datastream, dataset)
        print(f"Training model {i} on {path}")

    time.sleep(10)

    # optional: add labels
    # TODO: share location3 models with central
    labels = [
        "model_location3_2023-08-06_hour00",
        "model_location3_2023-08-06_hour01",
    ]
    for i, path in enumerate(dataset_paths):
        model = await runtime_workflow.models().nth(i)
        await model.add_labels(conn, kind=labels[i])
        print(f"Finished training model {i} on {path}")

asyncio.run(runtime())