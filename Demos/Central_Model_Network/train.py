import time
import rockfish as rf
import asyncio
import pickle
from pathlib import Path


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
    dirpath = Path("datafiles/location3_hours")
    dataset_paths = sorted([
        file.name for file in dirpath.glob('location3_*.csv')
        if not file.name.endswith('_timestamp.csv')
    ])
    # start_idx = 30
    # end_idx = 35
    # dataset_paths = dataset_paths[start_idx:end_idx]

    for i, path in enumerate(dataset_paths):
        dataset = rf.Dataset.from_csv("train", f"datafiles/location3_hours/{path}")
        await runtime_workflow.write_datastream(datastream, dataset)
        time.sleep(10)
        print(f"Training model {i} on {path}")

    # add labels
    # TODO: share location3 models with central
    # for i, path in enumerate(dataset_paths):
    #     model = await runtime_workflow.models().nth(i)
    #     label = path[10:-4]
    #     await model.add_labels(conn, kind=f"model_{label}")
    #     print(model)
    #     print(f"Finished training model {i} on {path}")

asyncio.run(runtime())