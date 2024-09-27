import time

import rockfish as rf
import asyncio
import pickle
from actions.dg.train import TrainTimeGAN

async def runtime():
    # connect to Rockfish platform
    conn = rf.Connection.from_config()

    # load runtime_conf (obtained after onboarding)
    runtime_conf = pickle.load(open("runtime_conf.pkl", "rb"))
    datastream = runtime_conf.actions["datastream-load"]

    # start runtime
    # builder = rf.WorkflowBuilder()
    # del runtime_conf.actions['datastream-load']
    # builder.add_path(rf.Dataset.from_csv('delete_after', 'location3.csv'),
    #                  TrainTimeGAN(rf.converter.unstructure(list(runtime_conf.actions.values())[0].config())))
    # runtime_workflow = await builder.start(conn)

    builder = rf.WorkflowBuilder()
    del runtime_conf.actions['datastream-load']
    builder.add_path(rf.Dataset.from_csv('delete_after', 'location3.csv'),
                     *runtime_conf.actions.values())
    runtime_workflow = await builder.start(conn)
    # runtime_workflow = await runtime_conf.start(conn)
    print(f'❗️Runtime ID [for debugging now, not shown in demo]: {runtime_workflow.id()}')

    # stream datasets to model
    # each running workflow is for one datasource, and each chunk
    # comes from this datasource over time
    # so depending on the demo, you might have to set up more than
    # one running workflow (e.g. for each location)
    # data files location https://drive.google.com/drive/folders/1PqESQgLIrz-ztBc9UoH5kZpqIwu9GFyd?usp=sharing
    dataset_paths = ["location3.csv"]  # ONLY CHANGE THIS PER DEMO USE CASE, example: ["jan_data.csv", "feb_data.csv"]
    for i, path in enumerate(dataset_paths):
        dataset = rf.Dataset.from_csv("train", path)
        # await runtime_workflow.write_datastream(datastream, dataset)
        time.sleep(7)
        print(f"Training model {i} on {path}")

    async for log in runtime_workflow.logs(level=rf.LogLevel.DEBUG):
        print(log)

    # optional: add labels
    for i, path in enumerate(dataset_paths):
        model = await runtime_workflow.models().nth(i)
        await model.add_labels(conn, kind=path)
        print(f"Finished training model {i} on {path}")

asyncio.run(runtime())