import time
import rockfish as rf
import asyncio
import pickle
from pathlib import Path


async def runtime():
    """
    Note:
        The connection object employs the from_config method to connect to the Rockfish platform.
        More on how to set this up for your system can be found at: https://docs142.rockfish.ai/sdk-overview.html#connection
        Alternatively, you can use the following code to connect to the platform:
        conn = rf.Connection.remote("https://api.rockfish.ai", "<API KEY>")
    """
    async with rf.Connection.from_config() as conn:  # connect to Rockfish platform
        # load runtime_conf (obtained after onboarding is complete)
        runtime_conf = pickle.load(open("runtime_conf.pkl", "rb"))
        datastream = runtime_conf.actions["datastream-load"]

        # start runtime
        runtime_workflow = await runtime_conf.start(conn)

        # stream datasets to model
        dirpath = Path("datafiles/location3_hours")
        dataset_paths = sorted([
            file.name for file in dirpath.glob('location3_*.csv')
            if not file.name.endswith('_timestamp.csv')
        ])

        # uncomment if specific datasets need to be trained on

        # start_idx = 0
        # end_idx = 1
        # dataset_paths = dataset_paths[start_idx:end_idx]

        try:
            for i, path in enumerate(dataset_paths):
                dataset = rf.Dataset.from_csv("train", f"datafiles/location3_hours/{path}")
                await runtime_workflow.write_datastream(datastream, dataset)
                time.sleep(10)
                print(f"Training model {i} on {path}")

            await runtime_workflow.close_datastream(datastream)
        except:
            await runtime_workflow.stop()
            raise

        # add labels
        for i, path in enumerate(dataset_paths):
            model = await runtime_workflow.models().nth(i)
            label = path[10:-4]
            await model.add_labels(conn, kind=f"model_{label}")
            print(f"Finished training model {i} on {path}")

asyncio.run(runtime())