import rockfish as rf
import asyncio
import pickle

async def runtime():
    async with rf.Connection.from_config() as conn:  # connect to Rockfish platform
        # load runtime_conf (obtained after onboarding)
        runtime_conf = pickle.load(open("runtime_conf.pkl", "rb"))
        datastream = runtime_conf.actions["datastream-load"]

        # start runtime
        runtime_workflow = await runtime_conf.start(conn)
        print(f'Runtime ID [for debugging now, not shown in demo]: {runtime_workflow.id()}')

        # stream datasets to model
        # each running workflow is for one datasource, and each chunk
        # comes from this datasource over time
        # so depending on the demo, you might have to set up more than
        # one running workflow (e.g. for each location)
        try:
            dataset_paths = ["test.csv"]  # ONLY CHANGE THIS PER DEMO USE CASE, example: ["jan_data.csv", "feb_data.csv"]
            for i, path in enumerate(dataset_paths):
                dataset = rf.Dataset.from_csv("train", path)
                await runtime_workflow.write_datastream(datastream, dataset)
                print(f"Training model {i} on {path}")
            await runtime_workflow.close_datastream(datastream)
        except:
            await runtime_workflow.stop()
            raise

        # optional: add labels
        labels = ["model1"]  # ONLY CHANGE THIS PER DEMO USE CASE, example: ["jan_data.csv", "feb_data.csv"]
        for i, path in enumerate(dataset_paths):
            model = await runtime_workflow.models().nth(i)
            await model.add_labels(conn, kind=labels[i])
            print(f"Finished training model {i} on {path}")

asyncio.run(runtime())