import rockfish as rf
import rockfish.actions as ra
import asyncio
import pickle

def get_dataset(dbrx_url, table_name):
    return ra.DatabricksSqlLoad(
        token="{{ secret.databricks_token }}",
        http_path="TBD",
        server_hostname=dbrx_url,
        sql=f"SELECT * FROM rockfish_data_dev.default.{table_name}",
    )


async def runtime():
    # connect to Rockfish platform
    async with rf.Connection.from_config() as conn:
        # load runtime_conf (obtained after onboarding)
        runtime_conf = pickle.load(open("runtime_conf.pkl", "rb"))
        datastream = runtime_conf.actions["datastream-load"]

        # start runtime
        runtime_workflow = await runtime_conf.start(conn)
        print(f'Runtime ID [for debugging now, not shown in demo]: {runtime_workflow.id()}')

        # stream datasets to model
        try:
            dataset_names = [
                "transactions_2023-08-01_hour09",
                "transactions_2023-08-01_hour10",
                "transactions_2023-08-01_hour11"
            ]
            for i, name in enumerate(dataset_names):
                # dataset = get_dataset(
                #     dbrx_url="dbc-224b2644-c532.cloud.databricks.com/sql/1.0/warehouses/bbdd6ab06ef5dc44/",
                #     table_name=name
                # )
                dataset = rf.Dataset.from_csv(name, f"{name}.csv")
                await runtime_workflow.write_datastream(datastream, dataset)
                print(f"Training model {i} on table: {name}")

            await runtime_workflow.close_datastream(datastream)
        except:
            await runtime_workflow.stop()
            raise

        # optional: add labels
        for i, name in enumerate(dataset_names):
            model = await runtime_workflow.models().nth(i)
            await model.add_labels(conn, kind=f"model_{name}")
            print(f"Finished training model {i} on {name}")

asyncio.run(runtime())