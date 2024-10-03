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
    conn = rf.Connection.from_config()

    # load runtime_conf (obtained after onboarding)
    runtime_conf = pickle.load(open("runtime_conf.pkl", "rb"))
    datastream = runtime_conf.actions["datastream-load"]

    # start runtime
    runtime_workflow = await runtime_conf.start(conn)
    print(f'Runtime ID [for debugging now, not shown in demo]: {runtime_workflow.id()}')

    # stream datasets to model on normal transactions
    dataset_names = [
        "normal_transactions_day1",
        "normal_transactions_day2",
    ]
    for i, name in enumerate(dataset_names):
        dataset = get_dataset(
            dbrx_url="dbc-224b2644-c532.cloud.databricks.com/sql/1.0/warehouses/bbdd6ab06ef5dc44/",
            table_name=name
        )
        await runtime_workflow.write_datastream(datastream, dataset)
        print(f"Training model {i} on {name}")

    # optional: add labels
    for i, path in enumerate(dataset_names):
        model = await runtime_workflow.models().nth(i)
        await model.add_labels(conn, kind=path)
        print(f"Finished training model {i} on {path}")

    # stream datasets to model on abnormal transactions
    dataset_names = [
        "abnormal_transactions_day1",
        "abnormal_transactions_day2",
    ]
    for i, name in enumerate(dataset_names):
        dataset = get_dataset(
            dbrx_url="dbc-224b2644-c532.cloud.databricks.com/sql/1.0/warehouses/bbdd6ab06ef5dc44/",
            table_name=name
        )
        await runtime_workflow.write_datastream(datastream, dataset)
        print(f"Training model {i} on {name}")

    # optional: add labels
    for i, path in enumerate(dataset_names):
        model = await runtime_workflow.models().nth(i)
        await model.add_labels(conn, kind=path)
        print(f"Finished training model {i} on {path}")

asyncio.run(runtime())