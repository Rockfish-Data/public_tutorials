import rockfish as rf
import rockfish.actions as ra
import pyarrow as pa
import pickle
import asyncio
import requests

def upload_data(dataset, dbrx_url, table_name):
    databricks_token = "{{ secret.databricks_token }}"
    response = requests.post(
        f"https://dbc-224b2644-c532.cloud.databricks.com/api/2.0/dbfs/put",
        headers={"Authorization": f"Bearer {databricks_token}"},
        data={"path": dbrx_url, "overwrite": "true"},
        files=dataset,
    )
    if response.status_code == 200:
        print(f"File uploaded to {dbrx_url}")
    else:
        print(f"Upload failed: {response.text}")


async def get_synthetic_data(model_to_gen_conf):
    # connect to Rockfish platform
    conn = rf.Connection.from_config()

    syn_datasets = []
    for model_label, gen_params in model_to_gen_conf.items():
        print(f"Generating from model {model_label} with params = {gen_params}")
        generate_conf = pickle.load(open("generate_conf.pkl", "rb"))

        model = await conn.list_models(
            labels={"kind": model_label, "workflow_id": "75DPfO3rlchVoA1bybrXFE"}
        ).last()

        generate_conf.config().doppelganger.sessions = gen_params["sessions"]

        builder = rf.WorkflowBuilder()
        builder.add_path(
            model,
            generate_conf,
            ra.DatasetSave(name="synthetic", concat_tables=True, concat_session_key="session_key")
        )
        workflow = await builder.start(conn)
        syn_datasets.append((await workflow.datasets().concat(conn)).table)
        print(f"Finished generating {gen_params['sessions']} sessions from model {model_label}")

    await conn.close()

    return pa.concat_tables(syn_datasets)

async def generate():
    generate_conf = {
        "source1": {
            "start_time": "2023-08-08 09:00:00",
            "end_time": "2023-08-08 18:00:00",
            "model": "model_normal_transactions",
            "sessions": 1500,
        },
        "source2": {
            "start_time": "2023-08-08 12:30:00",
            "end_time": "2023-08-08 14:30:00",
            "model": "model_abnormal_transactions",
            "sessions": 500,
        },
    }
    syn_data = await get_synthetic_data(generate_conf)

    upload_data(
        dataset=syn_data,
        dbrx_url="dbc-224b2644-c532.cloud.databricks.com/sql/1.0/warehouses/bbdd6ab06ef5dc44/",
        table_name="demo_transactions"
    )

asyncio.run(generate())