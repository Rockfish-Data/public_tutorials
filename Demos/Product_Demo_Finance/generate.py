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


async def get_synthetic_data(generate_conf):
    async with rf.Connection.from_config() as conn:  # connect to Rockfish platform
        syn_datasets = []
        for source, params in generate_conf.items():
            model_label = params["model"]
            print(f"Generating from {model_label} with params {params}")

            model = await conn.list_models(
                labels={"kind": model_label, "workflow_id": "1nBIPGpZ1pLOmbB0YehK7s"}
            ).last()

            generate_action = pickle.load(open("generate_conf.pkl", "rb"))
            session_target = ra.SessionTarget(target=params["sessions"], max_cycles=1000)
            save = ra.DatasetSave(name="synthetic", concat_tables=True, concat_session_key="session_key")

            builder = rf.WorkflowBuilder()

            builder.add_model(model)
            builder.add_action(generate_action, parents=[model, session_target])

            if "conditions" in params:
                post_amplify = ra.PostAmplify({
                    "query_ast": {
                        "and": [{"eq": ["fraud", 1]}, {"eq": ["category", "transportation"]}],
                    },
                    "drop_match_percentage": 0.0,
                    "drop_other_percentage": 0.95,
                })
                builder.add_action(post_amplify, parents=[generate_action])
                builder.add_action(session_target, parents=[post_amplify])
                builder.add_action(save, parents=[post_amplify])
            else:
                builder.add_action(session_target, parents=[generate_action])
                builder.add_action(save, parents=[generate_action])

            workflow = await builder.start(conn)

            syn = (await workflow.datasets().concat(conn))
            syn.to_pandas().to_csv(f"{model_label[6:]}_syn.csv", index=False)
            syn_datasets.append(syn.table)
            print(f"Finished generating {params['sessions']} sessions from model {model_label}")

        return pa.concat_tables(syn_datasets)

async def generate():
    generate_conf = {
        "source1": {
            "start_time": "2023-08-08 09:00:00",
            "end_time": "2023-08-08 18:00:00",
            "model": "model_transactions_2023-08-01_hour09",
            "sessions": 10000,
        },
        "source2": {
            "start_time": "2023-08-08 12:30:00",
            "end_time": "2023-08-08 14:30:00",
            "model": "model_transactions_2023-08-01_hour11",
            "conditions": {"category": ["transportation"]},
            "sessions": 1500,
        },
    }
    syn_data = await get_synthetic_data(generate_conf)
    syn_data.to_pandas().to_csv(f"story.csv", index=False)

    # upload_data(
    #     dataset=syn_data,
    #     dbrx_url="dbc-224b2644-c532.cloud.databricks.com/sql/1.0/warehouses/bbdd6ab06ef5dc44/",
    #     table_name="demo_transactions"
    # )

asyncio.run(generate())