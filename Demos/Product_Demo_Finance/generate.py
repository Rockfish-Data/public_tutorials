import rockfish as rf
import rockfish.actions as ra
import pyarrow as pa
import pickle
import asyncio

from rockfish.arrow import concat_tables


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
    # TODO: introduce this story in the beginning
    model_label_to_gen_conf = {
        "week1": {
            # start:
            # end:
            # model:
            "sessions": 250,
            # TODO: accept/retain time range
        },
        "week2": {
            "sessions": 500,
            # TODO: accept/retain time range
        },
    }
    syn_data = await get_synthetic_data(model_label_to_gen_conf)

    # TODO: show connector
    pa.csv.write_csv(syn_data, "synthetic.csv")

asyncio.run(generate())