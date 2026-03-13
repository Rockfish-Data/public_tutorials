import asyncio
import pickle
from pathlib import Path
import pandas as pd
import pyarrow as pa
import rockfish as rf
import rockfish.actions as ra
from downstream_utils import compare_models

async def get_synthetic_data(generate_conf):
    # connect to Rockfish platform
    """
    Note:
        The connection object employs the from_config method to connect to the Rockfish platform.
        More on how to set this up for your system can be found at: https://docs142.rockfish.ai/sdk-overview.html#connection
        Alternatively, you can use the following code to connect to the platform:
        conn = rf.Connection.remote("https://api.rockfish.ai", "<API KEY>")
    """
    conn = rf.Connection.from_config()

    syn_datasets = []
    for source, params in generate_conf.items():
        model_label = params["model"]
        print(f"Generating from {model_label}")
        generate_conf = pickle.load(open("generate_conf.pkl", "rb"))

        model = await conn.list_models(
            labels={"kind": model_label}
        ).last()

        builder = rf.WorkflowBuilder()
        builder.add_path(model, generate_conf, ra.DatasetSave(name="synthetic"))
        workflow = await builder.start(conn)

        # optionally, uncomment the below if you want to view logs
        # async for log in workflow.logs():
        #     print(log)

        filename = model_label[6:]

        # save syn data for quality checks
        syn_dataset = (await workflow.datasets().concat(conn)).table
        timestamps = pd.read_csv(f"datafiles/location3_hours/location3_{filename}_timestamp.csv")[
            "timestamp"
        ].to_list()
        syn_dataset = syn_dataset.slice(length=len(timestamps))  # keep len the same as real timestamp len
        syn_dataset.to_pandas().to_csv(f"datafiles/syn_location3_hours/location3_{filename}.csv", index=False)

        # add timestamps to syn data
        syn_dataset = syn_dataset.append_column("timestamp", [timestamps])
        syn_datasets.append(syn_dataset)

    await conn.session.close()

    return pa.concat_tables(syn_datasets)


async def generate():
    dirpath = Path("datafiles/location3_hours")
    dataset_paths = sorted([
        file.name for file in dirpath.glob('location3_*.csv')
        if not file.name.endswith('_timestamp.csv')
    ])
    start_idx,end_idx=None,None
    # uncomment if specific models need to be used during generation
    # start_idx = 0
    # end_idx = 1
    # dataset_paths = dataset_paths[start_idx:end_idx]

    # label matching for model querying
    generate_conf = {}
    for i, path in enumerate(dataset_paths):
        generate_conf[f"source{i}"] = {
            "model": f"model_{path[10:-4]}"
        }

    # uncomment the following lines if you want to generate your own synthetic data
    # it is commented out to allow for reproducibility of the demo
    # syn_data = await get_synthetic_data(generate_conf)
    # syn_data.to_pandas().to_csv(f"datafiles/loc3_syn_tabgan_{start_idx or ''}to{end_idx or ''}.csv", index=False)

    # post data generation, this function will compare the different model performance scenarios we have
    compare_models(True, True, True, True)


asyncio.run(generate())
