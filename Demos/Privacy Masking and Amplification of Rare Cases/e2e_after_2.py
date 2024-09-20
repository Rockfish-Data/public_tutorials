# %% [markdown]
# ### Low-Level Stuff That's Hidden

# %%
import time
import pandas as pd
import rockfish as rf
import rockfish.actions as ra
from rockfish.labs.dataset_properties import DatasetPropertyExtractor
from rockfish.labs.steps import Recommender


# %%
import rockfish as rf
rf.product_version

# %%
# connect to the rockfish platform
conn = rf.Connection.remote(
    'https://sunset-beach.rockfish.ai',
    'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE3MTE2NDc0MDQsImlzcyI6ImFwaSIsIm5iZiI6MTcxMTY0NzQwNCwidG9rZW5faWQiOiIxd3pBUWliNjRVb0c2MWVUazQ4SzBMIiwidXNlcl9pZCI6IjQ2MVNUOXZ4a0hYekpYRnJKYm4yWm0ifQ.MxG4VB5IrXQ2U_2ePUaoEN7gfy2fqPhD5tzSYYhnn2k'
)

# %%
async def start_rockfish_runtime(dataset_paths, session_key, metadata_fields, timestamp, mask_columns):
    chunk = rf.Dataset.from_csv("ds", dataset_paths[0])
    dataset_props = DatasetPropertyExtractor(
        dataset=chunk,
        session_key=session_key,
        metadata_fields=metadata_fields,
        timestamp=timestamp
    ).extract()
    recommender_output = Recommender(dataset_props).run()
    print(recommender_output.report)
    
    rec_train_actions = recommender_output.actions[:-1]
    rec_generate_actions = recommender_output.actions[-1]

    stream = ra.DatastreamLoad()
    builder = rf.WorkflowBuilder()
    builder.add_path(stream, *rec_train_actions)
    workflow = await builder.start(conn)
    print(f'Runtime Workflow ID [for debugging now, not shown in demo]: {workflow.id()}')
    
    return stream, workflow

# %%
async def create_rockfish_models(stream, workflow, dataset_paths):
    for dataset_path in dataset_paths:
        chunk = rf.Dataset.from_csv("ds", dataset_path)
        await workflow.write_datastream(stream, chunk)

# %%
async def create_blended_dataset(workflow, normal_sessions=100, abnormal_sessions=500):
    dataset_time_periods = [1, 2]
    
    i = 0
    model_map = {}
    async for model in (conn.models(labels={'workflow_id': workflow.id()})):
        chunk_id = dataset_time_periods[i]
        model_map[chunk_id] = model
        i = i + 1

    builder = rf.WorkflowBuilder()
    builder.add_path(model_map[1], rec_generate_actions, ra.DatasetSave(name='syn_data'))
    workflow = await builder.start(conn)
    print(f'Model 1 Generate Workflow ID [for debugging now, not shown in demo]: {workflow.id()}')
    syn_data1 = await workflow.datasets().concat(conn)

    builder = rf.WorkflowBuilder()
    builder.add_path(model_map[2], rec_generate_actions, ra.DatasetSave(name='syn_data'))
    workflow = await builder.start(conn)
    print(f'Model 2 Generate Workflow ID [for debugging now, not shown in demo]: {workflow.id()}')
    syn_data2 = await workflow.datasets().concat(conn)

    local_conn = rf.Connection.local()
    save = ra.DatasetSave(name="sampled_data", concat_tables=True)
    
    sample1 = ra.Sample(sample_size=100, session_key="session_key")
    builder = rf.WorkflowBuilder()
    builder.add_path(syn_data1, sample1, save)
    workflow = await builder.start(local_conn)
    d1 = await workflow.datasets().concat(local_conn)
    
    sample2 = ra.Sample(sample_size=500, session_key="session_key")
    builder = rf.WorkflowBuilder()
    builder.add_path(syn_data2, sample2, save)
    workflow = await builder.start(local_conn)
    d2 = await workflow.datasets().concat(local_conn)

    blended = pa.concat_tables([d1.table, d2.table])

    return blended

# %% [markdown]
# ### Customer-Facing Demo Code

# %%
pd.read_csv("customer_transactions/jan_week2.csv")

# %%
dataset_paths = ["customer_transactions/jan_week1.csv", "customer_transactions/jan_week2.csv"]
session_key = "customer"
metadata_fields = ["age", "gender"]
timestamp = "timestamp"

# %%
stream, workflow = await start_rockfish_runtime(
    dataset_paths, 
    session_key, 
    metadata_fields, 
    timestamp, 
    mask_columns=["email"]
)

# %%
await create_rockfish_models(stream, workflow, dataset_paths)

# %%
blended = await create_blended_dataset(
    workflow, 
    normal_sessions=100, 
    abnormal_sessions=500
)

# %%
