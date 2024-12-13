# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% id="64c0d8a8"
# %%capture
# %pip install -U 'rockfish[labs]' -f 'https://docs.rockfish.ai/packages/index.html'

# %% id="71b6977e"
import rockfish as rf
import rockfish.actions as ra
import rockfish.labs as rl

# %% [markdown]
# Please replace `YOUR_API_KEY` with the assigned API key string. Note that it should be without quotes.
#
# For example, if the assigned API Key is `abcd1234`, you can do the following
#
# ```python
# %env ROCKFISH_API_KEY=abcd1234
# conn = rf.Connection.from_env()
# ```
#
# If you do not have API Key, please reach out to support@rockfish.ai.
#

# %% id="712ac74c"
# %env ROCKFISH_API_KEY=YOUR_API_KEY
conn = rf.Connection.from_env()

# %% id="9828d37e"
# download our example of timeseries data: finance.csv
# !wget --no-clobber https://docs.rockfish.ai/tutorials/finance.csv

# %% id="5b9f8764"
dataset = rf.Dataset.from_csv("finance", "finance.csv")
dataset.to_pandas()

# %% id="248b1e82"
config = ra.TrainTimeGAN.Config(
    encoder=ra.TrainTimeGAN.DatasetConfig(
        timestamp=ra.TrainTimeGAN.TimestampConfig(field="timestamp"),
        metadata=[
            ra.TrainTimeGAN.FieldConfig(field="customer", type="session"),
            ra.TrainTimeGAN.FieldConfig(field="age", type="categorical"),
            ra.TrainTimeGAN.FieldConfig(field="gender", type="categorical"),
        ],
        measurements=[
            ra.TrainTimeGAN.FieldConfig(field="merchant", type="categorical"),
            ra.TrainTimeGAN.FieldConfig(field="category", type="categorical"),
            ra.TrainTimeGAN.FieldConfig(field="amount"),
            ra.TrainTimeGAN.FieldConfig(field="fraud", type="categorical"),
        ],
    ),
    doppelganger=ra.TrainTimeGAN.DGConfig(
        sample_len=19,
        epoch=10,
        epoch_checkpoint_freq=10,
        batch_size=64,
        sessions=3765,
    ),
)

# create train action
train = ra.TrainTimeGAN(config)

# %% id="14dc3fc5"
builder = rf.WorkflowBuilder()
builder.add_dataset(dataset)
builder.add_action(train, parents=[dataset])
workflow = await builder.start(conn)
print(f"Workflow: {workflow.id()}")

# %% id="99d8eb87"
async for progress in workflow.progress().notebook():
    pass

# %% id="28eb4982"
model = await workflow.models().last()
model

# %% id="e0e74ad3"
generate = ra.GenerateTimeGAN(config)
save = ra.DatasetSave({"name": "synthetic"})
builder = rf.WorkflowBuilder()
builder.add_model(model)
builder.add_action(generate, parents=[model])
builder.add_action(save, parents=[generate])
workflow = await builder.start(conn)
print(f"Workflow: {workflow.id()}")

# %% id="f776b814"
async for progress in workflow.progress().notebook():
    pass

# %% id="e7f0c1a7"
syn = None
async for sds in workflow.datasets():
    syn = await sds.to_local(conn)
syn.to_pandas()

# %% [markdown] id="d49b9a01"
# ### Evaluation
#

# %% id="ad5070ea"
source_data = rf.Dataset.from_pandas("source", dataset.to_pandas())
syn_data = rf.Dataset.from_pandas(
    "rf-dg", syn.to_pandas().rename(columns={"session_key": "customer"})
)
schema_metadata = rf.arrow.SchemaMetadata(
    metadata=["customer", "age", "gender"]
)
source_data.table = rf.arrow.replace_schema_metadata(
    source_data.table, schema_metadata
)
syn_data.table = syn_data.table.select(source_data.table.column_names)
syn_data.table = rf.arrow.replace_schema_metadata(
    syn_data.table, schema_metadata
)

syn_data.table = syn_data.table.cast(source_data.table.schema, safe=False)

# %% [markdown] id="e428ad7c"
# **1. session length**
#

# %% id="a8bb1516"
source_sess = rf.metrics.session_length(source_data)
syn_sess = rf.metrics.session_length(syn_data)
rf.labs.vis.plot_kde([source_sess, syn_sess], "session_length")

# %% [markdown] id="1b59cf6f"
# **2. interarrival time**
#
# Here the unit for duration is in seconds
#

# %% id="e5a2fabd"
timestamp = "timestamp"
source_interarrival = rf.metrics.interarrivals(source_data, timestamp)
syn_interarrival = rf.metrics.interarrivals(syn_data, timestamp)
rf.labs.vis.plot_kde(
    [source_interarrival, syn_interarrival], "interarrival", duration_unit="s"
)

# %% [markdown] id="8a7422b2"
# **3. numerical columns**
#

# %% id="5ba21ea6"
rf.labs.vis.plot_kde([source_data, syn_data], "amount")

# %% [markdown] id="87535dfb"
# **4. categorical columns**
#
# If there is a large categorical column containing over 10 categories, we plot the Top10 bars for users to compare.
#
# NB. If you want to show more than Top10, update the number of `nlargest`
#

# %% id="0658944c"
for col in ["age", "gender", "merchant", "category", "fraud"]:
    source_agg = rf.metrics.count_all(dataset, col, nlargest=10)
    syn_agg = rf.metrics.count_all(syn_data, col, nlargest=10)
    rf.labs.vis.plot_bar([source_agg, syn_agg], col, f"{col}_count")
