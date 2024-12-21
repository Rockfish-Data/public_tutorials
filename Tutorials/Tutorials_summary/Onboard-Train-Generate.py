# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# INSTALL ROCKFISH SDK
#

# %%
# GENERATE SYNTHETIC DATASET USING ROCKFISH

# %pip install -U 'rockfish[labs]' -f 'https://packages.rockfish.ai'
# %restart_python

# %% [markdown]
# SETUP ROCKFISH ENVIRONMENT VARIABLES
#

# %%
# Rockfish Environment Key & API URL
import rockfish as rf
import rockfish.actions as ra
import rockfish.labs as rl
#api_key = "<ENTER YOUR API_KEY HERE>"
#api_url = "<ENTER YOUR API_URL HERE>"
# %env ROCKFISH_API_KEY=api_key
conn = rf.Connection.remote(api_url, api_key)

# %% [markdown]
# Read Input Data into a dataframe.
#

# %%
import pandas as pd

# Read a CSV file
df = pd.read_csv("<PATH_TO_SAMPLE_DATA_CSV>")

# %% [markdown]
# GENERATE SYNTHETIC DATA **Onboard Train and Generate** using **Rockfish GenAI** Models.
#

# %%
# Onboard
# Perform any necessary feature engineering or preprocessing
dataset = rf.Dataset.from_pandas("<NAME_OF_DATASET>", df)

categorical_fields = df.select_dtypes(include=["object"]).columns
print(categorical_fields)

config = {
    "encoder": {
        "metadata": [
            {"field": field, "type": "categorical"}
            for field in categorical_fields
        ]
        + [
            {"field": field, "type": "continuous"}
            for field in dataset.table.column_names
            if field not in categorical_fields
        ],
    },
    "tabular-gan": {
        "epochs": 20,
        "records": 100000,
    },
}
print(dataset.table.column_names)

# %%
# Train
train = ra.TrainTabGAN(config)

builder = rf.WorkflowBuilder()
builder.add_dataset(dataset)
builder.add_action(train, parents=[dataset])
workflow = await builder.start(conn)
print(f"Workflow: {workflow.id()}")

async for log in workflow.logs():
    print(log)

model = await workflow.models().nth(0)
await model.add_labels(conn)
model

# %%
# Generate
generate = ra.GenerateTabGAN(config)
save = ra.DatasetSave({"name": "synthetic"})
builder = rf.WorkflowBuilder()
builder.add_model(model)
builder.add_action(generate, parents=[model])
builder.add_action(save, parents=[generate])
workflow = await builder.start(conn)
print(f"Workflow: {workflow.id()}")

syn = None
async for sds in workflow.datasets():
    syn = await sds.to_local(conn)

# %% [markdown]
# Synthetic Data Assessor
#

# %%
for col in dataset.table.column_names:
    source_agg = rf.metrics.count_all(dataset, col, nlargest=10)
    syn_agg = rf.metrics.count_all(syn, col, nlargest=10)
    rl.vis.plot_bar([source_agg, syn_agg], col, f"{col}_count")
