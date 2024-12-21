# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: colab
#     language: python
#     name: python3
# ---

# %% id="7fc3ca7c"
# %%capture
# %pip install -U 'rockfish[labs]' -f 'https://packages.rockfish.ai'

# %% id="3e19aac9"
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

# %% id="9d1e2d31"
# %env ROCKFISH_API_KEY=YOUR_API_KEY
conn = rf.Connection.from_env()

# %% id="a9274584"
# download our example of tabular data: fall_detection.csv
# !wget --no-clobber https://docs.rockfish.ai/tutorials/fall_detection.csv

# %% id="52dce54d"
dataset = rf.Dataset.from_csv("fall_detection", "fall_detection.csv")
dataset.to_pandas()

# %%
# user can manually provide a list of categorical column names
categorical_fields = (
    dataset.to_pandas().select_dtypes(include=["object"]).columns
)
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
        "epochs": 100,
        "records": 2582,
    },
}
# create train action
train = ra.TrainTabGAN(config)

# %% id="1347f248"
builder = rf.WorkflowBuilder()
builder.add_dataset(dataset)
builder.add_action(train, parents=[dataset])
workflow = await builder.start(conn)

print(f"Workflow: {workflow.id()}")

# %% id="0f113aed"
async for progress in workflow.progress().notebook():
    pass

# %% id="8071d86e"
model = await workflow.models().nth(0)
model

# %% id="4eb3c65d"
generate = ra.GenerateTabGAN(config)
save = ra.DatasetSave({"name": "synthetic"})
builder = rf.WorkflowBuilder()
builder.add_model(model)
builder.add_action(generate, parents=[model])
builder.add_action(save, parents=[generate])
workflow = await builder.start(conn)
print(f"Workflow: {workflow.id()}")

# %%
async for log in workflow.logs():
    print(log)

# %% id="689b00b9"
syn = None
async for sds in workflow.datasets():
    syn = await sds.to_local(conn)
syn.to_pandas()

# %% [markdown] id="5f16385a"
# ### Evaluation
#

# %% [markdown] id="31ab12cb"
# **1. categorical columns**
#

# %% id="22fea68f"
for col in ["Age range of patient", "Sex"]:
    source_agg = rf.metrics.count_all(dataset, col, nlargest=10)
    syn_agg = rf.metrics.count_all(syn, col, nlargest=10)
    rl.vis.plot_bar([source_agg, syn_agg], col, f"{col}_count")

# %% [markdown] id="838e59fe"
# **2. numerical columns**
#

# %% id="a39fa972"
for col in ["BBS Score", "Body Temperature"]:
    rl.vis.plot_kde([dataset, syn], col)

# %% [markdown] id="d0ed9a62"
# **3. correlation between numerical columns**
#

# %% id="e69f05b6"
col1 = "SBP"
col2 = "DBP"
rl.vis.plot_correlation([dataset, syn], col1, col2, alpha=0.5)

# %% [markdown] id="6308d280"
# **4. correlation heatmap between several numerical columns**
#

# %% id="a8b7e864"
n_cols = ["Body Temperature", "SBP", "BBS Score", "DBP", "Heart Rate"]
rl.vis.plot_correlation_heatmap([dataset, syn], n_cols, annot=True, fmt=".2f")
