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

# %% id="f4a27814"
# %%capture
# %pip install -U 'rockfish[labs]' -f 'https://docs.rockfish.ai/packages/index.html'

# %% id="057c8083"
import rockfish as rf
import rockfish.actions as ra
import rockfish.labs

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

# %% id="90c10b31"
# %env ROCKFISH_API_KEY=YOUR_API_KEY
conn = rf.Connection.from_env()

# %% id="d11e3e80"
# download our example of tabular data: spotify-2023-short.csv
# !wget --no-clobber https://docs.rockfish.ai/tutorials/spotify-2023-short.csv

# %% id="4c6006b0"
dataset = rf.Dataset.from_csv("Spotify", "spotify-2023-short.csv")
dataset.to_pandas()

# %% id="13a2d7b6"
cat_fields = "released_year released_month released_day key mode".split()
con_fields = "in_spotify_playlists bpm".split()
config = {
    "encoder": {
        "metadata": [
            {"field": col, "type": "categorical"} for col in cat_fields
        ]
        + [{"field": col, "type": "continuous"} for col in con_fields]
    },
    "rtf": {
        "mode": "tabular",
        "num_bootstrap": 2,
        "tabular": {
            "epochs": 1,
            "transformer": {
                "gpt2_config": {"layer": 1, "head": 1, "embed": 1}
            },
        },
    },
}
# create the train action
train = ra.TrainTransformer(config)

# %% id="8752390a"
builder = rf.WorkflowBuilder()
builder.add_dataset(dataset)
builder.add_action(train, parents=[dataset])
workflow = await builder.start(conn)

print(f"Workflow: {workflow.id()}")

# %% id="ee6af284"
async for log in workflow.logs():
    print(log)

# %% id="b1a15f54"
model = await workflow.models().last()
model

# %% id="1176843c"
generate = ra.GenerateTransformer(config)
save = ra.DatasetSave({"name": "synthetic"})
builder = rf.WorkflowBuilder()
builder.add_model(model)
builder.add_action(generate, parents=[model])
builder.add_action(save, parents=[generate])
workflow = await builder.start(conn)
print(f"Workflow: {workflow.id()}")

# %% id="af684dd4"
async for log in workflow.logs():
    print(log)

# %% id="a2b4f4a9"
syn = None
async for sds in workflow.datasets():
    syn = await sds.to_local(conn)
syn.table = syn.table.select(
    "released_year released_month released_day in_spotify_playlists bpm key mode".split()
)
syn.to_pandas()

# %% [markdown] id="a0965393"
# ### Evaluation
#
# **1. Categorical columns**
#

# %% id="ac84b683"
for col in ["released_year", "mode"]:
    source_agg = rf.metrics.count_all(dataset, col, nlargest=10)
    syn_agg = rf.metrics.count_all(syn, col, nlargest=10)
    rf.labs.vis.plot_bar([source_agg, syn_agg], col, f"{col}_count")

# %% [markdown] id="54b0ef44"
# **2. numerical columns**
#

# %% id="c994c08b"
for col in ["in_spotify_playlists", "bpm"]:
    rf.labs.vis.plot_kde([dataset, syn], col)
