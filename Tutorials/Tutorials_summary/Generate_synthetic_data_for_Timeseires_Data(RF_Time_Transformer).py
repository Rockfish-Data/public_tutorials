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

# %% id="hgVdjEBIGUub"
# %%capture
# %pip install -U 'rockfish[labs]' -f 'https://docs.rockfish.ai/packages/index.html'

# %% id="lplEGc_Syjdf"
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

# %% id="zfprA7dZyjaw"
# %env ROCKFISH_API_KEY=YOUR_API_KEY
conn = rf.Connection.from_env()

# %% colab={"base_uri": "https://localhost:8080/"} id="22x8RaE4yjYS" outputId="21e9a929-8f71-4dd2-960b-ee004aa2292a"
# download our example of timeseries data: pcap.csv
# !wget --no-clobber https://docs.rockfish.ai/tutorials/pcap.csv

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="rLqODOndyjVo" outputId="6adb87b6-96c8-487a-f83d-c8a7cec373d7"
dataset = rf.Dataset.from_csv("DC pcap", "pcap.csv")
dataset.to_pandas()

# %%
config = {
    "encoder": {
        "timestamp": {"field": "timestamp"},
        "metadata": [
            {"field": "srcip", "type": "categorical"},
            {"field": "dstip", "type": "categorical"},
            {"field": "srcport", "type": "categorical"},
            {"field": "dstport", "type": "categorical"},
            {"field": "proto", "type": "categorical"},
        ],
        "measurements": [{"field": "pkt_len", "type": "continuous"}],
    },
    "rtf": {
        "mode": "relational",
        "num_bootstrap": 2,
        "parent": {
            "epochs": 1,
            "transformer": {
                "gpt2_config": {"layer": 1, "head": 1, "embed": 1}
            },
        },
        "child": {"output_max_length": 2048, "epochs": 1},
    },
}
# create train action
train = ra.TrainTransformer(config)

# %% colab={"base_uri": "https://localhost:8080/"} id="h2_N4q1ny6TD" outputId="d6e35d90-8a3b-416f-ee45-e07f449152d2"
builder = rf.WorkflowBuilder()
builder.add_dataset(dataset)
builder.add_action(train, parents=[dataset])
workflow = await builder.start(conn)
print(f"Workflow: {workflow.id()}")

# %% colab={"base_uri": "https://localhost:8080/"} id="jSNt3HEfy6Qa" outputId="db53c24b-079e-40f2-d972-cf618964fc97"
async for log in workflow.logs():
    print(log)

# %% colab={"base_uri": "https://localhost:8080/"} id="dutE6-cLy6OF" outputId="8c44feaf-d542-4b0d-c33b-748836afbdb7"
model = await workflow.models().last()
model

# %% colab={"base_uri": "https://localhost:8080/"} id="EmFFNTRwy6LF" outputId="3958a697-958f-4171-a901-001bbffc08f8"
generate = ra.GenerateTransformer(config)
save = ra.DatasetSave({"name": "Synthetic"})
builder = rf.WorkflowBuilder()
builder.add_model(model)
builder.add_action(generate, parents=[model])
builder.add_action(save, parents=[generate])
workflow = await builder.start(conn)
print(f"Workflow: {workflow.id()}")

# %% colab={"base_uri": "https://localhost:8080/"} id="5_riQ8Iuy6Iv" outputId="82db41e5-6f62-4497-8955-a98dd8f69e2d"
async for log in workflow.logs():
    print(log)

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="ifeaOMP6zXT3" outputId="b2e11bdb-918d-4409-e242-bcb8ebcac665"
syn = None
async for sds in workflow.datasets():
    syn = await sds.to_local(conn)
syn.to_pandas()

# %% [markdown] id="5b8-_h-rzunG"
# # Evaluate the synthetic dataset
#

# %% id="qRk9lD5MzXQz"
source_data = rf.Dataset.from_pandas("source", dataset.to_pandas())
syn_data = rf.Dataset.from_pandas(
    "rt-rtf-ts",
    syn.to_pandas()[
        "srcip dstip srcport dstport proto timestamp pkt_len".split()
    ],
)
schema_metadata = rf.arrow.SchemaMetadata(
    metadata="srcip dstip srcport dstport proto".split()
)
source_data.table = rf.arrow.replace_schema_metadata(
    source_data.table, schema_metadata
)
syn_data.table = rf.arrow.replace_schema_metadata(
    syn_data.table, schema_metadata
)
syn_data.table = syn_data.table.cast(source_data.table.schema)

# %% [markdown] id="AYKdYxQRz5DB"
# **Numerical columns**
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 501} id="hD8iKjTUzXLe" outputId="8f14adec-1d3e-4420-ac1f-926fe020aec8"
col = "pkt_len"
rf.labs.vis.plot_kde([source_data, syn_data], col)

# %% [markdown] id="iuy9lvU00T8A"
# **Categorical columns**
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="016Kj3kh0TbM" outputId="c89791b1-39cc-40ef-d9b7-40fbe34f89ed"
for col in ["proto", "srcport"]:
    source_agg = rf.metrics.count_all(source_data, col, nlargest=5)
    syn_agg = rf.metrics.count_all(syn_data, col, nlargest=5)
    rf.labs.vis.plot_bar([source_agg, syn_agg], col, f"{col}_count")
