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

# %% [markdown] id="CgJzHzDhcuJI"
# In this tutorial, we will guide you through the process of setting up an end-to-end continuously running workflow for the purposes of continuous ingestion of data.
#
# We will cover the following:
#
# - Preparing your dataset for synthetic data generation.
# - Utilizing Rockfish Recommendation Engine to automatically determine the most suitable model for training, along with key configurations and settings required for successful onboarding.
# - Generating and then evaluating synthetic data using the Rockfish Synthetic Data Assessor, which will help you improve the quality of your synthetic datasets.
# - Setting up an always on workflow using the settings generated from the onboarding process.
# - Applying custom labels to the models that are trained by the workflow.
# - Searching for a previously trained model in Rockfish's model store.
# - Using the model to generate synthetic data.
#

# %% [markdown] id="72cj67zLabYj"
# ### Install and Import Rockfish SDK
#

# %% id="GUWjYJW7Vspw"
# %%capture
# %pip install -U 'rockfish[labs]' -f 'https://docs.rockfish.ai/packages/index.html'

# %% id="I77DF8bPVx8j"
import rockfish as rf
import rockfish.actions as ra
from rockfish.labs.dataset_properties import (
    DatasetPropertyExtractor,
    FieldType,
    EncoderType,
)
from rockfish.labs.steps import Recommender
from rockfish.labs.metrics import marginal_dist_score
from rockfish.labs.sda import SDA

import time

# %% [markdown] id="GBGLOAALaZRt"
# ### Connect to the Rockfish Platform
#
# ❗❗ Replace API_KEY and API_URL.
#

# %% id="_r56lqHPZfBT"
api_key = "API_KEY"

conn = rf.Connection.remote("https://api.rockfish.ai", api_key)

# %% [markdown]
# # 1. Onboard the dataset onto Rockfish
#

# %% [markdown] id="4fg-fmB4apMI"
# ### Load the Dataset
#
# We support ingesting other data formats, refer documentation for more details.
#

# %% id="3foo29nQaf6U"
# %%capture
# !wget --no-clobber https://docs.rockfish.ai/tutorials/finance-1.csv
dataset = rf.Dataset.from_csv("finance", "finance-1.csv")

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="wa2qr_ZIfDrL" outputId="be8c88d2-de90-42f7-89b3-a673e0f7eb16"
dataset.to_pandas()

# %% [markdown] id="dd4qldYKbRo_"
# ### Onboard the dataset onto Rockfish
#
# The onboarding workflow is a good starting point to get to a synthetic version of your dataset quickly.
#
# To ensure optimal synthetic data generation, it's crucial to provide domain-specific information related to your dataset. This helps Rockfish’s Recommendation Engine tailor the workflow to your specific needs.
#

# %% colab={"base_uri": "https://localhost:8080/"} id="PuvZsn7tbbI0" outputId="133f5554-15ca-4c29-dcc4-6938cb6ac00e"
dataset_properties = DatasetPropertyExtractor(
    dataset,
    session_key="customer",
    metadata_fields=["age", "gender"],
    additional_property_keys=["association_rules"],
).extract()
recommender_output = Recommender(dataset_properties).run()
print(recommender_output.report)

# %% [markdown] id="ZQQOTUnxb6XJ"
# #### Run the recommended workflow to get a synthetic dataset
#

# %% colab={"base_uri": "https://localhost:8080/"} id="t4mY164eb9Ic" outputId="f7bcd58f-1d0c-4862-848c-169bee00b49a"
rec_actions = recommender_output.actions
save = ra.DatasetSave({"name": "synthetic"})

# use recommended actions in a Rockfish workflow
builder = rf.WorkflowBuilder()
builder.add_path(dataset, *rec_actions, save)

# run the Rockfish workflow
pre_workflow = await builder.start(conn)
print(f"Workflow: {pre_workflow.id()}")

# %% [markdown] id="Q6bScF8ncLN4"
# View logs for the running workflow:
#

# %% colab={"base_uri": "https://localhost:8080/"} id="_TeT2DFZcKj2" outputId="2c8df097-0640-4e90-caec-1c13fe77f36c"
async for log in pre_workflow.logs():
    print(log)

# %% [markdown] id="TEDizQAScg8J"
# Download and view the synthetic dataset locally:
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="G4_0TVMncghE" outputId="5ac69b4c-d4db-46f8-9f6b-63264954afc3"
syn = await pre_workflow.datasets().last()
syn = await syn.to_local(conn)
syn.to_pandas()

# %% [markdown] id="RmhdoeldtI00"
# ### Evaluate the synthetic dataset
#

# %% cellView="form" id="vmngYsA7cTBo"
# @title ##### Define a helper function `get_fidelity_score()` to calculate the marginal distribution score:

import copy


def get_fidelity_score(source, source_dataset_properties, syn):
    source = copy.deepcopy(source)
    syn = copy.deepcopy(syn)

    columns_to_drop = [source_dataset_properties.session_key]
    source.table = source.table.drop_columns(columns_to_drop)

    columns_to_drop = ["session_key"]
    syn.table = syn.table.drop_columns(columns_to_drop)

    categorical_measurements = source_dataset_properties.filter_fields(
        ftype=FieldType.MEASUREMENT, etype=EncoderType.CATEGORICAL
    )

    return marginal_dist_score(
        source,
        syn,
        metadata=source_dataset_properties.metadata_fields,
        other_categorical=categorical_measurements,
    )


# %% colab={"base_uri": "https://localhost:8080/"} id="oNMp4sEeq2oT" outputId="6e2a69f5-5104-4692-c4ab-e02d6f08a114"
get_fidelity_score(
    source=dataset, source_dataset_properties=dataset_properties, syn=syn
)

# %% [markdown]
# ### Since the actions look good, we can use them for setting up the always-on workflow.
#

# %%
rec_actions

# %%
train_actions = rec_actions[:-1]
generate_actions = rec_actions[-1:]

# %% [markdown]
# # 2. Set up an always-on workflow for continuous data ingestion
#

# %% [markdown]
# ### Employ the DataStreamLoad action to keep the workflow always on
#

# %%
# reduce the batch size for the following ingested data stream as the batch
# size should be smaller than the number of sessions in the dataset
train_actions[0].config().doppelganger.batch_size = 14
stream = ra.DatastreamLoad()

builder = rf.WorkflowBuilder()
builder.add(stream, alias="input")
builder.add_path(*train_actions, parents=["input"], alias="train_actions")
ingest_workflow = await builder.start(conn)
print(f"Ingestion Workflow ID: {ingest_workflow.id()}")

# %% [markdown]
# ### Write the data files to the workflow stream
#
# - each input is a dataset
# - each output is a trained model stored to the model_store
#

# %% [markdown]
# ### Write data files to the workflow stream
#
# Replace the workflow ID with the actual workflow ID of the workflow that was set up
#

# %% [markdown]
# ### Download the sample files for the datastream workflow
#

# %%
# %%capture
# !wget --no-clobber https://docs.rockfish.ai/tutorials/finance-2.csv
# !wget --no-clobber https://docs.rockfish.ai/tutorials/finance-3.csv
# !wget --no-clobber https://docs.rockfish.ai/tutorials/finance-4.csv

# %% [markdown]
# ### Replace the workflow ID with the ID of the workflow that was just set up.
#
# This also allows you to run the data-ingestion service in an independent process.
#

# %%
# Retrieve the workflow with the previous ID without need to re-build the workflow
workflow_id = ingest_workflow.id()  # insert workflow ID here
workflow = await conn.get_workflow(workflow_id)

# %%
for file_num in range(2, 5):
    data = rf.Dataset.from_csv("finance", f"finance-{file_num}.csv")
    await workflow.write_datastream(
        "input", data
    )  # "input" is the pre-set alias of the datastream
    print(f"Writing finance-{file_num} to datastream...")
    time.sleep(10)
await workflow.close_datastream(
    "input"
)  # "input" is the pre-set alias of the datastream

# %%
# check the status of the workflow
async for log in workflow.logs():
    print(log)

# %% [markdown]
#

# %% [markdown]
# ### Optional: Add custom labels to the models that are generated
#
# These labels can be used later to filter models based off custom parameters
#

# %%
usage = ["experimental", "staging", "production"]
i = 0
async for model in conn.list_models(labels={"workflow_id": workflow_id}):
    await model.add_labels(conn, usage=usage[i])
    i += 1

# %% [markdown]
# # 3. Generate synthetic data using the trained model
#

# %% [markdown]
# ### Provide query params to the model_store search to get appropriate models as response
#
# This can be used if the models trained were previously tagged, the default label that exists is 'workflow_id' which is the id of the workflow that trained the model
#

# %%
async for model in conn.list_models(labels={"usage": "production"}):
    print(model)

# %% [markdown]
# ### Select a model from the list of queried models and fetch it from remote
#

# %%
model = await rf.Model.from_id(
    conn,
    model.id,  # insert model id here of the filtered model after querying
)
print(model)

# %% [markdown]
# ### Provide the model and the synthesis config to a workflow to generate a synthetic dataset as the output
#

# %%
builder = rf.WorkflowBuilder()
builder.add(model)
builder.add(*generate_actions, parents=[model], alias="gen")
builder.add(ra.DatasetSave(name="syn_data"), parents=["gen"])
workflow = await builder.start(conn)
print(f"Workflow ID: {workflow.id()}")

# %%
async for log in workflow.logs():
    print(log)

# %%
syn_data = await workflow.datasets().concat(conn)
syn_data.to_pandas()
