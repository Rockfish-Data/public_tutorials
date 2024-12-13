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

# %% [markdown] id="CgJzHzDhcuJI"
# In this tutorial, we will guide you through the process of onboarding a dataset for synthetic data generation using the Rockfish Onboarding Module.
#
# We will cover the following:
#
# - Preparing your dataset for synthetic data generation.
# - Utilizing Rockfish Recommendation Engine to automatically determine the most suitable model for training, along with key configurations and settings required for successful onboarding.
# - Generating and then evaluating synthetic data using the Rockfish Synthetic Data Assessor, which will help you improve the quality of your synthetic datasets.
#

# %% [markdown] id="72cj67zLabYj"
# ### Install and Import Rockfish SDK
#

# %% id="GUWjYJW7Vspw"
# %%capture
# %pip install -U 'rockfish[labs]' -f 'https://packages.rockfish.ai'

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

# %% [markdown] id="GBGLOAALaZRt"
# ### Connect to the Rockfish Platform
#
# ❗❗ Replace API_KEY and API_URL.
#

# %% id="_r56lqHPZfBT"
api_key = "API_KEY"
api_url = "API_URL"

conn = rf.Connection.remote(api_url, api_key)

# %% [markdown] id="4fg-fmB4apMI"
# ### Load the Dataset
#
# We support ingesting other data formats, refer documentation for more details.
#

# %% id="3foo29nQaf6U"
# %%capture
# !wget --no-clobber https://docs.rockfish.ai/tutorials/finance.csv
dataset = rf.Dataset.from_csv("finance", "finance.csv")

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="wa2qr_ZIfDrL" outputId="fcb0f460-c042-41af-98d0-bc6a2815479c"
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
workflow = await builder.start(conn)
print(f"Workflow: {workflow.id()}")

# %% [markdown] id="Q6bScF8ncLN4"
# View logs for the running workflow:
#

# %% colab={"base_uri": "https://localhost:8080/"} id="_TeT2DFZcKj2" outputId="2c8df097-0640-4e90-caec-1c13fe77f36c"
async for log in workflow.logs():
    print(log)

# %% [markdown] id="TEDizQAScg8J"
# Download and view the synthetic dataset locally:
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="G4_0TVMncghE" outputId="5ac69b4c-d4db-46f8-9f6b-63264954afc3"
syn = None
async for sds in workflow.datasets():
    syn = await sds.to_local(conn)
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

# %% [markdown] id="XJItkMnpdJ-e"
# ### Next Steps
#
# As you just saw, the onboarding workflow is a good starting point to get to a synthetic dataset quickly.
#
# You can now modify this workflow according to your requirements to get your final synthetic dataset!
#
# The following pages in the Rockfish documentation will be useful for this purpose:
#
# 1. Adding more steps (i.e. Rockfish actions) to a Rockfish workflow: https://docs142.rockfish.ai/sdk-overview.html#actions-and-workflows
# 2. Hyperparameters you can change to improve the performance of Rockfish models: https://docs142.rockfish.ai/models.html
# 3. Using more metrics and plots to evaluate your synthetic dataset: https://docs142.rockfish.ai/data-eval.html
#
