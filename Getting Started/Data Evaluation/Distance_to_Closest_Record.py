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

# %% id="bRpmkjihaIC3"
# %%capture
# %pip install scikit-learn
# %pip install -U 'rockfish[labs]' -f 'https://docs.rockfish.ai/packages/index.html'

# %% id="YU20KUZPaRSx"
import rockfish as rf
import rockfish.actions as ra
import rockfish.labs as rl
import pandas as pd
from sklearn.model_selection import train_test_split

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

# %% id="L8Psm1L0aiL9"
# %env ROCKFISH_API_KEY=YOUR_API_KEY
conn = rf.Connection.from_env()

# %% id="RHuXlcTIaq8h"
# download our example of tabular data: fall_detection.csv
# !wget --no-clobber https://docs.rockfish.ai/tutorials/fall_detection.csv

# %% id="jQDT1KtoauUF"
# split into train and test dataset
df = pd.read_csv("fall_detection.csv")
train_split, test_split = train_test_split(
    df, test_size=0.5, shuffle=True, random_state=1
)

# reset and drop original indices for both splits
train_split = train_split.reset_index(drop=True)
test_split = test_split.reset_index(drop=True)

# %% id="KxyFFoIjavf7"
train_dataset = rf.Dataset.from_pandas("fall_detection_train", train_split)
train_dataset.to_pandas()

# %% id="W0ivJLSSayVp"
test_dataset = rf.Dataset.from_pandas("fall_detection_test", test_split)
test_dataset.to_pandas()

# %%
# user can manually provide a list of categorical column names
categorical_fields = (
    train_dataset.to_pandas().select_dtypes(include=["object"]).columns
)
config = {
    "encoder": {
        "metadata": [
            {"field": field, "type": "categorical"}
            for field in categorical_fields
        ]
        + [
            {"field": field, "type": "continuous"}
            for field in train_dataset.table.column_names
            if field not in categorical_fields
        ],
    },
    "tabular-gan": {
        "epochs": 100,
        "records": len(train_split),
    },
}
# create train action
train = ra.TrainTabGAN(config)

# %% id="yVi5nTrOa1cR"
builder = rf.WorkflowBuilder()
builder.add_dataset(train_dataset)
builder.add_action(train, parents=[train_dataset])
workflow = await builder.start(conn)

print(f"Workflow: {workflow.id()}")

# %% id="WkmHTWD2a27X"
async for progress in workflow.progress().notebook():
    pass

# %% id="9RCyff_Xa4iU"
model = await workflow.models().nth(0)
model

# %% id="DBZqri-Ca6Ky"
generate = ra.GenerateTabGAN(config)
save = ra.DatasetSave({"name": "synthetic"})
builder = rf.WorkflowBuilder()
builder.add_model(model)
builder.add_action(generate, parents=[model])
builder.add_action(save, parents=[generate])
workflow = await builder.start(conn)
print(f"Workflow: {workflow.id()}")

# %% id="mW4w9xTEa6lP"
syn = None
async for sds in workflow.datasets():
    syn = await sds.to_local(conn)
syn.to_pandas()

# %% [markdown] id="9AkuaHH_a_SJ"
# ### DCR Score
#
# The Distance to Closest Record (DCR) score quantifies privacy risk by checking how similar records in the synthetic
# dataset are w.r.t. the source dataset.
#
# It does so by measuring the similarity between the DCR distributions between the two dataset pairs - (source, synthetic)
# and (source, test). The more similar these two DCR distributions are, the more "private" the synthetic data.
#
# Note that the test dataset should be sampled from the same distribution as the source dataset, and should not be used to
# train your synthetic data generator.
#
# The DCR score is a value between 0 and positive infinity. It can be interpreted using the following Likert scale for
# quality:
#
# 1. Low: [0 - 0.75)
# 2. Medium: [0.75 - 1.0)
# 3. High: [1.0, positive infinity)
#

# %% id="5OhN_sc-bAAg"
score = rl.metrics.distance_to_closest_record_score(
    train_dataset=train_dataset, test_dataset=test_dataset, syn=syn
)

# %% id="8ykThPlRbMeG"
score
