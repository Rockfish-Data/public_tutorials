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

# %% [markdown] id="MCBbARVwh_db"
# # Pre Generation Filtering
#
# Inputs:
#
# 1. `model`: Trained rf-time-gan model
# 2. `given_metadata`: Conditions with metadata values that sessions should have (e.g., all sessions should have `gender = "M"`)
#
# Output:
#
# 1. `dataset`: Synthetic dataset with sessions whose metadata fields match the values in `given_metadata`
#

# %% [markdown] id="cGXiiEscl2Dj"
# ### Install and Import Rockfish SDK
#

# %% id="bC6avaqHmN8O"
# %%capture
# !pip install -U 'rockfish[labs]' -f 'https://docs.rockfish.ai/packages/index.html' 

# %% id="cKHQAhEal_Gw"
import rockfish as rf
import rockfish.actions as ra
import rockfish.labs as rl
import pandas as pd

# %% [markdown] id="Coh2p6X9l7U2"
# ### Connect to the Rockfish Platform
#

# %% id="p2dclvL7mPHz"
api_key = ""  # API key for environment
api_url = ""  # URL for environment
conn = rf.Connection.remote(api_url, api_key)

# %% [markdown] id="b5QfMd72Vclu"
# ### Train Rf-Time-GAN on the Finance Dataset
#

# %% id="Cn0spZWWmdL8"
# !wget --no-clobber https://docs.rockfish.ai/tutorials/finance.csv
dataset = rf.Dataset.from_csv("finance", "finance.csv")


# %% id="4u1emyNCmCYd"
def get_config(
    epochs=1, sample_len=2, sessions=100, batch_size=512, given_metadata=None
):
    timestamp = "timestamp"
    session = "customer"
    metadata_cols = ["age", "gender"]
    con_measurement_cols = ["amount"]
    cat_measurement_cols = ["merchant", "category", "fraud"]
    config = ra.TrainTimeGAN.Config(
        encoder=ra.TrainTimeGAN.DatasetConfig(
            timestamp=ra.TrainTimeGAN.TimestampConfig(field=timestamp),
            metadata=[
                ra.TrainTimeGAN.FieldConfig(field=col, type="categorical")
                for col in metadata_cols
            ]
            + [ra.TrainTimeGAN.FieldConfig(field=session, type="session")],
            measurements=[
                ra.TrainTimeGAN.FieldConfig(field=col, type="continuous")
                for col in con_measurement_cols
            ]
            + [
                ra.TrainTimeGAN.FieldConfig(field=col, type="categorical")
                for col in cat_measurement_cols
            ],
        ),
        doppelganger=ra.TrainTimeGAN.DGConfig(
            epoch=epochs,
            epoch_checkpoint_freq=epochs,
            sample_len=sample_len,
            sessions=sessions,
            batch_size=batch_size,
            given_metadata=given_metadata,
        ),
    )
    return config


# %% colab={"base_uri": "https://localhost:8080/"} id="6FE0MmdKmoZ9" outputId="0df4a561-d70d-4e5c-c2f3-54560d10f6dc"
train_config = get_config()  # pass appropriate args for setting hyperparams
train = ra.TrainTimeGAN(train_config)

builder = rf.WorkflowBuilder()
builder.add_path(dataset, train)
workflow = await builder.start(conn)
print(f"Train Workflow: {workflow.id()}")

# %% colab={"base_uri": "https://localhost:8080/"} id="YqfnMnk9nzb7" outputId="26d78404-93b1-48c0-b66d-6d4f1bf3e52f"
async for log in workflow.logs(level=rf.events.LogLevel.DEBUG):
    print(log)

# %% [markdown] id="YgMhz1Z9Eh9I"
# ### Fetch the Trained Model
#

# %% id="poovJ-LLEdg9"
model = await workflow.models().last()

# %% [markdown] id="mrqZX6vCVheE"
# ### Precondition While Generating
#

# %% [markdown] id="w256JU4BVwHN"
# #### Example 1: One Condition
#
# **User Intent**
#
# Generate the following synthetic dataset: 25 sessions with metadata (age = 4, gender = M).
#

# %% [markdown] id="VAeKpyS4DRjk"
# ##### Specify these conditions in the generate config
#

# %% id="QyAZpKJ5mxAv"
# input: given_metadata = [(age = 4, gender = M)]
# expected output: 25 sessions with metadata = (age = 4, gender = M)
given_metadata1 = {"age": ["4"], "gender": ["M"]}
generate1_config = get_config(sessions=25, given_metadata=given_metadata1)
generate1 = ra.GenerateTimeGAN(generate1_config)

# %% [markdown] id="uyjRS10ADWTk"
# ##### Run the generate workflow
#

# %% colab={"base_uri": "https://localhost:8080/"} id="LPsW5ImcWCeM" outputId="095a94c5-04f4-499c-a911-dd984b144533"
save = ra.DatasetSave(name="synthetic")
builder = rf.WorkflowBuilder()
builder.add_path(model, generate1, save)
workflow = await builder.start(conn)
print(f"Generate Workflow: {workflow.id()}")

# %% id="9UJnBm3GWL3Z"
async for log in workflow.logs(level=rf.events.LogLevel.DEBUG):
    print(log)

# %% id="k0X3s6VLWNm3"
syn = None
async for sds in workflow.datasets():
    syn = await sds.to_local(conn)

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="P7Z7ajT2WPjI" outputId="aaeb74c7-1e9b-42cc-aa56-3e78f6a3acea"
syn.to_pandas()

# %% [markdown] id="U-y2NMWvhx0X"
# ##### Validate metadata fields are constrained to given_metadata:
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 519} id="7rKRg7UYLN04" outputId="f11265fa-e784-4fb5-80b1-752285621bb1"
# plot age
rl.vis.plot_bar(datasets=[syn], field="age")

# %% colab={"base_uri": "https://localhost:8080/", "height": 519} id="6zRZWNm5eWqp" outputId="7df4a9ca-8fc2-4f97-b47f-48433738996c"
# plot gender
rl.vis.plot_bar(datasets=[syn], field="gender")

# %% [markdown] id="UAXMaAmcV04n"
# #### Example 2: Multiple Conditions, Controlling Number of Sessions
#
# **User Intent**
#
# Generate the following synthetic dataset (using the same model):
#
# 1. 25 sessions with metadata (age = 4, gender = M)
# 2. 50 sessions with metadata either (age = 2, gender = F) or (age = 5, gender = M)
#

# %% [markdown] id="hS6h14tHC7f-"
# ##### Specify these conditions in the generate config
#

# %% id="eDehCdwEmykP"
# given_metadata = [(age = 2, gender = F), (age = 5, gender = M)]
# expected output: 50 sessions with metadata that can be either
# (age = 2, gender = F) or (age = 5, gender = M).
given_metadata2 = {"age": ["2", "5"], "gender": ["F", "M"]}
generate2_config = get_config(sessions=50, given_metadata=given_metadata2)
generate2 = ra.GenerateTimeGAN(generate2_config)

# %% [markdown] id="zN9W_FNODD_k"
# ##### Run the generate workflow
#

# %% colab={"base_uri": "https://localhost:8080/"} id="LoKKazcDnX67" outputId="bad83b7d-c7be-4493-85f7-e32b5848bc2f"
save = ra.DatasetSave(
    name="synthetic", concat_tables=True, concat_session_key="session_key"
)

# this shows how you can add two different generate actions to the same workflow
builder = rf.WorkflowBuilder()
builder.add(model)
builder.add(
    generate1, parents=[model]
)  # re-using generate action from example 1
builder.add(
    generate2, parents=[model]
)  # using generate action from example 2
builder.add(save, parents=[generate1, generate2])
workflow = await builder.start(conn)
print(f"Generate Workflow: {workflow.id()}")

# %% id="ElmuvVyGnboW"
async for log in workflow.logs(level=rf.events.LogLevel.DEBUG):
    print(log)

# %% id="cmICaybvrlR2"
syn = None
async for sds in workflow.datasets():
    syn = await sds.to_local(conn)

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="-ZhUm7YZnc3T" outputId="ca9b795e-9ef1-4b46-b590-4105d7263b3a"
syn.to_pandas()

# %% [markdown] id="JIl40FuXh62k"
# ##### Validate metadata fields are constrained to given_metadata:
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 519} id="x-08XE0nLJwI" outputId="2eb81505-fa91-421d-fe8d-72c787ecd9da"
# plot age
rl.vis.plot_bar(datasets=[syn], field="age")

# %% colab={"base_uri": "https://localhost:8080/", "height": 519} id="XV3t7_jXeaw7" outputId="5439c1fb-12d0-4d80-bbe2-ed714195f50a"
# plot gender
rl.vis.plot_bar(datasets=[syn], field="gender")
