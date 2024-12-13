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

# %% id="77XWYICHBy5A"
# %%capture
# %pip install -U 'rockfish[labs]' -f 'https://docs.rockfish.ai/packages/index.html'

# %% id="Y4ZunWxA-9qr"
import io
import rockfish as rf
import rockfish.actions as ra

# %% [markdown] id="zXyieApXVTkX"
# ## Fill missing values
#
# 1. fill by indicated value
# 2. fill by previous value
# 3. fill by next value
# 4. fill by its mean
# 5. fill by its median
#

# %% id="12r2Tdma-9h7"
# create a dataset with missing value
data = b"""\
a,b,c
1,2,3
4,5,6
,7,8
9,0,1
"""

dataset = rf.Dataset.from_csv("nulls", io.BytesIO(data))

# %% [markdown] id="_2_VuqoUGMxv"
# ### 1. fill missing values by the indicated value
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 175} id="T9cfTBQ4Vqyo" outputId="eec5edcd-a27a-462a-cceb-2a6779dccdb1"
dataset.to_pandas()

# %% id="Gy8nB7WxT5Pf"
conn = rf.Connection.local()

# %% id="zMM7N7foH8_M"
fill_value = 42
fill_col = "a"
fill_null = ra.Transform({"function": {"fill_null": [fill_col, fill_value]}})

# %% colab={"base_uri": "https://localhost:8080/"} id="EBlEpkADH84S" outputId="b879d5b8-4851-456d-a09d-027bc9a5eb32"
save = rf.actions.DatasetSave(name="fill_value_dataset")
builder = rf.WorkflowBuilder.local()
builder.add_dataset(dataset)
builder.add_action(fill_null, parents=[dataset])
builder.add_action(save, parents=[fill_null])
workflow = await builder.start(conn)
print(f"Workflow: {workflow.id()}")

# %% colab={"base_uri": "https://localhost:8080/", "height": 175} id="Hfs7CX7qH8xI" outputId="1020564c-f87b-4749-e350-bc98914978a2"
new_dataset = None
async for sds in workflow.datasets():
    new_dataset = await sds.to_local(conn)
new_dataset.to_pandas()

# %% [markdown] id="ZgZ2UbdLR-sY"
# ### 2. fill missing values by its previous value in that column
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 175} id="RzIMV_D1SBC5" outputId="2961dc1f-9e2a-437e-9566-ae1e34a72a11"
dataset.to_pandas()

# %% id="eEzT3_xqSA4X"
fill_col = "a"
fill_null = ra.Transform({"function": {"fill_null_forward": [fill_col]}})

# %% colab={"base_uri": "https://localhost:8080/"} id="4WP8rO1bSAgo" outputId="d4a478de-1b75-4578-86fc-86052ee83002"
save = rf.actions.DatasetSave(name="fill_null_forward_dataset")
builder = rf.WorkflowBuilder()
builder.add_dataset(dataset)
builder.add_action(fill_null, parents=[dataset])
builder.add_action(save, parents=[fill_null])
workflow = await builder.start(conn)
print(f"Workflow: {workflow.id()}")

# %% colab={"base_uri": "https://localhost:8080/", "height": 175} id="Xwkf78WASIgn" outputId="7d1ccc03-41c9-4761-d225-3beb596c3bdd"
new_dataset = None
async for sds in workflow.datasets():
    new_dataset = await sds.to_local(conn)
new_dataset.to_pandas()

# %% [markdown] id="tx4lV8FZKdGJ"
# ### 3. fill missing values by its next value in that column
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 175} id="bOb83reMJ1E6" outputId="1dc5917f-b6e1-4769-fd25-fec6325803be"
dataset.to_pandas()

# %% id="FTFD7p3KJ1B5"
fill_col = "a"
fill_null = ra.Transform({"function": {"fill_null_backward": [fill_col]}})

# %% colab={"base_uri": "https://localhost:8080/"} id="PXFmpuy0J0_x" outputId="763d5126-59d1-4354-a6dd-4c284db69ac0"
save = rf.actions.DatasetSave(name="fill_null_backward_dataset")
builder = rf.WorkflowBuilder()
builder.add_dataset(dataset)
builder.add_action(fill_null, parents=[dataset])
builder.add_action(save, parents=[fill_null])
workflow = await builder.start(conn)
print(f"Workflow: {workflow.id()}")

# %% colab={"base_uri": "https://localhost:8080/", "height": 175} id="PEs4EkfFJ09J" outputId="dfcfbc71-8456-4cfb-fe33-1ee255fc06d6"
new_dataset = None
async for sds in workflow.datasets():
    new_dataset = await sds.to_local(conn)
new_dataset.to_pandas()

# %% [markdown] id="kWlVxdr-PxhF"
# ### 4. fill missing values by its mean value in that column
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 175} id="eST5gc8QJ06Z" outputId="158a88a1-c00e-4163-9d60-128da6e2abf4"
dataset.to_pandas()

# %% id="nDsd-vIxJ034"
fill_col = "a"
fill_method = "mean"
fill_null = ra.Transform(
    {"function": {"fill_null_aggregation": [fill_col, fill_method]}}
)

# %% colab={"base_uri": "https://localhost:8080/"} id="gfrNyuYqJ01S" outputId="d5fb3f5c-6d09-46d0-e44c-4c0e95f6e60c"
save = rf.actions.DatasetSave(name="fill_mean_dataset")
builder = rf.WorkflowBuilder()
builder.add_dataset(dataset)
builder.add_action(fill_null, parents=[dataset])
builder.add_action(save, parents=[fill_null])
workflow = await builder.start(conn)
print(f"Workflow: {workflow.id()}")

# %% colab={"base_uri": "https://localhost:8080/", "height": 175} id="DUYfDtwjJ0yt" outputId="0ba38739-1848-4bd3-d745-78f46d58c70d"
new_dataset = None
async for sds in workflow.datasets():
    new_dataset = await sds.to_local(conn)
new_dataset.to_pandas()

# %% [markdown] id="AslotMEGQT-R"
# ### 5. fill missing values by its median value in that column
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 175} id="V87_OqHTQJsH" outputId="02bb70ce-eafc-4629-d7db-30c25c0d1374"
dataset.to_pandas()

# %% id="L-IRNZrnRF5p"
fill_col = "a"
fill_method = "median"
fill_null = ra.Transform(
    {"function": {"fill_null_aggregation": [fill_col, fill_method]}}
)

# %% colab={"base_uri": "https://localhost:8080/"} id="IoDLCrFzRF2_" outputId="07c5cc9e-74ed-4429-d43c-6550713512bf"
save = rf.actions.DatasetSave(name="fill_median_dataset")
builder = rf.WorkflowBuilder()
builder.add_dataset(dataset)
builder.add_action(fill_null, parents=[dataset])
builder.add_action(save, parents=[fill_null])
workflow = await builder.start(conn)
print(f"Workflow: {workflow.id()}")

# %% colab={"base_uri": "https://localhost:8080/", "height": 175} id="A0iNxT0QRFuu" outputId="e959ee41-eb15-4a9b-9164-85e743f4f87c"
new_dataset = None
async for sds in workflow.datasets():
    new_dataset = await sds.to_local(conn)
new_dataset.to_pandas()

# %% [markdown] id="Ax0iShxLITgj"
# ## Append new column for the transformed field
#
# Add new column for the result after filling missing with indicated values and the original column with missing values keeps the same
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 175} id="jdkMPpHFTDwF" outputId="d3c046b8-563f-4933-fbab-a81de6540bb5"
dataset.to_pandas()

# %% id="Xj2TL77S-9eW"
fill_value = 42
fill_col = "a"
new_col_name = "new_a"
fill_null = ra.Apply(
    {
        "function": {"fill_null": [fill_col, fill_value]},
        "append_field": new_col_name,
    }
)

# %% colab={"base_uri": "https://localhost:8080/"} id="2UyI3YYLEH9W" outputId="d64835e1-7308-40b6-c438-51c05bbaa3c3"
save = ra.DatasetSave({"name": "new_column_filled_dataset"})
builder = rf.WorkflowBuilder()
builder.add_dataset(dataset)
builder.add_action(fill_null, parents=[dataset])
builder.add_action(save, parents=[fill_null])
workflow = await builder.start(conn)

print(f"Workflow: {workflow.id()}")

# %% colab={"base_uri": "https://localhost:8080/", "height": 175} id="TUhGuPL-ELxX" outputId="2c53c94d-b614-4c2e-fa05-21be70624ba1"
new_dataset = None
async for sds in workflow.datasets():
    new_dataset = await sds.to_local(conn)
new_dataset.to_pandas()
