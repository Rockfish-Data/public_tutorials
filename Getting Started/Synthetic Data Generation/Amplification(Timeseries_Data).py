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

# %% id="URcxxdQA5X5y"
import io

import rockfish as rf
import rockfish.actions as ra
import rockfish.labs as rl

# %% [markdown] id="9VelWpAOEPDS"
# ### Amplify on measurement field from timeseries data
#

# %% id="EgLxVUhV5XzY"
timeseries_data = b"""\
customer,age,gender,category,amount,timestamp,fraud
C249282739,4,Male,transportation,6.18,1970-02-11,0
C1847494890,3,Male,transportation,30.34,1970-02-17,0
C1415998962,2,Female,food,26.58,1970-05-29,0
C528985379,3,Female,transportation,47.86,1970-02-25,0
C911364935,3,Male,transportation,29.67,1970-02-12,0
C469059843,5,Female,transportation,6.59,1970-04-15,0
C1938976805,2,Male,transportation,38.26,1970-04-11,0
C1692355722,2,Male,transportation,33.2,1970-01-03,0
C851156881,1,Male,transportation,2.89,1970-02-26,0
C985732014,2,Female,transportation,33.25,1970-04-04,0
C1896828249,2,Male,transportation,42.42,1970-04-13,0
C1548142512,4,Female,transportation,31.13,1970-02-13,0
C1092512638,3,Female,transportation,47.28,1970-05-08,0
C226835436,4,Female,food,24.48,1970-04-11,0
C232369406,5,Female,transportation,17.81,1970-03-30,0
C1002759277,1,Female,transportation,38.87,1970-06-11,0
C877486986,0,Female,transportation,2.03,1970-04-26,0
C2064491438,5,Female,transportation,37.83,1970-04-10,0
C1127983201,6,Female,transportation,6.54,1970-06-21,0
C1806072501,5,Male,transportation,0.06,1970-03-18,0
C1070277785,4,Female,wellnessandbeauty,300.58,1970-05-01,1
C1560676680,2,Female,otherservices,66.6,1970-01-01,1
"""

# %% colab={"base_uri": "https://localhost:8080/", "height": 739} id="ewHOFHPMLAWk" outputId="817a41fe-4747-477e-a07f-a15c8658cff0"
ts_dataset = rf.Dataset.from_csv(
    "Before_amplified", io.BytesIO(timeseries_data)
)
schema_metadata = rf.arrow.SchemaMetadata(
    metadata=["customer", "age", "gender"]
)
ts_dataset.table = rf.arrow.replace_schema_metadata(
    ts_dataset.table, schema_metadata
)
ts_dataset.to_pandas()

# %% id="7wynERc2fTNH"
# connect locally
conn = rf.Connection.local()

# %% [markdown] id="_oPUJ8e7Ctrg"
# ### Example 1
#
# Amplify on one condition
#

# %% id="7WGaNX4J5XtB"
# amplify fraud transaction by dropping non-fraud
post_amplify = ra.PostAmplify(
    {
        "query_ast": {
            "eq": ["fraud", 1],
        },
        "drop_match_percentage": 0.0,
        "drop_other_percentage": 0.5,
    }
)

save_amplified = ra.DatasetSave({"name": "After_amplified_1"})
builder = rf.WorkflowBuilder()
builder.add_dataset(ts_dataset)
builder.add_action(post_amplify, parents=[ts_dataset])
builder.add_action(save_amplified, parents=[post_amplify])
workflow = await builder.start(conn)

# %% colab={"base_uri": "https://localhost:8080/"} id="8yQNK8D35QKO" outputId="15469eb7-5e9d-450d-bd44-9d1e46320044"
async for progress in workflow.progress("post-amplify"):
    print(progress)

# %% colab={"base_uri": "https://localhost:8080/", "height": 363} id="znWVejijEBfx" outputId="51edbdd4-5ecd-4be0-aa2d-58e527c9fe3e"
amplified_ts_dataset_1 = None
async for sds in workflow.datasets():
    amplified_ts_dataset_1 = await sds.to_local(conn)
amplified_ts_dataset_1.to_pandas()

# %% colab={"base_uri": "https://localhost:8080/", "height": 503} id="JAg0gxHzEBYR" outputId="bd633e74-7ca3-4dbb-a4b5-66f39e8622dc"
# before vs after amplifying fraud transactions
col = "fraud"
before_agg = rf.metrics.count_all(ts_dataset, col)
after_agg = rf.metrics.count_all(amplified_ts_dataset_1, col)
rl.vis.plot_bar([before_agg, after_agg], col, f"{col}_count")

# %% [markdown] id="Pye-C2I0CzRc"
# ### Example 2
#
# Amplify on multi conditions
#

# %% id="l9-M0IEv-cT4"
# amplify the conditions when their "gender" values are not "Female" and their "age" values are less than 4
post_amplify = ra.PostAmplify(
    {
        "query_ast": {
            "and": [
                {"lt": ["age", 4]},
                {"ne": ["gender", "Female"]},
            ]
        },
        "drop_match_percentage": 0.0,
        "drop_other_percentage": 0.5,
    }
)

save_amplified = ra.DatasetSave({"name": "After_amplified_2"})
builder = rf.WorkflowBuilder()
builder.add_dataset(ts_dataset)
builder.add_action(post_amplify, parents=[ts_dataset])
builder.add_action(save_amplified, parents=[post_amplify])
workflow = await builder.start(conn)

# %% colab={"base_uri": "https://localhost:8080/"} id="g7EWaaTZ--Mg" outputId="7685de28-82b6-4c21-db88-6f61f2fd12f7"
async for progress in workflow.progress("post-amplify"):
    print(progress)

# %% colab={"base_uri": "https://localhost:8080/", "height": 645} id="yh_i9JjH-_kF" outputId="59b4ee45-cef6-40bd-e35d-f940939a0097"
amplified_ts_dataset_2 = None
async for sds in workflow.datasets():
    amplified_ts_dataset_2 = await sds.to_local(conn)
amplified_ts_dataset_2.to_pandas()

# %% colab={"base_uri": "https://localhost:8080/", "height": 535} id="Cz1DvFsq_CEI" outputId="e3a20932-ba68-410e-fee8-49b26630be13"
# before vs after amplifying the gender which is not as Female --> amplify Male
col = "gender"
before_agg = rf.metrics.count_all(ts_dataset, col)
after_agg = rf.metrics.count_all(amplified_ts_dataset_2, col)
rl.vis.plot_bar([before_agg, after_agg], col, f"{col}_count")

# %% colab={"base_uri": "https://localhost:8080/", "height": 501} id="7Y9fyVq4KS6j" outputId="77eb748d-8852-4e2c-9cf6-e194f412a47d"
# before vs after amplifying the age less than 4
col = "age"
rl.vis.plot_kde([ts_dataset, amplified_ts_dataset_2], col)
