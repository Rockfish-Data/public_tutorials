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

# %% id="hgVdjEBIGUub"
# %%capture
# %pip install -U 'rockfish[labs]' -f 'https://packages.rockfish.ai'

# %% id="URcxxdQA5X5y"
import io
import rockfish as rf
import rockfish.actions as ra
import rockfish.labs

# %% [markdown] id="oGt18-G6EHtM"
# ### Amplify on selected field from tabular data
#

# %% id="l9UNn2CBEZvB"
tabular_data = b"""\
Age range of patient,Sex,BBS Score,Heart Rate
60<70,M,41,80
30<40,F,41,78
60<70,M,43,81
80<90,M,40,82
60<70,M,40,90
60<70,M,38,90
20<30,M,38,70
1<13,F,38,75
70<80,M,38,70
40<50,F,38,80
80<90,M,42,88
40<50,F,42,89
60<70,M,42,90
80<90,M,42,101
70<80,M,42,100
80<90,F,42,101
60<70,M,39,99
1<13,F,39,98
80<90,M,39,75
70<80,M,39,74
"""

# %% id="QJtis_SKgQZm"
# connect locally
conn = rf.Connection.local()

# %% colab={"base_uri": "https://localhost:8080/", "height": 677} id="De95IZHmEZz8" outputId="7c3dd34d-96cb-43a5-d3b5-fc96f1a3eb77"
tab_dataset = rf.Dataset.from_csv(
    "Before_amplified", io.BytesIO(tabular_data)
)
tab_dataset.to_pandas()

# %% id="L75GVN_TEZxc"
# dropping records in column "Sex" other than the value of "M" by given percentage
post_amplify = ra.PostAmplify(
    {
        "query_ast": {
            "eq": ["Sex", "M"],
        },
        "drop_match_percentage": 0.0,
        "drop_other_percentage": 0.7,
    }
)

save_amplified = ra.DatasetSave({"name": "After_amplified"})
builder = rf.WorkflowBuilder()
builder.add_dataset(tab_dataset)
builder.add_action(post_amplify, parents=[tab_dataset])
builder.add_action(save_amplified, parents=[post_amplify])
workflow = await builder.start(conn)

# %% colab={"base_uri": "https://localhost:8080/"} id="roIN6wZBhMdM" outputId="e77dd252-c600-4017-87c2-1cf83c1769cb"
async for progress in workflow.progress("post-amplify"):
    print(progress)

# %% colab={"base_uri": "https://localhost:8080/", "height": 551} id="AChCXk5vhMUb" outputId="b79075dc-058e-4c21-f380-e32529f9cd1c"
amplified_tab_dataset = None
async for sds in workflow.datasets():
    amplified_tab_dataset = await sds.to_local(conn)
amplified_tab_dataset.to_pandas()

# %% colab={"base_uri": "https://localhost:8080/", "height": 505} id="kuybVyV3hX4y" outputId="c048101c-da28-4282-c910-532da6a910fd"
# before vs after amplifying the condition: Sex == "M"
col = "Sex"
before_tab_agg = rf.metrics.count_all(tab_dataset, col)
after_tab_agg = rf.metrics.count_all(amplified_tab_dataset, col)
rf.labs.vis.plot_bar([before_tab_agg, after_tab_agg], col, f"{col}_count")
