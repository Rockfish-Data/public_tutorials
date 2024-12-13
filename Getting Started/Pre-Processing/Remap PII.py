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

# %% [markdown] id="MRNVCRrvcixF"
# # Remap PII Tutorial
#

# %% [markdown] id="wZFzCJSDHGEO"
# This tutorial demonstrates how sensitive data can be anonymized in Rockfish. We show two examples here for anonymizing datasets with multiple kinds of Personally Identifiable Information (PII).
#

# %% id="ivaAZJJyrfAW"
# %%capture
# %pip install -U 'rockfish[labs]' -f 'https://docs.rockfish.ai/packages/index.html'

# %% id="uNfoyOhlrs-T"
import rockfish as rf
import rockfish.actions as ra

# %% id="Qw0y32WWXtCY"
# connect locally
conn = rf.Connection.local()

# %% [markdown] id="_hdWeKdWEzO2"
# # Example 1
#

# %% [markdown] id="ndrqE0pjcuEk"
# Download sample dataset with PII:
#

# %% [markdown] id="FYd_xlTpcyuG"
# Convert into a Rockfish dataset:
#

# %% colab={"base_uri": "https://localhost:8080/"} id="xUWR9j8cr5cT" outputId="8cab0ca3-ecdf-4f29-9add-c9dc442e4457"
# !wget --no-clobber https://raw.githubusercontent.com/tokern/piicatcher/master/tests/samples/sample-data.csv

# %% id="SGVVbWe5r-FU"
dataset = rf.Dataset.from_csv("sample-data", "sample-data.csv")

# %% [markdown] id="LPZF58e_eCcM"
# We can see that this dataset has PII: SSNs, birthdates, email addresses, etc.
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 990} id="rtL3KX7ud9sj" outputId="af053345-bd39-440c-d79d-ed9edf928080"
dataset.to_pandas()

# %% [markdown] id="CxRhaXQ0dXC4"
# ## Remap = Action
#
# Users can add remap actions to a Rockfish Workflow.
#
# To use a `remap` function, the following things need to be specified:
#
# 1. `remap_type`: The type of remap function you want to use. Supported remap types are described below.
# 2. `select_col`: The name of the field in your dataset that you want to remap.
# 3. `new_remapped_col`: The name of the new remapped field that should be added to the dataset, in case you don't want to
#    overwrite the original field (e.g., for testing purposes).
# 4. `options`: An optional dictionary of arguments to customize your remap function.
#
# We will look at a few example remap actions below.
#

# %% [markdown] id="zJ2qacFmR2He"
# ## Remapping SSNs
#
# Mask the last 8 characters using "X".
#
# - `remap_type`: "ssn"
# - `options` to customize the default function if needed:
#   - `mask_char`: Any string that will be used as the masking character.
#   - `mask_length`: Number of characters to mask.
#   - `from_end`: A boolean to mask from the beginning of the field (False) or from the end (True).
#

# %% id="8A8gOOzvsC0I"
remap_type = "ssn"
select_col = "id"
options = None
remap_ssn = ra.Transform(
    {"function": {"remap": [remap_type, select_col, options]}}
)

# %% [markdown] id="excHeXOYR6rZ"
# ## Remapping Dates
#
# Replace timestamps that contain both time and date with day, month, and year. To remap field that only have dates, you
# can specify a more general `format_str` option (e.g., to keep month and year use "%b %Y").
#
# - `remap_type`: "date"
# - `options` to customize the default function if needed: - `format_str`: A valid datetime format string (see
#   [datetime documentation](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior) for possible formats).
#

# %% id="fMOyoLoDQBoW"
remap_type = "date"
select_col = "birthdate"
options = {"format_str": "%b %Y"}
new_remapped_col = "remapped_birthdate"
remap_date = ra.Apply(
    {
        "function": {"remap": [remap_type, select_col, options]},
        "append_field": new_remapped_col,
    }
)

# %% [markdown] id="F2LnR2C9xUon"
# ## Remapping Email Addresses
#
# Replace emails with randomly generated fake email addresses.
#
# - `remap_type`: "email"
# - `options` to customize the default function if needed:
#   - `gender`: "M" to generate email addresses with male first names (default = female first names).
#   - `locale`: See [locale documentation](https://faker.readthedocs.io/en/master/locales.html)
#     for supported locale types (default locale = "en_US").
#   - `seed`: Seed for the random generator (default = 0).
#

# %% id="Fm8zxJAQxc3D"
remap_type = "email"
select_col = "email"
options = None
new_remapped_col = "remapped_email"
remap_email = ra.Apply(
    {
        "function": {"remap": [remap_type, select_col, options]},
        "append_field": new_remapped_col,
    }
)

# %% [markdown] id="C4QrxCB3SS5E"
# ## Remapping Phone Numbers
#
# Replace with randomly generated fake phone numbers.
#
# - `remap_type`: "phone_number"
# - `options` to customize the default function if needed:
#   - `locale`: See [locale documentation](https://faker.readthedocs.io/en/master/locales.html)
#     for supported locale types (default locale = "en_US").
#   - `seed`: Seed for the random generator (default = `None`).
#

# %% id="CcEHJOmdSJpi"
remap_type = "phone_number"
select_col = "phone"
options = None
new_remapped_col = "remapped_phone"
remap_phone = ra.Apply(
    {
        "function": {"remap": [remap_type, select_col, options]},
        "append_field": new_remapped_col,
    }
)

# %% [markdown] id="hDeWUv9nDvN9"
# ## Remapping CVCs Using Custom Bins
#
# Replace CVC values with the bucket they fall into.
#
# - `remap_type`: "custom_bins"
# - `options`:
#   - `bins`: A number `n` to split the range of values into `n` buckets, or a list that contains specific intervals.
#     For example, to use intervals "[0, 10)" and "[10, 20)", specify `bins = [0, 10, 20]`.
#   - `right`: A boolean to make the right side of the interval inclusive (True) or not (False).
#   - `labels`: Labels for intervals, if needed (default = None).
#
# See [documentation for pandas.cut()](https://pandas.pydata.org/docs/dev/reference/api/pandas.cut.html) for a more detailed explanation of the arguments.
#

# %% id="tVX_shg4FSZj"
remap_type = "custom_bins"
select_col = "cc_cvc"
options = {
    "bins": 10,
    "right": False,
    "labels": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
}
new_remapped_col = "remapped_cc_cvc"
remap_cvc = ra.Apply(
    {
        "function": {"remap": [remap_type, select_col, options]},
        "append_field": new_remapped_col,
    }
)

# %% [markdown] id="lT3ovto1VhNg"
# ## Remapping Gender Using Custom Dict
#
# Replace values according to a dictionary containing mappings from original values to new values.
#
# - `remap_type`: "custom_dict"
# - `options`:
#   - `new_values_dict`: Dictionary with mappings from original values to new values. Not all original values in a field
#     need to have a mapping. Values without a mapping will not be replaced.
#

# %% id="XK4sp-7cVr2Y"
remap_type = "custom_dict"
select_col = "gender"
options = {
    "new_values_dict": {
        "m": "Male",
        "M": "Male",
        "f": "Female",
    }
}
new_remapped_col = "remapped_gender"
remap_gender = ra.Apply(
    {
        "function": {"remap": [remap_type, select_col, options]},
        "append_field": new_remapped_col,
    }
)

# %% [markdown] id="k-ZaMbUiZXPM"
# ## Save Remapped Dataset
#

# %% id="v4Da-D5mP_Nr"
save_remapped = rf.actions.DatasetSave({"name": "remapped_dataset"})

# %% [markdown] id="y6Kmr3auSxPm"
# ## Build And Run Workflow
#

# %% id="2ISr54mbsHDU"
preprocess_builder = rf.WorkflowBuilder()
preprocess_builder.add_dataset(dataset)
preprocess_builder.add_action(remap_ssn, alias="remap_ssn", parents=[dataset])
preprocess_builder.add_action(
    remap_date, alias="remap_date", parents=[remap_ssn]
)
preprocess_builder.add_action(
    remap_email, alias="remap_email", parents=[remap_date]
)
preprocess_builder.add_action(
    remap_phone, alias="remap_phone", parents=[remap_email]
)
preprocess_builder.add_action(
    remap_cvc, alias="remap_cvc", parents=[remap_phone]
)
preprocess_builder.add_action(
    remap_gender, alias="remap_gender", parents=[remap_cvc]
)
preprocess_builder.add_action(save_remapped, parents=[remap_gender])

# %% colab={"base_uri": "https://localhost:8080/"} id="tgv-ToXewHBa" outputId="e8732452-3718-47e9-dbbc-f4eaf91482f2"
for action in preprocess_builder.actions:
    print(action)

# %% colab={"base_uri": "https://localhost:8080/"} id="WpYzhCHBsNgk" outputId="02f34fb9-6253-470d-eabe-5097480480c7"
preprocess_workflow = await preprocess_builder.start(conn)
remapped_dataset = None

print(f"Workflow: {preprocess_workflow.id()}")

# %% colab={"base_uri": "https://localhost:8080/"} id="QUHG-PA5snK0" outputId="82dcee38-7f68-4d92-8b68-b424cb2fe21e"
async for log in preprocess_workflow.logs():
    print(log)

# %% id="9r-CTQiisZeD"
async for sds in preprocess_workflow.datasets():
    remapped_dataset = await sds.to_local(conn)

# %% [markdown] id="3lqmifQzS3NF"
# ## Outputs
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 363} id="-Um2Hwo8scT0" outputId="bfe1c9a4-a543-49be-e4e2-9559f6c08ee3"
remapped_dataset.to_pandas()[["id"]][:10]

# %% colab={"base_uri": "https://localhost:8080/", "height": 363} id="fviIc4wtQ6lA" outputId="0fbc2d92-1e1d-484f-c4d6-ab3949ba3bb3"
remapped_dataset.to_pandas()[["birthdate", "remapped_birthdate"]][:10]

# %% colab={"base_uri": "https://localhost:8080/", "height": 363} id="y8yQsWodx_rx" outputId="ad40449c-32fe-41cf-e648-2c5229fefd54"
remapped_dataset.to_pandas()[["email", "remapped_email"]][:10]

# %% colab={"base_uri": "https://localhost:8080/", "height": 363} id="hBSFSRHZS9qg" outputId="07b28a7e-cb9e-4744-d87a-9c4c17d8ef94"
remapped_dataset.to_pandas()[["phone", "remapped_phone"]][:10]

# %% colab={"base_uri": "https://localhost:8080/", "height": 363} id="YxJJ6Ou0FmVv" outputId="f52d344a-c59c-46bc-a1e5-9028bc3deb60"
remapped_dataset.to_pandas()[["cc_cvc", "remapped_cc_cvc"]][:10]

# %% colab={"base_uri": "https://localhost:8080/", "height": 363} id="Mp8R7yUwXwEy" outputId="bc5a9f62-f859-43f7-e92c-9e2e6dbc0855"
remapped_dataset.to_pandas()[["gender", "remapped_gender"]][:10]

# %% [markdown] id="nAd0BRh0E56h"
# # Example 2
#

# %% [markdown] id="iETSwoO_E_AS"
# Download sample dataset with IP addresses:
#

# %% colab={"base_uri": "https://localhost:8080/"} id="Fk1A0ZFUE8a9" outputId="11914586-cda7-4de0-f9e9-0c639c3507aa"
# !wget --no-clobber https://docs142.rockfish.ai/tutorials/pcap.csv

# %% [markdown] id="YjN_rYyIGtgn"
# Convert into Rockfish dataset:
#

# %% id="UWMRz-mBGwSO"
dataset = rf.Dataset.from_csv("pcap", "pcap.csv")

# %% [markdown] id="3Zlt0ueWG1v6"
# We can see that this dataset has PII: IP Addresses.
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="FkIjrINcG75K" outputId="50b49a16-e536-4f15-d7dd-484009a39d0c"
dataset.to_pandas()

# %% [markdown] id="7FseSh2CHlcv"
# ## Remapping IP Addresses
#
# Replace IP addresses with randomly generated fake IP addresses.
#
# - `remap_type`: "ip"
# - `options` to customize the default function if needed:
#   - `cidr`: A netmask value in ["/0", "/8", "/16", "/24"] (default = "/24").
#   - `seed`: Seed for the random generator (default = `None`).
#

# %% id="jWBvjvrSHpRo"
remap_type = "ip"
select_col = "srcip"
options = None
new_remapped_col = "remapped_srcip"
remap_ip = ra.Apply(
    {
        "function": {"remap": [remap_type, select_col, options]},
        "append_field": new_remapped_col,
    }
)

# %% id="KspWsqZpINJn"
save_remapped = rf.actions.DatasetSave({"name": "remapped_dataset"})

# %% [markdown] id="-_JX33UyIIH_"
# ## Build And Run Workflow
#

# %% id="HQja9RWXIBTq"
preprocess_builder = rf.WorkflowBuilder()
preprocess_builder.add_dataset(dataset)
preprocess_builder.add_action(remap_ip, alias="remap_ip", parents=[dataset])
preprocess_builder.add_action(save_remapped, parents=[remap_ip])

# %% colab={"base_uri": "https://localhost:8080/"} id="bANy4qSdIaQb" outputId="7572c60f-cde8-4515-df43-bc008a3e8cbd"
for action in preprocess_builder.actions:
    print(action)

# %% colab={"base_uri": "https://localhost:8080/"} id="OqNoTGigIbc6" outputId="17bcd19a-b3f1-4e5c-f5f1-ee3c12799aa4"
preprocess_workflow = await preprocess_builder.start(conn)
remapped_dataset = None

print(f"Workflow: {preprocess_workflow.id()}")

# %% colab={"base_uri": "https://localhost:8080/"} id="e5YoqmPbIfgn" outputId="8523be3b-342b-4ebc-b45d-d446a06a9462"
async for log in preprocess_workflow.logs():
    print(log)

# %% id="VT2RKWXxJApY"
async for sds in preprocess_workflow.datasets():
    remapped_dataset = await sds.to_local(conn)

# %% [markdown] id="tKScEqLTIjhy"
# ## Outputs
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 363} id="Dpck3WPvIlF0" outputId="6ddb82ed-12eb-477a-dd7b-2df225512c8b"
remapped_dataset.to_pandas()[["srcip", "remapped_srcip"]][:10]
