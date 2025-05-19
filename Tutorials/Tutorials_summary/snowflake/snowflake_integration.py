# Rockfish Environment Key & API URL
import rockfish as rf
import rockfish.actions as ra
import rockfish.labs as rl
import asyncio
import os

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv
import snowflake.connector
from cryptography.hazmat.primitives import serialization
with open("./rsa_key_unencrypted.p8", "rb") as key_file:
    private_key = serialization.load_pem_private_key(
        key_file.read(),
        password=None
    )

conn = snowflake.connector.connect(
     user="<MYUSER>",
     private_key=private_key,
     account="<ACCOUNT>",
     warehouse="COMPUTE_WH",
     database="<DB>",
     schema="<SCHEMA>",
     application="RockfishData_RockfishSyntheticDataPlatform"
 )

cursor = conn.cursor()

cursor.execute("""Select * from <DB>.<SCHEMA>.<SOURCE_TABLE>""")
df = cursor.fetch_pandas_all()
print(df.head())


runner = asyncio

api_key = "<API_KEY>"
api_url = "<API_URL>"

os.environ["ROCKFISH_API_KEY"] = "api_key"


async def start_rockfish():
    conn = rf.Connection.remote(api_url, api_key)

    #Onboard
    # Perform any necessary feature engineering or preprocessing
    dataset = rf.Dataset.from_pandas("<MYDATASETNAME>",df)
    
    categorical_fields = (
        df.select_dtypes(include=["object"]).columns
    )
    print(categorical_fields)
    
    config = {
        "encoder": {
            "metadata": [
                {"field": field, "type": "categorical"}
                for field in categorical_fields
            ]
            + [
                {"field": field, "type": "continuous"}
                for field in dataset.table.column_names
                if field not in categorical_fields
            ],
        },
        "tabular-gan": {
            "epochs": 10,
            "records": 10000,
        }
    }
    print(dataset.table.column_names)

    #Train
    train = ra.TrainTabGAN(config)
    
    builder = rf.WorkflowBuilder()
    builder.add_dataset(dataset)
    builder.add_action(train, parents=[dataset])
    workflow = await builder.start(conn)
    print(f"Training - Workflow: {workflow.id()}")
    
    async for log in workflow.logs():
        print(log) 
    
    model = await workflow.models().nth(0)
    await model.add_labels(conn)
    #model

    #Generate
    generate = ra.GenerateTabGAN(config)
    save = ra.DatasetSave({"name": "synthetic"})
    builder = rf.WorkflowBuilder()
    builder.add_model(model)
    builder.add_action(generate, parents=[model])
    builder.add_action(save, parents=[generate])
    workflow = await builder.start(conn)
    print(f"Generate - Workflow: {workflow.id()}")
    
    syn = None
    async for sds in workflow.datasets():
        syn = await sds.to_local(conn)
    await conn.close()
    return syn
syn_data = runner.run(start_rockfish())

import csv

syn_data_pandas = syn_data.to_pandas()

syn_data_pandas.to_csv(f"synthetic.csv", index=False, quoting=csv.QUOTE_ALL)
print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVV")
print("VVV Sample Synthetic Data VVV")
print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVV")
print(syn_data_pandas.head())

cursor.execute("""CREATE OR REPLACE STAGE <DB>.<SCHEMA>.ROCKFISH_STAGE COMMENT = 'Stage for Rockfish demo data';""")

# --- Configuration ---
notebook_file_path_str = './synthetic.csv' # Verify this path within the notebook!
target_stage = '@<DB>.<SCHEMA>.ROCKFISH_STAGE' # Use the fully qualified stage name
# --- End Configuration ---



print(f"File found at: {notebook_file_path_str}")
# --- Prepare and Execute PUT command ---
file_uri = f"file://{str(notebook_file_path_str)}"
put_sql = f"PUT '{file_uri}' '{target_stage}' AUTO_COMPRESS=TRUE OVERWRITE=TRUE"
print(f"Executing via cursor session: {put_sql}")

# Execute and fetch results correctly using Snowpark/connector methods
put_results = cursor.execute(put_sql)

print("PUT command executed successfully.")
print("Results:")
for row in put_results:
    print(row) # Connector handles getting status correctly

cursor.execute(""" TRUNCATE TABLE <DB>.<SCHEMA>.<SYNTHETIC_TABLENAME>; """)

results = cursor.execute(""" SELECT COUNT(*) FROM <DB>.<SCHEMA>.<SYNTHETIC_TABLENAME>; """)
print("Results:")
for row in results:
    print(row) # Connector handles getting status correctly

results = cursor.execute(""" COPY INTO <DB>.<SCHEMA>.<SYNTHETIC_TABLENAME> FROM '@"<DB>"."<SCHEMA>"."ROCKFISH_STAGE"/synthetic.csv.gz' FILE_FORMAT = ( TYPE = CSV FIELD_DELIMITER = ',' SKIP_HEADER = 1 EMPTY_FIELD_AS_NULL = TRUE FIELD_OPTIONALLY_ENCLOSED_BY = '\"') ON_ERROR = 'CONTINUE'; """)
print("Results:")
for row in results:
    print(row) # Connector handles getting status correctly

results = cursor.execute(""" SELECT COUNT(*) FROM <DB>.<SCHEMA>.<SYNTHETIC_TABLENAME>; """)
print("Results:")
for row in results:
    print(row) # Connector handles getting status correctly

results = cursor.execute(""" SELECT * FROM <DB>.<SCHEMA>.<SYNTHETIC_TABLENAME> LIMIT 4; """)
print("Results:")
for row in results:
    print(row) # Connector handles getting status correctly

cursor.close()
conn.close()
