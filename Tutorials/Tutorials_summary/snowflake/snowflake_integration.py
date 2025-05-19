import os
import asyncio
import pandas as pd
import snowflake.connector
from cryptography.hazmat.primitives import serialization

import rockfish as rf
import rockfish.actions as ra

# --- Configuration (fill in or use env vars) ---
DB = os.getenv("SNOWFLAKE_DB", "<DB>")
SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", "<SCHEMA>")
USER = os.getenv("SNOWFLAKE_USER", "<MYUSER>")
ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT", "<ACCOUNT>")
WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
API_KEY = os.getenv("ROCKFISH_API_KEY", "<API_KEY>")
API_URL = os.getenv("ROCKFISH_API_URL", "<API_URL>")
PRIVATE_KEY_PATH = os.getenv("PRIVATE_KEY_PATH", "./rsa_key_unencrypted.p8")
SOURCE_TABLE = os.getenv("SOURCE_TABLE", "<SOURCE_TABLE>")
SYNTHETIC_TABLENAME = os.getenv("SYNTHETIC_TABLENAME", "<SYNTHETIC_TABLENAME>")
DATASET_NAME = os.getenv("ROCKFISH_DATASET_NAME", "<MYDATASETNAME>")
STAGE_NAME = f"{DB}.{SCHEMA}.ROCKFISH_STAGE"
CSV_PATH = "./synthetic.csv"

# --- Helper Functions ---

def load_private_key(key_path):
    with open(key_path, "rb") as key_file:
        return serialization.load_pem_private_key(key_file.read(), password=None)

def get_snowflake_connection():
    private_key = load_private_key(PRIVATE_KEY_PATH)
    return snowflake.connector.connect(
        user=USER,
        private_key=private_key,
        account=ACCOUNT,
        warehouse=WAREHOUSE,
        database=DB,
        schema=SCHEMA,
        application="RockfishData_RockfishSyntheticDataPlatform"
    )

def fetch_source_data(cursor):
    cursor.execute(f"SELECT * FROM {DB}.{SCHEMA}.{SOURCE_TABLE}")
    return cursor.fetch_pandas_all()

async def run_rockfish(df):
    conn = rf.Connection.remote(API_URL, API_KEY)
    dataset = rf.Dataset.from_pandas(DATASET_NAME, df)
    categorical_fields = df.select_dtypes(include=["object"]).columns.tolist()
    config = {
        "encoder": {
            "metadata": [
                {"field": field, "type": "categorical"} for field in categorical_fields
            ] + [
                {"field": field, "type": "continuous"}
                for field in dataset.table.column_names if field not in categorical_fields
            ]
        },
        "tabular-gan": {"epochs": 10, "records": 10000}
    }

    # Train
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

    # Generate
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

def upload_to_snowflake_stage(cursor, file_path, stage):
    cursor.execute(f"CREATE OR REPLACE STAGE {stage} COMMENT = 'Stage for Rockfish demo data';")
    file_uri = f"file://{file_path}"
    put_sql = f"PUT '{file_uri}' '@{stage}' AUTO_COMPRESS=TRUE OVERWRITE=TRUE"
    print(f"Executing: {put_sql}")
    put_results = cursor.execute(put_sql)
    for row in put_results:
        print(row)

def load_synthetic_to_table(cursor, stage, table):
    cursor.execute(f"TRUNCATE TABLE {DB}.{SCHEMA}.{table};")
    cursor.execute(f"""
        COPY INTO {DB}.{SCHEMA}.{table}
        FROM '@{stage}/synthetic.csv.gz'
        FILE_FORMAT = (TYPE = CSV FIELD_DELIMITER = ',' SKIP_HEADER = 1 EMPTY_FIELD_AS_NULL = TRUE FIELD_OPTIONALLY_ENCLOSED_BY = '\"')
        ON_ERROR = 'CONTINUE';
    """)
    cursor.execute(f"SELECT COUNT(*) FROM {DB}.{SCHEMA}.{table};")
    for row in cursor:
        print(row)
    cursor.execute(f"SELECT * FROM {DB}.{SCHEMA}.{table} LIMIT 4;")
    for row in cursor:
        print(row)

# --- Main Workflow ---

def main():
    os.environ["ROCKFISH_API_KEY"] = API_KEY  # For rockfish client

    with get_snowflake_connection() as conn:
        with conn.cursor() as cursor:
            print("Fetching source data from Snowflake...")
            df = fetch_source_data(cursor)
            print(df.head())

            print("Running Rockfish synthetic data pipeline...")
            syn_data = asyncio.run(run_rockfish(df))
            syn_df = syn_data.to_pandas()
            syn_df.to_csv(CSV_PATH, index=False, quoting=1)  # quoting=csv.QUOTE_ALL = 1

            print("Sample Synthetic Data:")
            print(syn_df.head())

            print("Uploading synthetic data to Snowflake stage...")
            upload_to_snowflake_stage(cursor, CSV_PATH, STAGE_NAME)

            print("Loading synthetic data into target table...")
            load_synthetic_to_table(cursor, STAGE_NAME, SYNTHETIC_TABLENAME)

    print("All done!")

if __name__ == "__main__":
    main()

