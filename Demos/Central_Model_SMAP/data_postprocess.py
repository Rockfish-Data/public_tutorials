import pandas as pd
from pathlib import Path

dirpath = Path("syn_location3_hours")
filenames = sorted([
    file.name for file in dirpath.glob('location3_*.csv')
])

syn_datasets = []
for filename in filenames:
    timestamps = pd.read_csv(f"location3_hours/{filename[:-4]}_timestamp.csv")["timestamp"].to_list()

    syn_dataset = pd.read_csv(f"syn_location3_hours/{filename}")
    syn_dataset = syn_dataset.iloc[:len(timestamps)] # keep len the same as real timestamp len
    syn_dataset["timestamp"] = timestamps

    syn_datasets.append(syn_dataset)

final_syn = pd.concat(syn_datasets)
final_syn.to_csv("loc3_syn_tabgan.csv", index=False)