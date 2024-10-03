import pandas as pd

loc3_filepath = "location3.csv"
loc3_data = pd.read_csv(loc3_filepath)
loc3_data["timestamp"] = pd.to_datetime(loc3_data["timestamp"])
loc3_data = loc3_data.drop(columns=["sessionID"])

FIRST_TS = pd.Timestamp("2023-08-05 00:00:00")
LAST_TS = loc3_data["timestamp"].iloc[-1]
timedelta = pd.Timedelta(hours=1)

start_ts = FIRST_TS

def get_readable_ts(ts):
    return ts.strftime('%Y-%m-%d_hour%H')

while start_ts <= LAST_TS:
    end_ts = start_ts + timedelta
    split = loc3_data[(loc3_data["timestamp"] >= start_ts) & (loc3_data["timestamp"] < end_ts)]
    split["timestamp"].to_csv(f"location3_hours/location3_{get_readable_ts(start_ts)}_timestamp.csv", index=False)
    split = split.drop(columns=["timestamp"])
    split.to_csv(f"location3_hours/location3_{get_readable_ts(start_ts)}.csv", index=False)
    start_ts = end_ts