import pandas as pd
import os

CSV_FILE = "data/data_log.csv"

def log_data(data):
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame(columns=["timestamp","distance","temperature","brightness"])
        df.to_csv(CSV_FILE, index=False)

    df = pd.DataFrame([data])
    df.to_csv(CSV_FILE, mode='a', header=False, index=False)