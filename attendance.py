import pandas as pd
from datetime import datetime
import os

FILE = "attendance.csv"

def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    if os.path.exists(FILE):
        df = pd.read_csv(FILE)
        if ((df["Name"] == name) & (df["Date"] == date)).any():
            return False
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    df.loc[len(df)] = [name, date, time]
    df.to_csv(FILE, index=False)
    return True
