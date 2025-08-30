import pandas as pd

def load_data():
    # Simple file loading - make sure tiktok_dataset.csv is in the same folder
    df = pd.read_csv("tiktok_dataset.csv")
    return df