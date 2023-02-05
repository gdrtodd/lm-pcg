import pandas as pd

def h5_to_df(path):
    df = pd.read_hdf(path, key="data")
    return df.loc[~df.solution.isna()]