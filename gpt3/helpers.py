import pandas as pd

def h5_to_df(path):
    df = pd.read_hdf(path, key="data")
    return df.loc[~df.solution.isna()]

def training_levels(path, is_h5=True):
    """
    Function to get training levels
    """
    if is_h5:
        df = h5_to_df(path)
    else:
        df = pd.read_csv(path)
    return list(df.level)