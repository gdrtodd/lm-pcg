import pandas as pd

import json
from helpers import h5_to_df


def extract_data(series, filename):

    data = []
    completions = list(series)
    #print(completions)
    for i in series:
        prompt = {"prompt": f"Map. ->", "completion": f" {i}. END"}
        data.append(prompt)
    with open(filename, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    


def main():
    
    file_path = "microban_data.h5"
    df = h5_to_df(file_path)
    extract_data(df["level"], "microban.jsonl")

main()