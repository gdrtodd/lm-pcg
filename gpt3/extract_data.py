import pandas as pd

import json
from helpers import h5_to_df


def extract_data(df, filename, experiment):

    data = []
    completions = list(df["level"])
    sol_len = list(df["solution_len"])
    for i in range(0,df.shape[0]):
        if experiment == "sample":
            prompt = {"prompt": f"Map: ->", "completion": f" {completions[i]}. END"}
        elif experiment == "control":
            prompt = {"prompt": f"solution length = {sol_len[i]}: ->", "completion": f" {completions[i]}. END"}
        data.append(prompt)
    with open(filename, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    


def main():
    
    file_path = "cache/microban_data.h5"
    df = h5_to_df(file_path)
    extract_data(df, "microban_control.jsonl", "control")

main()