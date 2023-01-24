import hashlib
import os
import multiprocessing as mp
from multiprocessing import set_start_method, get_context
import numpy as np
import pandas as pd
from tqdm import tqdm

from sokoban_solvers import EnhancedAStarAgent, State
from utils import encode_boxoban_text

class BoxobanData():
    def __init__(self,
                 data_dir,
                 save_dir):  

        # The data will ultimately be stored in a dataframe that stores:
        #  - the level, represented as an ASCII grid
        #  - the level, represented as a linguistic description
        #  - the level's hash
        #  - the width and height of the level
        #  - the number of boxes and targets
        #  - the proportion of empty spaces to overall level size
        #  - the number of steps to solve the level
        #  - the level solution, das a list of actions
        #  - the split (train/val)
        data_keys = ["level", "level_text", "level_hash", "width", "height", "num_targets", "prop_empty", "solution_len", "solution", "split"]

        cache_file = os.path.join(save_dir, "boxoban_data.h5")
        if os.path.exists(cache_file):
            self.data = pd.read_hdf(cache_file, key="data")
        else:
            self.data = pd.DataFrame(columns=data_keys)

        train_levels_dir = os.path.join(data_dir, "boxoban-medium", "train")
        val_levels_dir = os.path.join(data_dir, "boxoban-medium", "val")

        train_level_files = [os.path.join(train_levels_dir, file) for file in os.listdir(train_levels_dir) if file.endswith(".txt")]
        val_level_files = [os.path.join(val_levels_dir, file) for file in os.listdir(val_levels_dir) if file.endswith(".txt")]

        # Process train levels
        for level_file in train_level_files:
            print(f"\nProcessing {level_file}...")

            with open(level_file, "r") as f:
                raw_levels = [level for level in f.read().split("; ") if level != ""]

            filtered_levels = [level for level in raw_levels if self._hash_level(level) not in self.data["level_hash"].values]
            print(f"Already have values for {len(raw_levels) - len(filtered_levels)} levels, processing {len(filtered_levels)} levels...")

            if len(filtered_levels) == 0:
                continue

            with get_context("spawn").Pool(8) as pool:
                level_infos = list(tqdm(pool.imap(self._process_level, filtered_levels), 
                                        total=len(filtered_levels), desc="Processing levels"))

                # Rearrange the level infos into a list of lists, where each list contains the specific value for every level
                vals = list(zip(*level_infos))
                
                info_df = pd.DataFrame.from_dict(dict(zip(data_keys, vals)))
                info_df["split"] = "train"

                print("Num no solution:", info_df.loc[info_df["solution_len"] == -1].shape[0])

                # Add to the main dataframe and save it progressively
                self.data = pd.concat([self.data, info_df], ignore_index=True)
                self.data.to_hdf(cache_file, key="data")
    

    def _process_level(self, level):
        '''
        Given a boxoban level, return a dictionary containing all of the relevant information, 
        (see comment in __init__) for details
        '''

        solver = EnhancedAStarAgent()

        level_hash = self._hash_level(level)

        if level_hash == 10:
            return [0] * 9

        level_text = encode_boxoban_text(level)
        level_state = State().stringInitialize(level.split("\n"))

        # Remove the first line of each level, which just contains the level number, and replace spaces with dashes
        level = level[level.find("\n")+1:].strip().replace(" ", "-")

        width = len(level.split("\n")[0])
        height = len(level.split("\n"))
        num_targets = level.count("$") # oddly, this seems to be 4 for every single level in the dataset!
        prop_empty = level.count("-") / (width * height)

        solution, node, _ = solver.getSolution(level_state, maxIterations=100000, maxTime=15)
        if node.checkWin():
            solution_len = len(solution)
        else:
            solution_len = -1
            solution = None

        solution_len = -1
        solution = None

        return level, level_text, level_hash, width, height, num_targets, prop_empty, solution_len, solution

    def _hash_level(self, level):
        return int(hashlib.md5(level.encode("utf-8")).hexdigest(), 16)

if __name__ == "__main__":
    B = BoxobanData("./data",
                    "./caches")