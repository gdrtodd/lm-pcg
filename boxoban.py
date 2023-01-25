import hashlib
import math
import os
import multiprocessing as mp
from multiprocessing import set_start_method, get_context
import hydra
import numpy as np
import pandas as pd
from tqdm import tqdm

from sokoban_solvers import EnhancedAStarAgent, State
from utils import encode_boxoban_text

class BoxobanData():
    def __init__(self,
                data_dir: str = './data', 
                save_dir: str = './caches/boxoban_data',
                n_proc: int = 1,
                level_file_idx: int = None,  # None value processes all files sequentially
                save_freq: int= None,  # Tradeoff between speed and safety! Complex levels will hog a process after others have
        ):  
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

        train_levels_dir = os.path.join(data_dir, "boxoban-medium", "train")
        val_levels_dir = os.path.join(data_dir, "boxoban-medium", "val")

        # All files sequentially
        if level_file_idx is None:
            cache_file = os.path.join(save_dir, "boxoban_data.h5")
            train_level_files = [os.path.join(train_levels_dir, file) for file in os.listdir(train_levels_dir) if file.endswith(".txt")]
        # Single file
        else:
            save_dir = os.path.join(cfg.save_dir, 'boxoban_data')
            cache_file = os.path.join(save_dir, f"boxoban_data_{level_file_idx:03}.h5")
            train_level_files = [os.path.join(train_levels_dir, f"{level_file_idx:03}.txt")]

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


        val_level_files = [os.path.join(val_levels_dir, file) for file in os.listdir(val_levels_dir) if file.endswith(".txt")]

        # Resume
        if os.path.exists(cache_file):
            self.data = pd.read_hdf(cache_file, key="data")
        else:
            self.data = pd.DataFrame(columns=data_keys)

        # Process train levels
        for level_file in train_level_files:
            print(f"\nProcessing {level_file}...")

            with open(level_file, "r") as f:
                raw_levels = [level for level in f.read().split("; ") if level != ""]

            filtered_levels = [level for level in raw_levels if self._hash_level(level) not in self.data["level_hash"].values]
            print(f"Already have values for {len(raw_levels) - len(filtered_levels)} levels, processing {len(filtered_levels)} levels...")

            if len(filtered_levels) == 0:
                continue
 
            n_levels = len(filtered_levels)
            if save_freq is None:
                save_freq = 1000
            n_save_chunks = math.ceil(n_levels / save_freq)

            for chunk_i in range(n_save_chunks):

                print(f"Processing file chunk {chunk_i+1}/{n_save_chunks}...")

                chunk_levels = filtered_levels[chunk_i*save_freq:(chunk_i+1)*save_freq]

                if n_proc == 1:
                    level_infos = [self._process_level(level) for level in tqdm(chunk_levels, desc="Processing levels")]

                with get_context("spawn").Pool(n_proc) as pool:
                    level_infos = list(tqdm(pool.imap(self._process_level, chunk_levels), 
                                            total=len(chunk_levels), desc="Processing levels"))

                    # Rearrange the level infos into a list of lists, where each list contains the specific value for every level
                    vals = list(zip(*level_infos))
                    
                    info_df = pd.DataFrame.from_dict(dict(zip(data_keys, vals)))
                    info_df["split"] = "train"

                    print("Num no solution:", info_df.loc[info_df["solution_len"] == -1].shape[0])

                    # Add to the main dataframe and save it progressively
                    self.data = pd.concat([self.data, info_df], ignore_index=True)

                print(f"Saving to {cache_file}...")
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

        solution, node, iterations = solver.getSolution(level_state, maxIterations=-1, maxTime=-1)
        if node.checkWin():
            solution_len = len(solution)
            print(f"Solved after {iterations} iterations.")
        else:
            solution_len = -1
            solution = None
            print(f"Failed after {iterations} iterations.")

        return level, level_text, level_hash, width, height, num_targets, prop_empty, solution_len, solution

    def _hash_level(self, level):
        return int(hashlib.md5(level.encode("utf-8")).hexdigest(), 16)


