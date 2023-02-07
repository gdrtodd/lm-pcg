import glob
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
from utils import _hash_level, _process_level, encode_boxoban_text


def generate_sokoban_data(source="boxoban",
                          data_dir="./data",
                          save_dir="./caches",
                          n_proc=1,
                          level_file_idx=None,
                          save_freq=1000):


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
    data_keys = ["level", "level_text", "level_hash", "width", "height", "num_targets", "prop_empty", "solution_len", "solution"]

    if source == "boxoban":
        levels_dir = os.path.join(data_dir, "boxoban-medium", "train")
    elif source == "microban":
        levels_dir = os.path.join(data_dir, "microban")
    else:
        exit(f"Invalid source: {source}")

    # If specified, process only a single file
    if level_file_idx is not None:
        cache_file = os.path.join(save_dir, f"{source}_data_{level_file_idx:03}.h5")

        if os.path.exists(os.path.join(levels_dir, f"{level_file_idx:03}.txt")):
            level_files = [os.path.join(levels_dir, f"{level_file_idx:03}.txt")]
        else:
            exit(f"Specified level file does not exist: {levels_dir}/{level_file_idx:03}.txt")

    # Otherwise, process every file in the directory sequentially
    else:
        cache_file = os.path.join(save_dir, f"{source}_data.h5")
        level_files = [os.path.join(levels_dir, file) for file in os.listdir(levels_dir) if file.endswith(".txt")]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    # If a cache exists, load it. Otherwise create an empty dataframe
    if os.path.exists(cache_file):
        data = pd.read_hdf(cache_file, key="data")
    else:
        data = pd.DataFrame(columns=data_keys)

    # Process the specified level files
    for level_file in level_files:
        print(f"\nProcessing {level_file}...")

        with open(level_file, "r") as f:
            raw_levels = [level for level in f.read().split("; ") if level != ""]

        filtered_levels = [level for level in raw_levels if _hash_level(level) not in data["level_hash"].values]
        print(f"Already have values for {len(raw_levels) - len(filtered_levels)} levels, processing {len(filtered_levels)} levels...")

        if len(filtered_levels) == 0:
            continue

        # Process the levels in chunks, saving the results to the cache file after each chunk
        n_levels = len(filtered_levels)
        n_save_chunks = math.ceil(n_levels / save_freq)

        for chunk_i in range(n_save_chunks):

            print(f"Processing file chunk {chunk_i+1}/{n_save_chunks}...")

            chunk_levels = filtered_levels[chunk_i*save_freq:(chunk_i+1)*save_freq]

            if n_proc == 1:
                level_infos = [_process_level(level) for level in tqdm(chunk_levels, desc="Processing levels")]

            else:
                with get_context("spawn").Pool(n_proc) as pool:
                    level_infos = list(tqdm(pool.imap(_process_level, chunk_levels), 
                                            total=len(chunk_levels), desc="Processing levels"))

            # Rearrange the level infos into a list of lists, where each list contains the specific value for every level
            vals = list(zip(*level_infos))
            
            info_df = pd.DataFrame.from_dict(dict(zip(data_keys, vals)))
            info_df["split"] = "train"

            print("Num no solution:", info_df.loc[info_df["solution_len"] == -1].shape[0])

            # Add to the main dataframe and save it progressively
            data = pd.concat([data, info_df], ignore_index=True)

            print(f"Saving to {cache_file}...")
            data.to_hdf(cache_file, key="data")

        # Save the final dataframe
        print(f"Saving to {cache_file}...")
        data.to_hdf(cache_file, key="data")

    return data
    
def aggregate_sokoban_data(args):
    '''
    Collect the sokoban level files (distributed across the cache directory), and combine them into
    one file after removing duplicates
    '''

    saved_files = glob.glob(os.path.join(args.save_dir, f"{args.source}_data*.h5"))

    # Combine the data-frames row-wise (they must all have the same columns)
    data = pd.concat([pd.read_hdf(os.path.join(args.save_dir, file), key="data") for file in saved_files])

    # Ensure no two levels have the same hash and remove any duplicates
    print("Deduplicating...")
    n_pre_dedup = data.shape[0]
    data.drop_duplicates(subset="level_hash", inplace=True)
    n_post_dedup = data.shape[0]
    print(f"Removed {n_pre_dedup - n_post_dedup} duplicate levels.")

    # Save the data
    df_file = os.path.join(args.save_dir, f"{args.source}_data.h5")
    data.to_hdf(df_file, key="data")
    print(f"Saved {data.shape[0]} levels to {args.save_dir}.")

    # Inspect size of saved file
    print(f"Saved file size: {os.path.getsize(df_file) / 1e6} MB")


def augment_sokoban_data(args):
    '''
    Create two new dataframes, containing each of the levels in the originally specified dataframe
    except:
        1. flipped vertically and horizontally
        2. flipped vertically and horizontally, and rotated 90 degrees clockwise and counter-clockwise
    '''

    save_path = os.path.join(args.save_dir, f"{args.source}_data.h5")
    if not os.path.exists(save_path):
        exit(f"Data file does not exist at: {save_path}")

    base_dataframe = pd.read_hdf(save_path, key="data")
    num_entries = base_dataframe.shape[0]
    
    for idx in tqdm(range(num_entries), desc="Constructing flipped dataset"):
        entry = base_dataframe.iloc[idx]
        level_array = np.array([list(row) for row in entry["level"].split("\n")])

        # Flip vertically 
        vert_flip_array = np.flipud(level_array)
        vert_flip_level = "\n".join(["".join(row) for row in vert_flip_array])
        vert_flip_solution = [{'x': step['x'], 'y': -step['y']} for step in entry['solution']] if entry['solution'] is not None else None

        vert_flip_entry = entry.copy()
        vert_flip_entry["level"] = vert_flip_level
        vert_flip_entry["level_text"] = encode_boxoban_text(vert_flip_level)
        vert_flip_entry["level_hash"] = _hash_level(vert_flip_level)
        vert_flip_entry["solution"] = vert_flip_solution
        vert_flip_entry = pd.DataFrame(vert_flip_entry).transpose()

        # Flip horizontally
        horiz_flip_array = np.fliplr(level_array)
        horiz_flip_level = "\n".join(["".join(row) for row in horiz_flip_array])
        horiz_flip_solution = [{'x': -step['x'], 'y': step['y']} for step in entry['solution']] if entry['solution'] is not None else None

        horiz_flip_entry = entry.copy()
        horiz_flip_entry["level"] = horiz_flip_level
        horiz_flip_entry["level_text"] = encode_boxoban_text(horiz_flip_level)
        horiz_flip_entry["level_hash"] = _hash_level(horiz_flip_level)
        horiz_flip_entry["solution"] = horiz_flip_solution
        horiz_flip_entry = pd.DataFrame(horiz_flip_entry).transpose()

        # Add the new entries to the dataframe
        base_dataframe = pd.concat([base_dataframe, vert_flip_entry, horiz_flip_entry], ignore_index=True)

    # Save the dataframe with flips
    base_dataframe.to_hdf(os.path.join(args.save_dir, f"{args.source}_flips_data.h5"), key="data")

    def rotate_step_ccw(step):
        return {'x': -step['y'], 'y': step['x']}

    def rotate_step_cw(step):
        return {'x': step['y'], 'y': -step['x']}

    for idx in tqdm(range(num_entries), desc="Constructing flipped and rotated dataset"):
        entry = base_dataframe.iloc[idx]
        level_array = np.array([list(row) for row in entry["level"].split("\n")])

        # Rotate 90 degrees counter-clockwise
        rot_ccw_array = np.rot90(level_array)
        rot_ccw_level = "\n".join(["".join(row) for row in rot_ccw_array])
        rot_ccw_solution = [rotate_step_ccw(step) for step in entry['solution']] if entry['solution'] is not None else None

        rot_ccw_entry = entry.copy()
        rot_ccw_entry["level"] = rot_ccw_level
        rot_ccw_entry["level_text"] = encode_boxoban_text(rot_ccw_level)
        rot_ccw_entry["level_hash"] = _hash_level(rot_ccw_level)
        rot_ccw_entry["solution"] = rot_ccw_solution
        rot_ccw_entry["width"] = entry["height"]
        rot_ccw_entry["height"] = entry["width"]
        rot_ccw_entry = pd.DataFrame(rot_ccw_entry).transpose()

        # Rotate 90 degrees clockwise
        rot_cw_array = np.rot90(level_array, k=3)
        rot_cw_level = "\n".join(["".join(row) for row in rot_cw_array])
        rot_cw_solution = [rotate_step_cw(step) for step in entry['solution']] if entry['solution'] is not None else None

        rot_cw_entry = entry.copy()
        rot_cw_entry["level"] = rot_cw_level
        rot_cw_entry["level_text"] = encode_boxoban_text(rot_cw_level)
        rot_cw_entry["level_hash"] = _hash_level(rot_cw_level)
        rot_cw_entry["solution"] = rot_cw_solution
        rot_cw_entry["width"] = entry["height"]
        rot_cw_entry["height"] = entry["width"]
        rot_cw_entry = pd.DataFrame(rot_cw_entry).transpose()

        # Add the new entries to the dataframe
        base_dataframe = pd.concat([base_dataframe, rot_ccw_entry, rot_cw_entry], ignore_index=True)

    # Save the dataframe with flips and rotations
    base_dataframe.to_hdf(os.path.join(args.save_dir, f"{args.source}_flips_rotations_data.h5"), key="data")




@hydra.main(config_path="conf", config_name="boxoban_preprocessing")
def main(args):
    if args.aggregate:
        aggregate_sokoban_data(args)
    elif args.augment:
        augment_sokoban_data(args)
    else:
        generate_sokoban_data(args.source,
                              args.data_dir,
                              args.save_dir,
                              args.n_proc,
                              args.level_file_idx,
                              args.save_freq,)

if __name__ == "__main__":
    main()