import os

import hydra
import pandas as pd

from boxoban import BoxobanData


@hydra.main(config_path="conf", config_name="boxoban_preprocessing")
def gen_solved_boxoban_data(cfg):
    B = BoxobanData(cfg.data_dir, cfg.save_dir, cfg.n_proc,  cfg.level_file_idx)


@hydra.main(config_path="conf", config_name="boxoban_preprocessing")
def aggregate_boxoban_data(cfg):
    save_subdir = os.path.join(cfg.save_dir, 'boxoban_data')
    saved_data = os.listdir(save_subdir)
    # Combine the data-frames row-wise (they must all have the same columns)
    data = pd.concat([pd.read_hdf(os.path.join(save_subdir, file), key="data") for file in saved_data])

    # Ensure no two levels have the same hash and remove any duplicates
    n_pre_dedup = data.shape[0]
    data.drop_duplicates(subset="level_hash", inplace=True)
    n_post_dedup = data.shape[0]
    print(f"Removed {n_pre_dedup - n_post_dedup} duplicate levels.")

    # Save the data
    data.to_hdf(os.path.join(cfg.save_dir, "boxoban_data.h5"), key="data")

    print(f"Saved {data.shape[0]} levels to {cfg.save_dir}.")


if __name__ == "__main__":

    # gen_solved_boxoban_data()
    aggregate_boxoban_data()
