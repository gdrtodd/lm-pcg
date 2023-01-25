import os

import hydra
import pandas as pd

from boxoban import BoxobanData


def gen_solved_boxoban_data(cfg):
    B = BoxobanData(cfg.data_dir, cfg.save_dir, cfg.n_proc,  cfg.level_file_idx)


def aggregate_boxoban_data(cfg):
    save_subdir = os.path.join(cfg.save_dir, 'boxoban_data')
    saved_data = os.listdir(save_subdir)
    # Combine the data-frames row-wise (they must all have the same columns)
    data = pd.concat([pd.read_hdf(os.path.join(save_subdir, file), key="data") for file in saved_data])

    # Ensure no two levels have the same hash and remove any duplicates
    print("Deduplicating...")
    n_pre_dedup = data.shape[0]
    data.drop_duplicates(subset="level_hash", inplace=True)
    n_post_dedup = data.shape[0]
    print(f"Removed {n_pre_dedup - n_post_dedup} duplicate levels.")

    # Inspect size of dataframe in memory (this is a lie?)
    # print(f"Dataframe size: {data.memory_usage().sum() / 1e6} MB")

    # Save the data
    df_file = os.path.join(cfg.save_dir, "boxoban_data.h5")
    data.to_hdf(df_file, key="data")
    print(f"Saved {data.shape[0]} levels to {cfg.save_dir}.")
    # Inspect size of saved file
    print(f"Saved file size: {os.path.getsize(df_file) / 1e6} MB")

    # Plot the distribution of solution lengths
    # Get the number of levels with each solution length
    sol_lens = data["solution_len"].value_counts()
    # Are there any "holes" in the path-lengths in the dataset? Print them
    for i in range(sol_lens.index.min(), sol_lens.index.max() + 1):
        if i not in sol_lens.index:
            print(f"Solution length {i} not found in dataset.")
    # Plot the distribution
    import matplotlib.pyplot as plt
    plt.bar(sol_lens.index, sol_lens.values)
    plt.title("number of Boxoban levels per solution length")
    plt.xlabel("solution length")
    plt.ylabel("number of Boxoban levels")
    im_path = os.path.join(cfg.data_dir, 'boxoban-medium', 'boxoban_sol_len_dist.png')
    plt.savefig(im_path)
    print(f"Saved figure to {im_path}.")
    plt.close()


@hydra.main(config_path="conf", config_name="boxoban_preprocessing")
def main(cfg):
    if not cfg.aggregate:
        gen_solved_boxoban_data(cfg)
    else:
        aggregate_boxoban_data(cfg)


if __name__ == "__main__":
    main()
