import os
import shutil
from datasets import LMazeLMDataset
from utils import generate_l_mazes

def main(overwrite: bool):
    data_dir = os.path.join("./", "data", "l-mazes")

    # Automatically overwrite
    if os.path.isdir(data_dir):
        shutil.rmtree(data_dir)

    l_mazes, path_lens = [], []
    for width in range(4, 12):
        for height in range(4, 12):
            l_mazes_i, path_lens_i = generate_l_mazes(width, height)
            l_mazes += l_mazes_i
            path_lens += path_lens_i
            print(f"Generated {len(l_mazes_i)} L Mazes of size {height}x{width}")

            # Save text files for each maze size
            # with open(os.path.join(data_dir, f"l_mazes_{height}x{width}.txt"), "w") as f:
            #     f.write("\n\n".join(l_mazes_i))


    # Split into train/val according to path lengths, save files containing 1000 mazes each
    holdout_pls = LMazeLMDataset.holdout_path_lens

    train = []
    val = []

    pls_to_mazes = {}

    for pl, maze in zip(path_lens, l_mazes):
        if pl in holdout_pls:
            val.append(maze)
        else:
            train.append(maze)

        if pl in pls_to_mazes:
            pls_to_mazes[pl].append(maze)
        else:
            pls_to_mazes[pl] = [maze]

    os.makedirs(os.path.join(data_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "val"), exist_ok=True)

    for i, mazes in enumerate([train, val]):
        num_files = len(mazes) // 1000 + 1
        for j in range(num_files):
            filename = os.path.join(data_dir, ["train", "val"][i], f"l_mazes_{j}.txt")
            with open(filename, "w") as f:
                f.write("\n\n".join(mazes[j*1000:(j+1)*1000]))

            print(f"Saved {len(mazes[j*1000:(j+1)*1000])} L Mazes to {filename}")

    # Visualize the distribution of mazes over path lengths
    x, y = [], []
    pl_i = 1
    while pl_i in pls_to_mazes:
        mazes_i = pls_to_mazes[pl_i]
        x.append(pl_i)
        y.append(len(mazes_i))
        pl_i += 1


    import matplotlib.pyplot as plt
    plt.bar(x, y)
    plt.title("number of L mazes per path length")
    plt.xlabel("path length")
    plt.ylabel("number of L mazes")
    
    # Save figure
    plt.savefig(os.path.join(data_dir, "l_mazes_distribution.png"))
    print(f"Saved figure to {os.path.join(data_dir, 'l_mazes_distribution.png')}")

if __name__ == "__main__":
    main(overwrite=True)