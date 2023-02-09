import json
import matplotlib.pyplot as plt
from matplotlib.scale import LogScale
import numpy as np
import os
# from tensorflow.python.summary.summary_iterator import summary_iterator
from tqdm import tqdm

from utils import get_run_name

# def collect_loss_curves(run_dirs):
#     '''
#     For each of the runs in the specified directories, collect a list of the train loss at each recorded step,
#     and return them as a list of lists
#     '''

#     loss_curves = []

#     for run_dir in run_dirs:
#         if os.path.exists(run_dir) and any([file.startswith("events.out") for file in os.listdir(run_dir)]):

#             # Determine how many total logged steps there are by finding the events file the largest step
#             total_steps = max([max([e.step for e in summary_iterator(os.path.join(run_dir, file))]) for file in 
#                                     os.listdir(run_dir) if file.startswith("events.out")])

#             # Initialize the loss curve
#             loss_curve = np.zeros(total_steps)

#             for file in [file for file in os.listdir(run_dir) if file.startswith("events.out")]:
#                 for event in summary_iterator(os.path.join(run_dir, file)):
#                     if len(event.summary.value) > 0 and event.summary.value[0].tag == "train/loss":
#                         loss_curve[event.step-1] = event.summary.value[0].simple_value

#             # Count the number of unfilled positions
#             print(f"{run_dir.split('/')[-1]}: {np.sum(loss_curve == 0)} / {total_steps} unfilled loss curve positions")

#         loss_curves.append(loss_curve)

#     return loss_curves

            

class ExperimentArgs():
    def set(self, args):
        for k, v in args.items():
            setattr(self, k, v)

        return self

class ExperimentOneArgs(ExperimentArgs):
    game = "sokoban"
    source = "boxoban"
    level_key = "level"
    annotation_keys = None
    num_annotation_buckets = None
    holdout_solution_lens = None
    chunk_size = 128
    learning_rate = 1e-4
    sample_prop = None

    novelty_threshold = 5

def collect_experiment_one_data():
    print("=" * 80)
    print("EXPERIMENT ONE: EFFECTS OF PRETRAINING")
    print("=" * 80)

    args = ExperimentOneArgs()

    model_types = ["gpt2", "gpt2-untrained", "java-gpt2"]
    seeds = [0, 1, 2, 3, 4]

    for model_type in model_types:
        print(f"\nCollecting data for model type: {model_type}...")

        run_dirs = [os.path.join("./logs", get_run_name(args.set({"model": model_type, "seed": seed}))) for seed in seeds]
        print(f"Data exists for seeds {seeds}: {[os.path.exists(dir) for dir in run_dirs]}")
        loss_curves = collect_loss_curves(run_dirs)

        # Plot the loss curves
        plt.figure()
        for idx, loss_curve in enumerate(loss_curves):
            plt.plot(loss_curve, label=f"Seed {seeds[idx]}")
        plt.title(f"Loss curves for {model_type}")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"./results/experiment_1/{model_type}_loss_curves.png")
        plt.close()
        
        # Print number of eval jsons
        print(f"Number of eval jsons per seed: {[len([file for file in os.listdir(dir) if file.startswith('temp')]) for dir in run_dirs]}")

    # Ugh, figure out which eval runs didn't finish
    temps, topps, beams = [1, 2, 3, 4], [0.33, 0.66, 1], [5, 10, 15]
    for model_type in model_types:
        for seed in [0, 1, 2, 3]:
            print(f"\nChecking eval runs for model type [{model_type}] and seed [{seed}]...")
            run_dir = os.path.join("./logs", get_run_name(args.set({"model": model_type, "seed": seed})))
            eval_result_paths = [file for file in os.listdir(run_dir) if file.startswith("temp")]
            for temp, top_p, beam in [(temp, top_p, beam) for temp in temps for top_p in topps for beam in beams]:
                filename = f"temp-{float(temp)}_topk-{50}_topp-{float(top_p)}_typicalp-{1.0}_beams-{beam}_threshold-{5}.json"

                if filename not in eval_result_paths:
                    # print(f"-Missing: temp={temp}, top_p={top_p}, beam={beam}")
                    print(f"-Missing: {filename}")

    # Extract the eval sweep from seed 0 on the gpt2 model
    print("\n\nExtracting eval sweep from seed 0 on the gpt2 model...")
    run_dir = os.path.join("./logs", get_run_name(args.set({"model": "gpt2", "seed": 0})))
    eval_result_paths = [file for file in os.listdir(run_dir) if file.startswith("temp")]
    
    best_prop_novel_playable_accurate = 0
    best_restricted_diversity = 0
    best_params = {}
    best_results = {}

    for path in eval_result_paths:
        try:
            results = json.load(open(os.path.join(run_dir, path), "r"))
        except json.decoder.JSONDecodeError:
            print(f"Error decoding {path}... skipping")
            continue

        prop_novel_playable_accurate = results["prop_novel_playable_accurate"]
        restricted_diversity = results["restricted_diversity"]

        if restricted_diversity > best_restricted_diversity:
            best_restricted_diversity = restricted_diversity
            best_results = results
            
            temp, topk, topp, typicalp, beams, threshold = path[:-5].split("_")
            best_params = {"temp": float(temp.split("-")[1]),
                           "topk": int(topk.split("-")[1]),
                           "topp": float(topp.split("-")[1]),
                           "typicalp": float(typicalp.split("-")[1]),
                           "beams": int(beams.split("-")[1]),
                           "threshold": int(threshold.split("-")[1])}

    print(f"Best eval params: {best_params}")
    print(f"Best eval restricted diversity: {best_restricted_diversity}")
    print("Best eval results:")
    for k, v in list(best_results.items())[:7]:
        print(f"  {k}: {v}")


        


if __name__ == "__main__":
    collect_experiment_one_data()