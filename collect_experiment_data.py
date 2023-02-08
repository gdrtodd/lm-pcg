import json
import numpy as np
import os
from tensorflow.python.summary.summary_iterator import summary_iterator
from tqdm import tqdm

from utils import get_run_name

def collect_loss_curves(run_dirs):
    '''
    For each of the runs in the specified directories, collect a list of the train loss at each recorded step,
    and return them as a list of lists
    '''

    loss_curves = []

    for run_dir in run_dirs:
        if os.path.exists(run_dir) and any([file.startswith("events.out") for file in os.listdir(run_dir)]):

            # Determine how many total logged steps there are by finding the events file the largest step
            total_steps = max([max([e.step for e in summary_iterator(os.path.join(run_dir, file))]) for file in 
                                    os.listdir(run_dir) if file.startswith("events.out")])

            # Initialize the loss curve
            loss_curve = np.zeros(total_steps)

            for file in [file for file in os.listdir(run_dir) if file.startswith("events.out")]:
                for event in summary_iterator(os.path.join(run_dir, file)):
                    if len(event.summary.value) > 0 and event.summary.value[0].tag == "train/loss":
                        loss_curve[event.step-1] = event.summary.value[0].simple_value

            # Count the number of unfilled positions
            print(f"{run_dir.split('/')[-1]}: {np.sum(loss_curve == 0)} / {total_steps} unfilled loss curve positions")

        loss_curves.append(loss_curve)

    return loss_curves

            

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
    seeds = [5, 6, 7, 8, 9]

    for model_type in model_types:
        print(f"\nCollecting data for model type: {model_type}...")

        run_dirs = [os.path.join("./logs", get_run_name(args.set({"model": model_type, "seed": seed}))) for seed in seeds]
        print(f"Data exists for seeds {seeds}: {[os.path.exists(dir) for dir in run_dirs]}")
        loss_curves = collect_loss_curves(run_dirs)
        print(f"Number of steps run for each seed: {[len(curve) for curve in loss_curves]}")

    # Extract the eval sweep from seed 5 on the gpt2 model
    print("\n\nExtracting eval sweep from seed 5 on the gpt2 model...")
    run_dir = os.path.join("./logs", get_run_name(args.set({"model": "gpt2", "seed": 5})))
    eval_result_paths = [file for file in os.listdir(run_dir) if file.startswith("temp")]
    
    best_prop_novel_playable_accurate = 0
    best_restricted_diversity = 0
    best_params = {}
    best_results = {}

    for path in eval_result_paths:
        results = json.load(open(os.path.join(run_dir, path), "r"))
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
    for k, v in list(best_results.items())[:6]:
        print(f"  {k}: {v}")


        


if __name__ == "__main__":
    collect_experiment_one_data()