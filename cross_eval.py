import glob
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Any, Sequence

import hydra
from hydra.core.utils import JobReturn
import numpy as np
from omegaconf import OmegaConf
from hydra.core.plugins import Plugins
from hydra.plugins.plugin import Plugin
from matplotlib import pyplot as plt

from collect_experiment_data import collect_loss_curves
from conf.config import Config
from utils import get_run_name


def collect_checkpoints(sweep_configs: List[Config], hyperparams):
    # Check whether we have a checkpoint trained to the target number of steps for each experiment.
    for cfg in sweep_configs:
        run_dir = os.path.join("./logs", get_run_name(cfg))
        ckpt_file = os.path.join(run_dir, f"checkpoint-{cfg.num_train_steps}")
        if os.path.exists(ckpt_file):
            print(f"Checkpoint found for {' '.join([f'{k}:{cfg[k]}' for k in hyperparams])} at {cfg.num_train_steps} steps")
        else:
            # Get any other checkpoint files
            ckpt_files = glob.glob(os.path.join(run_dir, "checkpoint-*"))
            
            # Get only the file name of each path above
            ckpt_files = [os.path.basename(file) for file in ckpt_files]
            print(f"Checkpoint not found for {' '.join([f'{k}:{cfg[k]}' for k in hyperparams])}. Saved checkpoints: {ckpt_files}")


# Call this only to initiate the hydra multirun launcher (which calls cross_eval()). This function is never entered.
# At runtime, the cross_eval config inherits from eval.yaml, so it knows what hyperparameters we have swept over
@hydra.main(version_base=None, config_path="conf", config_name="cross_eval")
def dummy_cross_eval(cfg): pass 


def cross_evaluate(sweep_configs: List[Config], sweep_params: Dict[str, str]):
    # TODO: Could be nice to also receive the hyperparams swept over, as lists. Can tinker with `cross_eval_launcher.py`
    #   to pass this forward, or just iterate through sweep_configs and collect hyperparams of interest manually.

    cfg_0 = sweep_configs[0]

    exp_name = cfg_0.exp_name
    print("=" * 80)
    print(f"EXPERIMENT: {exp_name}")
    print("=" * 80)

    exp_results_dir = os.path.join("./results", exp_name)
    if not os.path.exists(exp_results_dir):
        os.makedirs(exp_results_dir)

    # Names of the hyperparameters being swept over in the experiment
    hyperparams = [k for k in sweep_params.keys()]

    # Prioritize the order of the hyperparameters
    hyperparam_sort_order = ['model', 'sample_prop', 'annotation_keys', 'seed', 'gen_temp', 'gen_top_p', 'gen_beams']
    hyperparams = sorted(hyperparams, key=lambda k: hyperparam_sort_order.index(k) if k in hyperparam_sort_order else len(hyperparam_sort_order))

    # Sort the configs according to the order of relevant hyperparameters above
    _cfgs_sortable = [tuple([cfg[k] for k in hyperparams]) for cfg in sweep_configs]
    
    # Convert iterables to strings for sorting
    _cfgs_sortable = [[''.join(cfg[k]) if isinstance(cfg[k], Iterable) else cfg[k] for k in hyperparams] for cfg in sweep_configs]
    _cfg_sort_idxs = sorted(range(len(_cfgs_sortable)), key=lambda k: _cfgs_sortable[k])
    sweep_configs = [sweep_configs[i] for i in _cfg_sort_idxs]

    collect_checkpoints(sweep_configs, hyperparams)
    return


    for cfg in sweep_configs:
        print(f"\nChecking eval runs for model type [{cfg.model}] and seed [{cfg.seed}]...")
        run_dir = os.path.join("./logs", get_run_name(cfg))
        eval_result_paths = [file for file in os.listdir(run_dir) if file.startswith("temp")]
        for temp, top_p, beam in [(temp, top_p, beam) for temp in temps for top_p in topps for beam in beams]:
            filename = f"temp-{float(temp)}_topk-{50}_topp-{float(top_p)}_typicalp-{1.0}_beams-{beam}_threshold-{5}.json"

            if filename not in eval_result_paths:
                # print(f"-Missing: temp={temp}, top_p={top_p}, beam={beam}")
                print(f"Missing: {filename}")

    for cfg in sweep_configs:
        print(f"\n\nExtracting eval sweep from seed {cfg.seed} on the {cfg.model} model...")
        run_dir = os.path.join("./logs", get_run_name(cfg))
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
        for k, v in list(best_results.items())[:7]:
            print(f"  {k}: {v}")


    # TODO: Generate pandas dataframe --> latex table



if __name__ == "__main__":
    dummy_cross_eval()
