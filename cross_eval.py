import glob
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Any, Sequence

import hydra
from hydra.core.utils import JobReturn
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
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
            # glob misbehaves when run_dir contains, e.g. annotation_keys:['solution_len']. Huh!
            # ckpt_files = glob.glob(os.path.join(run_dir, "checkpoint-*"))
            ckpt_files = [os.path.join(run_dir, file) for file in os.listdir(run_dir) if file.startswith("checkpoint-")]

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

    # List of configurations sorted by the order of the hyperparameters
    sweep_configs = [sweep_configs[i] for i in _cfg_sort_idxs]

    # Create a dataframe that holds the results of each evaluation
    main_dataframe = []
 

    for config in sweep_configs:
        run_dir = os.path.join("./logs", get_run_name(config))
        eval_json_filename = f"temp-{config.gen_temp}_topk-{config.gen_top_k}_topp-{config.gen_top_p}_typicalp-{config.gen_typical_p}_beams-{config.gen_beams}_threshold-{config.novelty_threshold}.json"
        
        try:
            eval_data = json.load(open(os.path.join(run_dir, eval_json_filename), "r"))
        except (json.decoder.JSONDecodeError, FileNotFoundError) as error:
            print(f"Issue loading JSON: {error}")
            continue

        eval_dict = {"model": config.model,
                     "sample_prop": config.sample_prop,
                     "annotation_keys": config.annotation_keys,
                     "seed": config.seed,
                     "gen_temp": config.gen_temp,
                     "gen_top_p": config.gen_top_p,
                     "gen_beams": config.gen_beams,
                     "novelty_threshold": config.novelty_threshold,
                     "prop_novel": eval_data["prop_novel"],
                     "prop_playable": eval_data["prop_playable"],
                     "prop_accurate": eval_data["prop_accurate"],
                     "prop_novel_playable_accurate": eval_data["prop_novel_playable_accurate"],
                     "diversity": eval_data["diversity"],
                     "restricted_diversity": eval_data["restricted_diversity"]}

        main_dataframe.append(eval_dict)

    main_dataframe = pd.DataFrame(main_dataframe)
    main_dataframe.to_html("./results/main_dataframe.html")

    # First, we group by the swept hyperparameters except the seed and take the average. This gives us the average evaluation scores across
    # the seeds, for each setting of the evaluation hyperparameters
    hyperparams.remove("seed")
    average_over_seeds = main_dataframe.groupby(hyperparams).mean(numeric_only=True).reset_index()
    average_over_seeds.to_html("./results/new_df.html")

    # Group by the remaining non-evaluation hyperparameters and take the max with respect to prop_novel_playable_accurate
    [hyperparams.remove(param) for param in ["gen_temp", "gen_top_p", "gen_beams"]]
    max_over_eval_hyperparams = average_over_seeds.groupby(hyperparams)
    max_over_eval_hyperparams = max_over_eval_hyperparams.apply(lambda x: x.loc[x.restricted_diversity.idxmax()]).reset_index(drop=True)

    # For display purposes, restrict to just model, novelty, playability, accuracy, all three, and diversity
    to_display = max_over_eval_hyperparams[["model", "prop_novel", "prop_playable", "prop_accurate", "restricted_diversity"]]

    # Rename the columns to be more readable and save to LaTeX, bolding the highest value in each column
    to_display = to_display.rename(columns={"model": "Model",
                                            "prop_novel": "Novelty",
                                            "prop_playable": "Playability",
                                            "prop_accurate": "Accuracy",
                                            "restricted_diversity": "Score"})
    to_display = to_display.round(3)

    # Also separately record the eval hyperparameters
    eval_hyperparams = max_over_eval_hyperparams[["model", "gen_temp", "gen_top_p", "gen_beams"]]

    # Set up the save directory
    save_dir = os.path.join("./results", exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    to_display = to_display.style.to_latex(os.path.join(save_dir, "eval_sweep_table.tex"),
                                           index=False, escape=False, bold_rows=True, 
                                           column_format="lrrrr")

    eval_hyperparams.to_csv(os.path.join(save_dir, "eval_hyperparams.csv"))

    print("Done!")



if __name__ == "__main__":
    dummy_cross_eval()
