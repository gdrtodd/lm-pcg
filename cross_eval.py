import json
import os
from pathlib import Path
from typing import List, Any, Sequence

import hydra
from hydra.core.utils import JobReturn
from omegaconf import OmegaConf
from hydra.core.plugins import Plugins
from hydra.plugins.plugin import Plugin
from matplotlib import pyplot as plt

from collect_experiment_data import collect_loss_curves
from conf.config import Config
from utils import get_run_name


@hydra.main(config_path="conf", config_name="cross_eval")
def cross_evaluate(sweep_configs: List[Config]):
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

    # Ugh, figure out which eval runs didn't finish
    temps, topps, beams = [1, 2, 3, 4], [0.33, 0.66, 1], [5, 10, 5]
    for cfg in sweep_configs:
        # FIXME: Organize these hierarchically by model, then seeds
        run_dirs = [os.path.join("./logs", get_run_name(cfg))]
        loss_curves = collect_loss_curves(run_dirs)

        # Plot the loss curves
        plt.figure()
        for idx, loss_curve in enumerate(loss_curves):
            plt.plot(loss_curve, label=f"Seed {cfg.seed}")
        plt.title(f"Loss curves for {cfg.model}")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(exp_results_dir, f"{cfg.model}_{cfg.seed}_loss_curve.png"))
        plt.close()
        
        # Print number of eval jsons
        print(f"Number of eval jsons per seed: {[len([file for file in os.listdir(dir) if file.startswith('temp')]) for dir in run_dirs]}")

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
    cross_evaluate()
