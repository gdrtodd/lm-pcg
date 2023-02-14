import copy
import glob
import json
import os
from typing import Dict, Iterable, List, Any, Sequence

import hydra
from omegaconf import ListConfig
import pandas as pd
import yaml

from conf.config import Config, CrossEvalConfig, EvalConfig
from utils import filter_configs, process_hyperparam_str, sort_hyperparams, get_run_name, is_valid_config


def report_progress(sweep_configs: List[Config], hyperparams: Iterable):
    """ Check whether we have a checkpoint trained to the target number of steps for each experiment."""

    # Which sets of hyperparameters have we already checked for?
    param_tpls_checked = set()

    for cfg in sweep_configs:

        # Skip this experiment if we've already encountered an equivalent one according to `hyperparams`
        param_tpl = tuple([cfg[k] for k in hyperparams])
        if param_tpl in param_tpls_checked:
            continue
        param_tpls_checked.add(param_tpl)

        run_dir = os.path.join("./logs", get_run_name(cfg))
        ckpt_file = os.path.join(run_dir, f"checkpoint-{cfg.num_train_steps}")
        if not os.path.exists(run_dir):
            print(f"Run directory not found for {' '.join([f'{k}:{cfg[k]}' for k in hyperparams])}")
        elif os.path.exists(ckpt_file):
            print(f"Checkpoint found for {' '.join([f'{k}:{cfg[k]}' for k in hyperparams])} at {cfg.num_train_steps} steps")
        else:
            # Get any other checkpoint files
            # glob misbehaves when run_dir contains, e.g. annotation_keys:['solution_len']. Huh!
            # ckpt_files = glob.glob(os.path.join(run_dir, "checkpoint-*"))
            ckpt_files = [os.path.join(run_dir, file) for file in os.listdir(run_dir) if file.startswith("checkpoint-")]

            # Get only the file name of each path above
            ckpt_files = [os.path.basename(file) for file in ckpt_files]
            print(f"Checkpoint not found for {' '.join([f'{k}:{cfg[k]}' for k in hyperparams])}. Saved checkpoints: {ckpt_files}")




def filter_incomplete(sweep_configs: List[Config], eval_data_dicts: List[Dict], min_steps_trained: int):
    filtered_configs, filtered_eval_data_dicts = [], []
    for config, eval_data in zip(sweep_configs, eval_data_dicts):
        run_dir = os.path.join("./logs", config.run_name)

        if 'num_steps_trained' not in eval_data:
            print(f"Missing num_steps_trained in eval data. Skipping config.")
            continue
        
        # HACK for backward compat. FIXME delete me
        if 'less_restricted_diversity' not in eval_data:
            continue

        if eval_data['num_steps_trained'] < min_steps_trained:
            print(f"Skipping config because it was not trained for enough steps. Trained for {eval_data['num_steps_trained']} steps, but should have trained for {config.num_train_steps} steps.")
            continue

        # if eval_data['num_steps_trained'] > cross_eval_config.num_train_steps + 10:
        #     print(f"Skipping config because it was trained for too many steps. Trained for {eval_data['num_steps_trained']} steps, but should have trained for {config.num_train_steps} steps.")
        #     continue

        filtered_configs.append(config)
        filtered_eval_data_dicts.append(eval_data)

    return filtered_configs, filtered_eval_data_dicts

@hydra.main(version_base=None, config_path="conf", config_name="cross_eval")
def main(cross_eval_config: CrossEvalConfig):

    # Load up eval hyperparameters from conf/eval.yaml
    eval_sweep_params = yaml.load(open("conf/eval.yaml", "r"), Loader=yaml.FullLoader)['hydra']['sweeper']['params']
    train_sweep = yaml.load(open(f"conf/experiment/{cross_eval_config.sweep}.yaml"), Loader=yaml.FullLoader)
    train_sweep_params = train_sweep['hydra']['sweeper']['params']

    eval_sweep_params = {k: process_hyperparam_str(v) for k, v in eval_sweep_params.items()}
    train_sweep_params = {k: process_hyperparam_str(v) for k, v in train_sweep_params.items()}
    sweep_params = {**train_sweep_params, **eval_sweep_params}

    # Manually create per-experiment configs.
    sweep_configs = [copy.deepcopy(cross_eval_config)]
    for param_k, param_v_lst in sweep_params.items():
        new_sweep_configs = []

        # Take product of existing configs with new hyperparameter values
        for param_v in param_v_lst:
            for old_cfg in sweep_configs:
                new_cfg = copy.deepcopy(old_cfg)
                new_cfg[param_k] = param_v
                new_sweep_configs.append(new_cfg) 
        sweep_configs = new_sweep_configs

    # Filter out any invalid configs.
    sweep_configs = filter_configs(sweep_configs)

    exp_name = train_sweep['exp_name']

    print("=" * 80)
    print(f"EXPERIMENT: {exp_name}")
    print("=" * 80)

    exp_results_dir = os.path.join("./results", exp_name)
    if not os.path.exists(exp_results_dir):
        os.makedirs(exp_results_dir)

    # Names of the hyperparameters being swept over in the experiment
    hyperparams = [k for k in sweep_params.keys()]

    # Prioritize the order of the hyperparameters
    hyperparams = sort_hyperparams(hyperparams)
    
    # Sort the configs according to the order of relevant hyperparameters above
    _cfgs_sortable = [tuple([cfg[k] for k in hyperparams]) for cfg in sweep_configs]
    
    # Convert iterables to strings for sorting
    _cfgs_sortable = [[''.join(str(cfg[k])) if isinstance(cfg[k], Iterable) else cfg[k] for k in hyperparams] for cfg in sweep_configs]
    _cfg_sort_idxs = sorted(range(len(_cfgs_sortable)), key=lambda k: _cfgs_sortable[k])

    # List of configurations sorted by the order of the hyperparameters
    sweep_configs = [sweep_configs[i] for i in _cfg_sort_idxs]

    # Report training progress
    if not cross_eval_config.gen_table:
        report_progress(sweep_configs, train_sweep_params.keys())
        return

    # Create a dataframe that holds the results of each evaluation
    main_dataframe = []
 
    filtered_configs = []
    eval_data_dicts = []
    for config in sweep_configs:
        run_dir = os.path.join("./logs", config.run_name)
        eval_json_filename = f"temp-{config.gen_temp}_topk-{config.gen_top_k}_topp-{config.gen_top_p}_typicalp-{config.gen_typical_p}_beams-{config.gen_beams}_threshold-{config.novelty_threshold}.json"
        
        try:
            eval_data = json.load(open(os.path.join(run_dir, eval_json_filename), "r"))
        except (json.decoder.JSONDecodeError, FileNotFoundError) as error:
            print(f"Issue loading JSON: {error}")
            continue
        eval_data_dicts.append(eval_data)
        filtered_configs.append(config)
    sweep_configs = filtered_configs
    
    sweep_configs, eval_data_dicts = filter_incomplete(sweep_configs, eval_data_dicts, min_steps_trained=cross_eval_config.num_train_steps)

    filtered_configs = []
    for config, eval_data in zip(sweep_configs, eval_data_dicts):

        eval_dict = {
                     "gen_temp": config.gen_temp,
                     "gen_top_p": config.gen_top_p,
                     "gen_beams": config.gen_beams,
                     "novelty_threshold": config.novelty_threshold,
                     "prop_novel": eval_data["prop_novel"],
                     "prop_playable": eval_data["prop_playable"],
                     "prop_accurate": eval_data["prop_accurate"],
                     "prop_novel_playable_accurate": eval_data["prop_novel_playable_accurate"],
                     "diversity": eval_data["diversity"],
                     "restricted_diversity": eval_data["restricted_diversity"],
                     "less_restricted_diversity": eval_data["less_restricted_diversity"],
                    }
        filtered_configs.append(config)

        main_dataframe.append(eval_dict)

    sweep_configs = filtered_configs

    if len(main_dataframe) == 0:
        raise ValueError("No valid evaluation results found.")

    # Row indices will consist of training hyperparameters
    row_index_names = list(train_sweep_params.keys())
    row_tuples = [tuple([cfg[param] for param in row_index_names]) for cfg in sweep_configs]
    
    # Turn ListConfigs into strings. Remove any underscores.
    row_tuples = [tuple([', '.join(cfg[param]) if isinstance(cfg[param], ListConfig) else cfg[param] for param in row_index_names]) for cfg in sweep_configs]
    row_tuples = [tuple([v.replace("_", " ") if isinstance(v, str) else v for v in tpl]) for tpl in row_tuples]

    row_indices = pd.MultiIndex.from_tuples(row_tuples, names=row_index_names)

    main_dataframe = pd.DataFrame(main_dataframe, index=row_indices)
    main_dataframe.to_html("./results/main_dataframe.html")

    # First, we group by the swept hyperparameters except the seed and take the average. This gives us the average evaluation scores across
    # the seeds, for each setting of the evaluation hyperparameters
    hyperparams.remove("seed")
    average_over_seeds = main_dataframe.groupby(hyperparams).mean(numeric_only=True).reset_index()
    average_over_seeds.to_html("./results/new_df.html")

    # Group by the remaining non-evaluation hyperparameters and take the max with respect to prop_novel_playable_accurate
    [hyperparams.remove(param) for param in ["gen_temp", "gen_top_p", "gen_beams"] if param in hyperparams]
    max_over_eval_hyperparams = average_over_seeds.groupby(hyperparams)
    max_over_eval_hyperparams = max_over_eval_hyperparams.apply(lambda x: x.loc[x.restricted_diversity.idxmax()])

    eval_columns = ["prop_novel", "prop_playable", "prop_accurate", "diversity", "less_restricted_diversity", "restricted_diversity"]

    # If we're not controlling with prompts, exclude accuracy
    if cross_eval_config.sweep != "controls":
        eval_columns.remove("prop_accurate")
        eval_columns.remove("restricted_diversity")

    # For display purposes, restrict to just model, novelty, playability, accuracy, all three, and diversity
    to_display = max_over_eval_hyperparams[eval_columns]

    # Rename the columns to be more readable and save to LaTeX
    to_display = to_display.rename(columns={
                                            "prop_novel": "Novelty",
                                            "prop_playable": "Playability",
                                            "prop_accurate": "Accuracy",
                                            "diversity": "Diversity",
                                            "less_restricted_diversity": "Score",
                                            "restricted_diversity": "Control Score"},
                                    index={'model': 'Model',
                                           'annotation_keys': 'Annotation Keys'},
                                    )

    row_index_rename = {'annotation_keys': 'Controls'}

    to_display.index.names = [row_index_rename.get(v, v) for v in to_display.index.names]
    to_display.index.names = [v.replace("_", " ") for v in to_display.index.names]

    # Bold the max values
    to_display = to_display.style.highlight_max(axis=0,
                        props='bfseries: ;')

    # Round to 3 decimal places
    to_display = to_display.format("{:.2f}")

    # Also separately record the eval hyperparameters
    best_eval_hyperparams = max_over_eval_hyperparams[["gen_temp", "gen_top_p", "gen_beams"]]

    # Set up the save directory
    save_dir = os.path.join("./results", exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # to_display = to_display.style.hide(axis='index')
    to_display = to_display.to_latex(os.path.join(save_dir, "eval_sweep_table.tex"),
                                     hrules=True)

    best_eval_hyperparams.to_csv(os.path.join(save_dir, "eval_hyperparams.csv"))

    print("Done!")



# Run `python cross_eval.py +experiment=EXP_NAME -m`
if __name__ == "__main__":
    main()
