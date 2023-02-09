from functools import partial
import json
import os
import shutil

from griddly import GymWrapperFactory
import gym
import hydra
from Levenshtein import distance
import matplotlib.pyplot as plt
from multiprocessing import Pool, get_context
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from conf.config import Config
from datasets import GameDataset, AnnotatedSokobanDataset
from utils import BOXOBAN_TO_GRIDDLY_CHARS, GRIDDLY_ACTION_MAPPING, get_run_name, load_train_state, save_gif


def evaluate(model: AutoModelForCausalLM, device, tokenizer: AutoTokenizer, dataset: GameDataset, args: Config, 
             num_steps_trained: int, verbose=False, render_dir=None, num_proc=1):

    # Map the model to the available device
    model.to(device)

    # Set up for evaluation
    model.eval()

    # Generate samples
    if verbose: print("Generating samples...")
    with torch.no_grad():

        if args.sample_contexts:
            contexts = torch.stack([tokenizer.encode(tokenizer.bos_token + dataset.gen_context(), return_tensors="pt").to(device) for 
                                    _ in range(args.num_eval_samples)], axis=0).squeeze(1)
            return_sequences = 1

        else:
            contexts = tokenizer.encode(tokenizer.bos_token + dataset.gen_context(), return_tensors="pt").to(device)
            return_sequences = args.num_eval_samples
        
        if args.sample_sequential and not args.sample_contexts:
            samples = [model.generate(
                contexts,
                max_length=args.gen_len,
                temperature=args.gen_temp,
                do_sample=True,
                top_k=args.gen_top_k,
                top_p=args.gen_top_p,
                typical_p=args.gen_typical_p,
                num_beams=args.gen_beams,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
            )[0] for _ in range(args.num_eval_samples)]

        else:
            samples = model.generate(
                contexts,
                max_length=args.gen_len,
                temperature=args.gen_temp,
                do_sample=True,
                top_k=args.gen_top_k,
                top_p=args.gen_top_p,
                typical_p=args.gen_typical_p,
                num_beams=args.gen_beams,
                num_return_sequences=return_sequences,
                pad_token_id=tokenizer.eos_token_id,
            )

    # Decode samples
    samples = [dataset.decode(sample) for sample in samples]

    

    if args.num_eval_proc == 1:
        if verbose: print("Computing solutions...")
        solutions = [dataset.get_solution(sample, verbose=False) for sample in samples]
        novelties, nearest_lvls, nearest_lvl_sols = zip(*[dataset.is_novel(sample) for sample in samples])
        accuracies, infos = zip(*[dataset.is_accurate(sample, solution, args.eval_tolerance) for sample, solution in zip(samples, solutions)])
    
    else:
        # FIXME: This makes things much slower (at least with num_eval_proc=10 or so -- just multiproc overhead?)
        with get_context("spawn").Pool(args.num_eval_proc) as pool:
            get_solution = partial(dataset.get_solution, verbose=False)
            solutions = list(tqdm(pool.imap(get_solution, samples), total=len(samples), desc="Computing solutions"))
            samples_sols = list(zip(samples, solutions, [args.eval_tolerance] * len(samples)))
            accuracies, infos = zip(*list(tqdm(pool.imap(dataset.is_accurate_multi, samples_sols), total=len(samples_sols), desc="Computing accuracies")))
            novelties, nearest_lvls, nearest_lvl_sols = zip(*list(tqdm(pool.imap(dataset.is_novel, samples), total=len(samples), desc="Computing novelties")))
    
    
    # Convert solutions to strings using the griddly action mapping
    solutions = ["" if sol is False else "".join([str(GRIDDLY_ACTION_MAPPING[(step['x'], step['y'])]) for step in sol]) for sol in solutions]
    nearest_lvl_sols = ["" if sol is False else "".join([str(GRIDDLY_ACTION_MAPPING[(step['x'], step['y'])]) for step in sol]) for sol in nearest_lvl_sols]

    num_accurate = sum(accuracies)
    num_playable = sum([len(sol) > 0 for sol in solutions])
    num_novel = sum(novelties)

    prop_accurate = num_accurate / len(samples)
    prop_playable = num_playable / len(samples)
    prop_novel = num_novel / len(samples)

    if verbose: print("Computing diversity...")
    num_diverse = dataset.get_diversity(samples)
    diversity = num_diverse / len(samples)

    # Compute the number of levels that are novel, playable, and accurate
    novel_playable_accurate_levels = [level for idx, level in enumerate(samples) if novelties[idx] and len(solutions[idx]) > 0 and accuracies[idx]]
    prop_novel_playable_accurate = len(novel_playable_accurate_levels) / len(samples)
    restricted_diversity = dataset.get_diversity(novel_playable_accurate_levels) / len(samples)

    if verbose:
        print("GENERATION PARAMETERS:")
        print(f"\tLength: {args.gen_len}")
        print(f"\tTemperature: {args.gen_temp}")
        print(f"\tTop-k: {args.gen_top_k}")
        print(f"\tTop-p: {args.gen_top_p}")
        print(f"\tTypical-p: {args.gen_typical_p}")
        print(f"\tBeams: {args.gen_beams}")

        for idx, sample in enumerate(samples):
            print("_" * 80)

            line_offset = len(sample.split("\n")[0]) - len("SAMPLE") # just for lining things up
            print(f"SAMPLE{' ' * line_offset}\t\t \t\tNEAREST LEVEL")
            for l1, l2 in zip(sample.split("\n"), nearest_lvls[idx].split("\n")):
                print(f"{l1.replace('-', ' ')}\t\t|\t\t{l2.replace('-', ' ')}")



            print(f"\nSample {idx + 1} of {args.num_eval_samples}")
            print(f"Playable: {solutions[idx] != ''}" + (f" ({len(solutions[idx])} steps)" if solutions[idx] != "" else ""))
            print(f"Novel: {novelties[idx]}")

            print(f"-Level edit distance: {distance(sample, nearest_lvls[idx])}")
            if solutions[idx]:
                print(f"-Solution edit distance: {distance(solutions[idx], nearest_lvl_sols[idx])}")

            print(f"Accurate: {accuracies[idx]}")
            if args.annotation_keys is not None:
                for key in args.annotation_keys:
                    print(f"\t{key}: {infos[idx][key]}")

        print("_" * 80)
        print(f"Proportion accurate: {prop_accurate}")
        print(f"Proportion playable: {prop_playable}")
        print(f"Proportion novel: {prop_novel}")
        print(f"Diversity (lower bound): {diversity}")
        print(f"\nPropotion novel, playable, and accurate: {prop_novel_playable_accurate}")
        print(f"Diversity (restricted): {restricted_diversity}")

    # Save stats to json
    stats = {
        "num_steps_trained": num_steps_trained,
        "novelty_threshold": dataset.novelty_threshold,
        "prop_accurate": prop_accurate,
        "prop_playable": prop_playable,
        "prop_novel": prop_novel,
        "prop_novel_playable_accurate": prop_novel_playable_accurate,
        "diversity": diversity,
        "restricted_diversity": restricted_diversity,
        "samples": samples,
        "solutions": solutions,
        "accuracies": accuracies,
        "novelties": novelties,
        "infos": infos,
    }


    # Save json to disc
    run_name = get_run_name(args)
    eval_filename = f"temp-{args.gen_temp}_topk-{args.gen_top_k}_topp-{args.gen_top_p}_typicalp-{args.gen_typical_p}_beams-{args.gen_beams}_threshold-{dataset.novelty_threshold}.json"
    stats_path = os.path.join('logs', run_name, eval_filename)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    # Render generated levels and animate solutions if applicable.
    if render_dir:
        if os.path.isdir(render_dir):
            # Delete old renderings
            shutil.rmtree(render_dir)
        os.makedirs(render_dir)

        trans_table = {ord(k): v for k, v in BOXOBAN_TO_GRIDDLY_CHARS.items()}
        wrapper = GymWrapperFactory()
        wrapper.build_gym_from_yaml('sokoban', os.path.join('gdy_games', 'sokoban.yaml'))
        env = gym.make('GDY-sokoban-v0')
        for i, (sample, sol) in enumerate(zip(samples, solutions)):

            lvl = sample.translate(trans_table)
            lvl_render_dir = os.path.join(render_dir, f"lvl_{i}_{len(sol)}-sol")
            save_gif(env, lvl, sol, lvl_render_dir)

            nearest_lvl = nearest_lvls[i].translate(trans_table)
            nearest_lvl_sol = nearest_lvl_sols[i]
            lvl_render_dir = os.path.join(render_dir, f"lvl_{i}_nearest_train")
            save_gif(env, nearest_lvl, sol=nearest_lvl_sol, lvl_render_dir=lvl_render_dir)

        env.close()
    model.train()
            
    return prop_accurate, prop_playable, prop_novel, diversity


def eval_controllability(model: AutoModelForCausalLM, device, tokenizer: AutoTokenizer, dataset: GameDataset, args: Config):
    '''
    Evaluate the controllability of the specifeid model by generating a number of levels for a variety of sample conditions,
    and generating a confusion matrix based on the results
    '''

    model.to(device)

    assert args.annotation_keys is not None, "Must specify annotation keys to evaluate controllability"

    # Initialize confusion matrix. Each row represents a generated annotation bin, and each column represents a target
    # annotation bin. We have an extra row for the "unplayable" bin
    confusion_matrix = np.zeros((11, 10))
    bottom_row_idx = confusion_matrix.shape[0] - 1

    if args.annotation_keys == ["solution_len"]:
        targets, width = list(range(5, 100, 10)), 10                                # middle of each of 10 bins from 0 to 100
        contexts = [dataset._format_annotation([target]) for target in targets]

    else:
        raise NotImplementedError

    with torch.no_grad():
        for context_idx, context in tqdm(enumerate(contexts), total=len(contexts), desc="Determining controllability"):
            samples = model.generate(
                tokenizer.encode(tokenizer.bos_token + dataset.gen_context(), return_tensors="pt").to(device),
                max_length=args.gen_len,
                temperature=args.gen_temp,
                do_sample=True,
                top_k=args.gen_top_k,
                top_p=args.gen_top_p,
                typical_p=args.gen_typical_p,
                num_beams=args.gen_beams,
                num_return_sequences=args.num_eval_samples,
                pad_token_id=tokenizer.eos_token_id,
            )

            samples = [dataset.decode(sample) for sample in samples]

            if args.num_eval_proc == 1:
                solutions = [dataset.get_solution(sample, verbose=False) for sample in tqdm(samples, total=len(samples), desc="Computing solutions",
                                                                                            leave=False)]
            
            else:
                # FIXME: This makes things much slower (at least with num_eval_proc=10 or so -- just multiproc overhead?)
                with get_context("spawn").Pool(args.num_eval_proc) as pool:
                    get_solution = partial(dataset.get_solution, verbose=False)
                    solutions = list(tqdm(pool.imap(get_solution, samples), total=len(samples), desc="Computing solutions", leave=False))

            solutions = ["" if sol is False else "".join([str(GRIDDLY_ACTION_MAPPING[(step['x'], step['y'])]) for step in sol]) for sol in solutions]

            if args.annotation_keys == ["solution_len"]:
                name = "Solution Length"
                for solution in solutions:
                    if len(solution) == 0:
                        observed_idx = bottom_row_idx # bottom row is for unplayable levels

                    else:
                        observed_idx = max(bottom_row_idx - int(len(solution) / width) - 1, 0)

                    confusion_matrix[observed_idx, context_idx] += 1


        # Generate the heatmap
        fig, ax = plt.subplots()
        im = ax.imshow(confusion_matrix)

        if args.annotation_keys == ["solution_len"]:
            limits = [(int(target-(width/2)+1), int(target+(width/2))) for target in targets]

        x_labels = [f"{lower}-{upper}" for lower, upper in limits]
        y_labels = [f"{lower}-{upper}" for lower, upper in reversed(limits)] + ["Unplayable"]

        # Show ticks
        ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
        ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)

        # Rotate the x-tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        ax.set_title(f"Controllability Confusion Matrix for {name}")
        ax.set_xlabel(f"Target {name}")
        ax.set_ylabel(f"Actual {name}")

        fig.tight_layout()
        plt.savefig(f"./results/{'+'.join(args.annotation_keys)}_controllability.png")
            



@hydra.main(version_base="1.2.0", config_path="conf", config_name="eval")
def main(args: Config):
    run_name = get_run_name(args)
    output_dir = f"./logs/{run_name}"

    model, _, global_step = load_train_state(output_dir)

    model_mapping = {"gpt2": "gpt2",
                     "gpt2-untrained": "gpt2-untrained",
                     "codeparrot": "lvwerra/codeparrot",
                     "java-gpt2": "microsoft/CodeGPT-small-java-adaptedGPT2",
                     "incoder-1B": "facebook/incoder-1B",
                     "incoder-6B": "facebook/incoder-6B"}

    # Instantiate the tokenizer based on the model's
    model_name = model_mapping[args.model]

    if args.model == "gpt2-untrained":
        tokenizer_dir = os.path.join("./caches", "gpt2-custom-tokenizer", args.game)

        # Load the custom tokenizer if it exists
        if os.path.exists(os.path.join(tokenizer_dir, "vocab.json")) and os.path.exists(os.path.join(tokenizer_dir, "merges.txt")):
            print(f"Loading tokenizer from cache at {tokenizer_dir}...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

            tokenizer.add_special_tokens({"pad_token": "<pad>",
                                          "bos_token": "<s>",
                                          "eos_token": "</s>"})
        else:
            exit("No custom tokenizer found for gpt2-untrained. Please run train_lm.py first.")

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens({"pad_token": "PAD",
                                    "bos_token": "START"})

    # Instantiate the dataset
    if args.game == "sokoban":
        dataset = AnnotatedSokobanDataset(args.source,
                                          tokenizer,
                                          args.model,
                                          level_key=args.level_key,
                                          annotation_keys=args.annotation_keys,
                                          num_annotation_buckets=args.num_annotation_buckets,
                                          holdout_solution_lens=args.holdout_solution_lens,
                                          split="train",
                                          novelty_threshold=args.novelty_threshold,
                                          sample_prop=args.sample_prop,
                                          chunk_size=args.chunk_size,
                                          seed=args.seed)

    else:
        raise NotImplementedError

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    render_dir = None
    if args.render:
        render_dir = os.path.join(output_dir, 'renders')

    if args.eval_controllability:
        eval_controllability(model, device, tokenizer, dataset, args)
    else:
        evaluate(model, device, tokenizer, dataset, args, verbose=True, num_steps_trained=global_step, 
                render_dir=render_dir, num_proc=args.num_eval_proc)

    # SIGSEGV ?? ... griddly?

    return

if __name__ == "__main__":
    main()