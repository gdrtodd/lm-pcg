from functools import partial
import json
import os
import shutil

from griddly import GymWrapperFactory
import gym
import hydra
from Levenshtein import distance
from multiprocessing import Pool, get_context
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from conf.config import Config
from datasets import GameDataset, AnnotatedSokobanDataset
from utils import BOXOBAN_TO_GRIDDLY_CHARS, GRIDDLY_ACTION_MAPPING, get_run_name, load_train_state, save_gif


def evaluate(model: AutoModelForCausalLM, device, tokenizer: AutoTokenizer, dataset: GameDataset, args: Config, 
             verbose=False, render_dir=None, num_proc=1):

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
        accuracies, infos = zip(*[dataset.is_accurate(sample, solution) for sample, solution in zip(samples, solutions)])
    
    else:
        # FIXME: This makes things much slower (at least with num_eval_proc=10 or so -- just multiproc overhead?)
        with get_context("spawn").Pool(args.num_eval_proc) as pool:
            get_solution = partial(dataset.get_solution, verbose=False)
            solutions = list(tqdm(pool.imap(get_solution, samples), total=len(samples), desc="Computing solutions"))
            samples_sols = list(zip(samples, solutions))
            accuracies, infos = zip(*list(tqdm(pool.imap(dataset.is_accurate_multi, samples_sols), total=len(samples_sols), desc="Computing accuracies")))
            novelties, nearest_lvls, nearest_lvl_sols = zip(*list(tqdm(pool.imap(dataset.is_novel, samples), total=len(samples), desc="Computing novelties")))
    
    
    # Convert solutions to strings using the griddly action mapping
    # solutions = [[] if sol is False else sol for sol in solutions]

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

    evaluate(model, device, tokenizer, dataset, args, verbose=True, render_dir=render_dir, num_proc=args.num_eval_proc)

    # SIGSEGV ?? ... griddly?

    return

if __name__ == "__main__":
    main()