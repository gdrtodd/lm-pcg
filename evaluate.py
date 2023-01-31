import os

from griddly import GymWrapperFactory
import gym
import hydra
from multiprocessing import Pool
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from conf.config import Config
from datasets import GameDataset, AnnotatedSokobanDataset
from utils import BOXOBAN_TO_GRIDDLY_CHARS, GRIDDLY_ACTION_MAPPING, get_run_name, load_train_state


def evaluate(model: AutoModelForCausalLM, device, tokenizer: AutoTokenizer, dataset: GameDataset, args: Config, 
             verbose=False, render_dir=None):

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

    if verbose: print("Computing solutions...")
    solutions = [dataset.get_solution(sample) for sample in samples]

    novelties = [dataset.is_novel(sample) for sample in samples]
    accuracies, infos = zip(*[dataset.is_accurate(sample, solution) for sample, solution in zip(samples, solutions)])

    num_accurate = sum(accuracies)
    num_playable = sum([sol != False for sol in solutions])
    num_novel = sum(novelties)

    prop_accurate = num_accurate / len(samples)
    prop_playable = num_playable / len(samples)
    prop_novel = num_novel / len(samples)
    diversity = dataset.get_diversity(samples)

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
            print(sample)
            print(f"\nSample {idx + 1} of {args.num_eval_samples}")
            print(f"Playable: {solutions[idx] != False}" + (f" ({len(solutions[idx])} steps)" if solutions[idx] != False else ""))
            print(f"Novel: {novelties[idx]}")
            print(f"Accurate: {accuracies[idx]}")
            for key in args.annotation_keys:
                print(f"\t{key}: {infos[idx][key]}")

        print("_" * 80)
        print(f"Proportion accurate: {prop_accurate}")
        print(f"Proportion playable: {prop_playable}")
        print(f"Proportion novel: {prop_novel}")
        print(f"Diversity (lower bound): {diversity}")

    model.train()

    # Render generated levels and animate solutions if applicable.
    if render_dir:
        if not os.path.isdir(render_dir):
            os.makedirs(render_dir)
        trans_table = {ord(k): v for k, v in BOXOBAN_TO_GRIDDLY_CHARS.items()}
        wrapper = GymWrapperFactory()
        wrapper.build_gym_from_yaml('sokoban', os.path.join('gdy_games', 'sokoban.yaml'))
        env = gym.make('GDY-sokoban-v0')
        for i, (sample, sol) in enumerate(zip(samples, solutions)):
            lvl_render_dir = os.path.join(render_dir, f"lvl_{i}")
            if not os.path.isdir(lvl_render_dir):
                os.makedirs(lvl_render_dir)
            sample = sample.translate(trans_table)
            j = 0
            if sol != False:
                frames = []
                ep_rew = 0
                env.reset(level_string=sample)
                im_name = os.path.join(lvl_render_dir, f"{j}.png")
                im = env.render(mode='rgb_array')
                im = Image.fromarray(im)
                im.save(im_name)
                frames.append(im)
                for act_dict in sol:
                    j += 1
                    act_tpl = (act_dict['x'], act_dict['y'])
                    act_id = GRIDDLY_ACTION_MAPPING[act_tpl]
                    obs, rew, done, info = env.step(act_id)
                    ep_rew += rew
                    im_name = os.path.join(lvl_render_dir, f"{j}.png")
                    im = env.render(mode='rgb_array')
                    im = Image.fromarray(im)
                    im.save(im_name)
                    frames.append(im)
                
                # Save gif
                frames[0].save(os.path.join(render_dir, f"lvl_{i}.gif"), format='GIF',
                    append_images=frames[1:],
                    save_all=True,
                    duration=300, loop=0)
        env.close()
            
    return prop_accurate, prop_playable, prop_novel, diversity

@hydra.main(config_path="conf", config_name="config")
def main(args: Config):
    run_name = get_run_name(args)
    output_dir = f"./logs/{run_name}"

    model, _, global_step = load_train_state(output_dir)

    model_mapping = {"gpt2": "gpt2",
                     "codeparrot": "lvwerra/codeparrot",
                     "java-gpt2": "microsoft/CodeGPT-small-java-adaptedGPT2",
                     "incoder-1B": "facebook/incoder-1B",
                     "incoder-6B": "facebook/incoder-6B"}

    # Instantiate the tokenizer based on the model's
    model_name = model_mapping[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "PAD",
                                  "bos_token": "START"})

    # Instantiate the dataset
    if args.game == "sokoban":
        dataset = AnnotatedSokobanDataset(tokenizer,
                                          args.model,
                                          level_key=args.level_key,
                                          annotation_keys=args.annotation_keys,
                                          num_annotation_buckets=args.num_annotation_buckets,
                                          holdout_solution_lens=args.holdout_solution_lens,
                                          split="train",
                                          chunk_size=args.chunk_size)

    else:
        raise NotImplementedError

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    render_dir = os.path.join(output_dir, 'renders')

    evaluate(model, device, tokenizer, dataset, args, verbose=True, render_dir=None)

if __name__ == "__main__":
    main()