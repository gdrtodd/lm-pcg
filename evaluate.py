import os
import hydra
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from conf.config import Config
from datasets import SokobanLMDataset
from utils import get_run_name, load_train_state

def evaluate(model, device, tokenizer, dataset, args, verbose=False):

    # Map the model to the available device
    model.to(device)

    # Set up for evaluation
    model.eval()

    # Generate samples
    with torch.no_grad():
        samples = model.generate(
            tokenizer.encode(tokenizer.bos_token + args.gen_context, return_tensors="pt").to(device),
            max_length=args.gen_len,
            temperature=args.gen_temp,
            do_sample=True,
            top_k=args.gen_top_k,
            top_p=args.gen_top_p,
            typical_p=args.gen_typical_p,
            num_beams=args.gen_beams,
            num_return_sequences=args.num_eval_samples
        )

    # Decode samples
    samples = [dataset.decode(sample) for sample in samples]

    prop_playable = 0 # TODO: compute playability using ASTAR agent
    prop_novel = 1 - (len(dataset.level_hashes.intersection(set([dataset._hash_level(sample) for sample in samples]))) / len(samples))

    if verbose:
        print("GENERATION PARAMETERS:")
        print(f"\tContext: {args.gen_context}")
        print(f"\tLength: {args.gen_len}")
        print(f"\tTemperature: {args.gen_temp}")
        print(f"\tTop-k: {args.gen_top_k}")
        print(f"\tTop-p: {args.gen_top_p}")
        print(f"\tTypical-p: {args.gen_typical_p}")
        print(f"\tBeams: {args.gen_beams}")

        for idx, sample in enumerate(samples):
            print("_" * os.get_terminal_size().columns)
            print(sample)
            print(f"\nSample {idx + 1} of {args.num_eval_samples}")
            print("Playable: ???")
            print(f"Novel: {dataset._hash_level(sample) not in dataset.level_hashes}")

    model.train()

    return prop_playable, prop_novel

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
        data_source = args.data_source if args.data_source else "boxoban"
        dataset = SokobanLMDataset(tokenizer,
                                   data_source=data_source,
                                   chunk_size=args.chunk_size)

    else:
        raise NotImplementedError

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluate(model, device, tokenizer, dataset, args, verbose=True)

if __name__ == "__main__":
    main()