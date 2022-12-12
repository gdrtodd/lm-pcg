import torch
from tqdm import tqdm

def evaluate(model, device, tokenizer, dataset, args):

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

    model.train()

    return prop_playable, prop_novel
