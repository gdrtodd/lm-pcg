import hydra
import json
import omegaconf
import os
import torch
from tqdm import tqdm
import shutil
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from conf.config import Config

from transformers import set_seed
from transformers import get_linear_schedule_with_warmup
from transformers import DataCollatorForLanguageModeling
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from datasets import GameDataset, AnnotatedSokobanDataset, LMazeLMDataset
from evaluate import evaluate
from utils import get_run_name, save_train_state, load_train_state

def train_loop(model, tokenizer, optimizer, data_loader, output_dir, global_step, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Map the model to the available device
    model.to(device)

    # Initialize the log writer
    log_writer = SummaryWriter(output_dir, flush_secs=100)

    # Calcuate current epoch and batch based on global step (for resuming training)
    epoch, batch_i = global_step // len(data_loader), global_step % len(data_loader)

    # Convert data loader to iterator and progress it to the current batch
    data_loader_iter = iter(data_loader)
    dataset: GameDataset = data_loader.dataset
    for _ in range(batch_i):
        next(data_loader_iter)

    # Calculate the total number of training steps to initialize the scheduler
    # num_train_steps = len(data_loader) * args.epochs
    num_train_steps = args.num_train_steps

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_proportion * num_train_steps),
        num_training_steps=num_train_steps,
        last_epoch=global_step if global_step > 0 else -1)

    # Set up for training
    model.train()

    # Initialize the progress bar
    progress_bar = tqdm(total=num_train_steps, desc=f"Training {args.model} model")
    progress_bar.update(global_step)

    done_training = False
    
    try:
        # for epoch in range(epoch, args.epochs):
        while not done_training:
            epoch += 1
            for batch_i in range(batch_i, len(data_loader)):
                global_step += 1

                batch = next(data_loader_iter)

                token_ids = batch["input_ids"].to(device)
                labels = token_ids.clone().detach()

                # By default, the ignore index is -100, and we want to ignore all of the pad tokens
                labels[labels == tokenizer.pad_token_id] = -100

                loss = model(token_ids, labels=labels)[0]

                if torch.isnan(loss):
                    print(f"NaN loss detected in at global step: {global_step}, skipping")
                    continue

                # Clear some memory before the expensive gradient computation
                del token_ids
                del labels

                # Perform optimization and update the scheduler
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                if not args.no_log: 
                    log_writer.add_scalar("train/loss", loss, global_step)

                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item()})

                if global_step%args.gen_freq == 0:
                    context = dataset.gen_context()
                    inputs = tokenizer(tokenizer.bos_token + context, return_tensors="pt").input_ids
                    inputs = inputs.to(device)

                    outputs = model.generate(
                        inputs,
                        max_length=args.gen_len,
                        temperature=args.gen_temp,
                        do_sample=True,
                        top_k=args.gen_top_k,
                        top_p=args.gen_top_p,
                        typical_p=args.gen_typical_p,
                        num_beams=args.gen_beams,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id,
                    )[0]

                    sample = dataset.decode(outputs)
                    if not args.no_log: 
                        log_writer.add_text("eval/random_sample", f"```\n{sample.replace('-', ' ')}\n```", global_step)

                    solution = dataset.get_solution(sample)
                    accurate, info = dataset.is_accurate(sample, solution)

                    print(f"\nSample:\n{sample.replace('-', ' ')}\n")
                    print(f"Novel: {dataset.is_novel(sample)[0]}")    
                    print(f"Playable: {solution != False}")
                    print(f"Accurate: {accurate}")

                if global_step%args.save_freq == 0 and not args.no_log:
                    save_train_state(model, optimizer, global_step, output_dir)

                if global_step%args.eval_freq == 0:
                    print(f"\nGenerating samples for evaluation at step {global_step}...")
                    prop_accurate, prop_playable, prop_novel, diversity = evaluate(model, device, tokenizer, dataset, args, num_proc=args.num_eval_proc)

                    print("Proportion of accurate levels:", prop_accurate)
                    print("Proportion of playable levels:", prop_playable)
                    print("Proportion of novel levels:", prop_novel)
                    print("Diversity (lower bound):", diversity)

                    if not args.no_log:
                        log_writer.add_scalar("eval/prop_playable", prop_playable, global_step)
                        log_writer.add_scalar("eval/prop_novel", prop_novel, global_step)
                        log_writer.add_scalar("eval/prop_accurate", prop_accurate, global_step)
                        log_writer.add_scalar("eval/diversity", diversity, global_step)

                if global_step >= args.num_train_steps:
                    done_training = True
                    break

            # Reset the data loader iterator and save at the end of each epoch
            data_loader_iter = iter(data_loader)
            batch_i = 0

            if not args.no_log:
                save_train_state(model, optimizer, global_step, output_dir)


    except KeyboardInterrupt:
        progress_bar.close()
        exit("Stopping early due to user input")

    print(f"Finished training: {global_step} steps and {epoch} epochs.")
    progress_bar.close()
    if not args.no_log:
        save_train_state(model, optimizer, global_step, output_dir)

@hydra.main(version_base="1.2.0", config_path="conf", config_name="config")
def main(args: Config):

    # Set the seed
    set_seed(args.seed)

    run_name = get_run_name(args)

    # Map from model names to the load string transformers expects
    model_mapping = {"gpt2": "gpt2",
                     "gpt2-untrained": "gpt2-untrained",
                     "codeparrot": "lvwerra/codeparrot",
                     "java-gpt2": "microsoft/CodeGPT-small-java-adaptedGPT2",
                     "incoder-1B": "facebook/incoder-1B",
                     "incoder-6B": "facebook/incoder-6B"}

    # Set parallelism to false to silence deadlock warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
            os.makedirs(tokenizer_dir, exist_ok=True)

            if args.game == "sokoban":
                boxoban_levels_dir = os.path.join("./data", "boxoban-medium", "train")
                boxoban_level_files = [os.path.join(boxoban_levels_dir, file) for file in os.listdir(boxoban_levels_dir) if file.endswith(".txt")]
                microban_level_files = [os.path.join("./data", "microban", file) for file in os.listdir(os.path.join("./data", "microban")) if file.endswith(".txt")]

                tokenizer_train_levels = [open(file, "r").read() for file in boxoban_level_files + microban_level_files]

            else:
                raise NotImplementedError

            print("Training GPT2 tokenizer from scratch...")
            old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer = old_tokenizer.train_new_from_iterator(tokenizer_train_levels,
                                                              length=len(tokenizer_train_levels),
                                                              vocab_size=10000,
                                                              new_special_tokens=["<s>", "</s>", "<pad>"])

            tokenizer.save_pretrained(tokenizer_dir)

            tokenizer.add_special_tokens({"pad_token": "<pad>",
                                          "bos_token": "<s>",
                                          "eos_token": "</s>"})



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

    elif args.game == "l_maze":
        dataset = LMazeLMDataset(tokenizer,
                                 args.model,
                                 chunk_size=args.chunk_size)

    else:
        raise NotImplementedError

    # Initialize the modelm data collator, data loader, and optimizer
    if args.model == "gpt2-untrained":
        gpt2_config = AutoConfig.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_config(gpt2_config)
        model.resize_token_embeddings(len(tokenizer))

    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    data_loader = DataLoader(dataset, collate_fn=data_collator, batch_size=args.batch_size, shuffle=True, num_workers=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    global_step = 0

    # Create/load/overwrite the output directory if logging.
    output_dir = None
    if not args.no_log:
        output_dir = f"./logs/{run_name}"

        # Overwrite output directory, or load train state if checkpoint exists
        if args.overwrite:
            shutil.rmtree(output_dir, ignore_errors=True)
        
        # elif os.path.exists(output_dir):
        #     try:
        model, optimizer_state_dict, global_step = load_train_state(output_dir)
        optimizer.load_state_dict(optimizer_state_dict)
        print("Loaded checkpoint from step", global_step)
        #     except FileNotFoundError:
        #         print(f"No checkpoint not found in {output_dir}. Removing directory and starting from scratch.")
        #         shutil.rmtree(output_dir, ignore_errors=True)

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(os.path.join(output_dir, "config.json"), "w") as file:
            args_dict = {key: value if not isinstance(value, omegaconf.ListConfig) else list(value) for key, value in dict(args).items()}
            json.dump(args_dict, file)

    train_loop(model, tokenizer, optimizer, data_loader, output_dir, global_step, args)


if __name__ == "__main__":
    main()