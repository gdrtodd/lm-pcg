import hydra
import os
import json
import torch
from tqdm import tqdm
import shutil
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from conf.config import Config

from transformers import get_linear_schedule_with_warmup
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import SokobanLMDataset
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
    data_loader_iter, dataset = iter(data_loader), data_loader.dataset
    for _ in range(batch_i):
        next(data_loader_iter)

    # Calculate the total number of training steps to initialize the scheduler
    num_train_steps = len(data_loader) * args.epochs
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
    
    try:
        for epoch in range(epoch, args.epochs):
            for batch_i in range(batch_i, len(data_loader)):
                global_step += 1

                batch = next(data_loader_iter)

                token_ids = batch["input_ids"].to(device)
                labels = token_ids.clone().detach()

                # By default, the ignore index is -100, and we want to ignore all of the pad tokens
                labels[labels == tokenizer.pad_token_id] = -100

                loss = model(token_ids, labels=labels)[0]

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
                    inputs = tokenizer(tokenizer.bos_token + args.gen_context, return_tensors="pt").input_ids
                    inputs = inputs.to(device)

                    outputs = model.generate(inputs, max_length=args.gen_len, num_beams=args.gen_beams,
                                             temperature=args.gen_temp, do_sample=True)[0]
                    
                    sample = tokenizer.decode(outputs, skip_special_tokens=True)
                    if not args.no_log: 
                        log_writer.add_text("eval/random_sample", f"```\n{sample}\n```", global_step)
                    print(f"\nSample:\n{sample}\n")

                if global_step%args.save_freq == 0 and not args.no_log:
                    # torch.save(model.state_dict(), os.path.join(output_dir, f"model_weights_{global_step}.pth"))
                    save_train_state(model, optimizer, global_step, output_dir)

                if global_step%args.eval_freq == 0:
                    print(f"\nGenerating samples for evaluation at step {global_step}...")
                    prop_playable, prop_novel = evaluate(model, device, tokenizer, dataset,  args)

                    print("Proportion of playable levels:", prop_playable)
                    print("Proportion of novel levels:", prop_novel)

                    if not args.no_log:
                        log_writer.add_scalar("eval/prop_playable", prop_playable, global_step)
                        log_writer.add_scalar("eval/prop_novel", prop_novel, global_step)

            # Reset the data loader iterator and save at the end of each epoch
            data_loader_iter = iter(data_loader)
            batch_i = 0

            if not args.no_log:
                save_train_state(model, optimizer, global_step, output_dir)


    except KeyboardInterrupt:
        progress_bar.close()
        exit("Stopping early due to user input")

    print("Finished training.")
    progress_bar.close()
    if not args.no_log:
        save_train_state(model, optimizer, global_step, output_dir)

@hydra.main(config_path="conf", config_name="config")
def main(args: Config):
    run_name = get_run_name(args)

    # wandb.init(project="game-generation-modeling", entity="gdrtodd", config={}, name=run_name)
    # wandb.config.update(args)


    # Map from model names to the load string transformers expects
    model_mapping = {"gpt2": "gpt2",
                     "codeparrot": "lvwerra/codeparrot",
                     "java-gpt2": "microsoft/CodeGPT-small-java-adaptedGPT2",
                     "incoder-1B": "facebook/incoder-1B",
                     "incoder-6B": "facebook/incoder-6B"}

    # Set parallelism to false to silence deadlock warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Instantiate the tokenizer based on the model's
    model_name = model_mapping[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "PAD",
                                  "bos_token": "START"})

    # Instantiate the dataset
    if args.game == "sokoban":
        data_source = args.data_source if args.data_source else "boxoban"
        dataset = SokobanLMDataset(tokenizer,
                                   args.model,
                                   data_source=data_source,
                                   chunk_size=args.chunk_size)

    else:
        raise NotImplementedError

    # Initialize the modelm data collator, data loader, and optimizer
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
        
        elif os.path.exists(output_dir):
            try:
                model, optimizer_state_dict, global_step = load_train_state(output_dir)
                optimizer.load_state_dict(optimizer_state_dict)
                print("Loaded checkpoint from step", global_step)
            except FileNotFoundError:
                print(f"No checkpoint not found in {output_dir}. Removing directory and starting from scratch.")
                shutil.rmtree(output_dir, ignore_errors=True)

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(os.path.join(output_dir, "config.json"), "w") as file:
            json.dump(dict(args), file)

    train_loop(model, tokenizer, optimizer, data_loader, output_dir, global_step, args)


if __name__ == "__main__":
    main()