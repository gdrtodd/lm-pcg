from pdb import set_trace as TT
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

def save_train_state(model, optimizer, global_step, output_dir):
    # Get paths of any previous checkpoints
    prior_checkpoint_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith("checkpoint-")]

    # Save current checkpoint
    output_dir = os.path.join(output_dir, "checkpoint-{}".format(global_step))
    model.save_pretrained(output_dir)
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    with open(os.path.join(output_dir, "global_step.txt"), "w") as f:
        f.write(str(global_step))

    # Delete prior checkpoints
    [shutil.rmtree(path) for path in prior_checkpoint_paths]

def load_train_state(output_dir):
    print("Attempting to load checkpoint from {}...".format(output_dir))

    # Set output dir to most recent checkpoint
    prior_checkpoint_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith("checkpoint-")]
    prior_checkpoint_paths = sorted(prior_checkpoint_paths, key=lambda x: int(x.split("-")[-1]))
    output_dir = prior_checkpoint_paths[-1]

    # Load
    model = AutoModelForCausalLM.from_pretrained(output_dir)
    optimizer_state_dict = torch.load(os.path.join(output_dir, "optimizer.pt"))
    with open(os.path.join(output_dir, "global_step.txt"), "r") as f:
        global_step = int(f.read())

    return model, optimizer_state_dict, global_step

def train_loop(model, tokenizer, optimizer, data_loader, output_dir, global_step, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Map the model to the available device
    model.to(device)

    # Initialize the log writer
    log_writer = SummaryWriter(output_dir, flush_secs=100)

    # Calculate the total number of training steps to initialize the scheduler
    num_train_steps = (len(data_loader) // args.batch_size) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_proportion * num_train_steps),
        num_training_steps=num_train_steps,
        last_epoch=global_step if global_step > 0 else -1,)

    # Set up for training
    model.train()
    progress_bar = tqdm(total=num_train_steps, desc=f"Training {args.model} model")
    progress_bar.update(global_step)
    epoch, batch_i = global_step // len(data_loader), global_step % len(data_loader)

    try:
        for epoch in range(epoch, args.epochs):
            for batch_i in range(batch_i, len(data_loader)):
                global_step += 1

                # Note that when re-loading mid-epoch, we may repeat certain batches.
                batch = next(iter(data_loader))

                token_ids = batch["input_ids"].to(device)
                labels = token_ids.clone().detach()

                # By default, the ignore index is -100, and we want to ignore all of the pad tokens
                labels[labels == tokenizer.pad_token_id] = -100

                loss = model(token_ids, labels=labels)[0]
                perplexity = torch.exp(loss).item()

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
                    log_writer.add_scalar("train/perplexity", perplexity, global_step)

                    # wandb.log({"train_loss": loss})

                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item()})

                if global_step%args.gen_freq == 0:
                    inputs = tokenizer(tokenizer.bos_token + args.gen_context, return_tensors="pt").input_ids
                    inputs = inputs.to(device)

                    outputs = model.generate(inputs, max_length=args.gen_len, num_beams=args.gen_beams,
                                             temperature=args.gen_temp, do_sample=True)[0]
                    
                    sample = tokenizer.decode(outputs, skip_special_tokens=False)
                    if not args.no_log: 
                        log_writer.add_text("eval/random_sample", sample, global_step)
                    print(f"\nSample:\n{sample}\n")

                if global_step%args.save_freq == 0 and not args.no_log:
                    # torch.save(model.state_dict(), os.path.join(output_dir, f"model_weights_{global_step}.pth"))
                    save_train_state(model, optimizer, global_step, output_dir)

                if global_step%args.eval_freq == 0:
                    print(f"\nGenerating samples for evaluation at step {global_step}...")
                    prop_playable, prop_novel = evaluate(model, device, tokenizer, data_loader.dataset,  args)

                    print("Proportion of playable levels:", prop_playable)
                    print("Proportion of novel levels:", prop_novel)

                    if not args.no_log:
                        log_writer.add_scalar("eval/prop_playable", prop_playable, global_step)
                        log_writer.add_scalar("eval/prop_novel", prop_novel, global_step)


    except KeyboardInterrupt:
        print("Stopping early due to user input!")
        progress_bar.close()
        exit()

    print("Finished training.")
    progress_bar.close()
    if not args.no_log:
        # torch.save(model.state_dict(), os.path.join(output_dir, f"model_weights_{global_step}.pth"))
        save_train_state(model, optimizer, global_step, output_dir)


def get_run_name(args):
    # datetime_str = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    run_name = os.path.join(
        args.game,
        args.model,
        args.data_source if args.data_source else "",
        f"chunk_size-{args.chunk_size}_lr-{args.learning_rate}",
        args.exp_name,
        f"seed-{args.seed}",
    )
    return run_name


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