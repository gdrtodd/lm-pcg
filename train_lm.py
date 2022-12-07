from pdb import set_trace as TT
import hydra
import os
import json
import torch
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from conf.config import Config
from datasets import SokobanLMDataset

from transformers import get_linear_schedule_with_warmup
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForCausalLM

def train_loop(model, tokenizer, optimizer, data_loader, output_dir, args):
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
        num_training_steps=num_train_steps)

    # Set up for training
    model.train()
    global_step = 0
    progress_bar = tqdm(total=num_train_steps, desc=f"Training {args.model} model")

    try:
        for epoch in range(args.epochs):
            for batch in data_loader:
                global_step += 1

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
                    inputs = tokenizer(args.gen_context, return_tensors="pt").input_ids
                    inputs = inputs.to(device)

                    outputs = model.generate(inputs, max_length=args.gen_len, num_beams=args.gen_beams,
                                             temperature=args.gen_temp, pad_token_id=tokenizer.eos_token_id)[0]
                    
                    sample = tokenizer.decode(outputs, skip_special_tokens=True)
                    if not args.no_log: 
                        log_writer.add_text("eval/random_sample", sample, global_step)
                    print("\nSample:\n", sample, "\n")

                if global_step%args.save_freq == 0 and not args.no_log:
                    torch.save(model.state_dict(), os.path.join(output_dir, f"model_weights_{global_step}.pth"))


    except KeyboardInterrupt:
        print("Stopping early due to user input!")
        progress_bar.close()
        exit()

    print("Finished training.")
    progress_bar.close()
    if not args.no_log:
        torch.save(model.state_dict(), os.path.join(output_dir, f"model_weights_{global_step}.pth"))

@hydra.main(config_path="conf", config_name="config")
def main(args: Config):

    datetime_str = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    run_name = f"{datetime_str}-{args.model}-{args.game}"

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
    tokenizer.add_special_tokens({"pad_token": "PAD"})

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


    # Create the output directory
    output_dir_name = None
    if not args.no_log:
        output_dir_name = f"./logs/{run_name}"
        if not os.path.exists(output_dir_name):
            os.makedirs(output_dir_name)

        with open(os.path.join(output_dir_name, "config.json"), "w") as file:
            json.dump(dict(args), file)

    train_loop(model, tokenizer, optimizer, data_loader, output_dir_name, args)


if __name__ == "__main__":
    main()