import os
import re
import sys
import torch
import pandas
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

from utils import encode_boxoban_text

class SokobanLMDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 data_source="boxoban",
                 split="train",
                 chunk_size=128):

        self.split = split
        self.chunk_size = chunk_size

        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id

        all_levels = []

        if data_source == "boxoban":
            data_dir = os.path.join("./data", "boxoban-medium", split)

            level_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")]
            for file in level_files:
                with open(file, "r") as f:

                    # Split into individual levels
                    raw_levels = f.read().split("; ")

                    # Remove the first line of each level, which just contains the level number, and replace spaces with dashes
                    raw_levels = [level[level.find("\n")+1:].strip().replace(" ", "-") for level in raw_levels]

                    all_levels += raw_levels

        elif data_source == "boxoban-text":
            data_dir = os.path.join("./data", "boxoban-medium", split)

            level_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")]
            for file in level_files:
                with open(file, "r") as f:

                    # Split into individual levels
                    raw_levels = f.read().split("; ")

                    all_levels += [encode_boxoban_text(level) for level in raw_levels]

        else:
            raise NotImplementedError

        # Load token ids if available
        token_ids_path = os.path.join(data_dir, f"all_token_ids.npy")
        if os.path.isfile(token_ids_path):
            self.all_token_ids = np.load(token_ids_path)

        else:
            all_token_ids = []

            for level in tqdm(all_levels, desc="Tokenizing levels"):
                token_ids = self.tokenizer.encode(level)
                if len(token_ids) < self.chunk_size:
                    token_ids += [self.pad_token_id] * (self.chunk_size - len(token_ids))

                all_token_ids += token_ids

            # Save token ids to disk
            np.save(token_ids_path, all_token_ids)

            self.all_token_ids = np.array(all_token_ids, dtype=np.int32)

    def decode_ids(self, token_ids):
        '''
        Convert the list of provided GPT2 token ids to a string and return it
        '''
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)

        return text

    def __getitem__(self, idx):
        start, end = self.chunk_size * idx, self.chunk_size * (idx+1)
        return torch.tensor(self.all_token_ids[start:end], dtype=torch.long)

    def __len__(self):
        return len(self.all_token_ids) // self.chunk_size

if __name__ == "__main__":
    d = SokobanLMDataset(data_source="boxoban-text")
    # print(d[10])
    # print(d.decode_ids(d[10]))