import hashlib
import math
import os
import multiprocessing as mp
import random
from typing import Iterable
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from utils import BOXOBAN_MAPPING, encode_boxoban_text, decode_boxoban_text
from sokoban_solvers import EnhancedAStarAgent, State

VALID_DATA_SOURCES = ["boxoban", "boxoban-chars", "boxoban-text", "microban"]

class SokobanLMDataset(Dataset):
    '''
    Dataset for Sokobaln levels for use with a language model

    Args:
        -tokenizer: Huggingface tokenizer for a specific language model
        -model_name: Name of the language model
        -data_source: Which type dataset to use. Current options are "boxoban" (raw levels), "boxoban-chars"
            (tokenizing each character separately), and "boxoban-text" (representing levels in natural language)
        -annotation: What style of annotation to use. Current options are None (no annotation), "partial" 
            (width, length, # of boxes, % whitespace), and "full" (partial annotation + solution length)
        -split Which split of the dataset to use (if available)
        -chunk_size: Number of tokens per item in the datatset
        -cache_dir: Path to dataset cache files
    '''

class GameDataset(Dataset):
    def __init__(self):
        self.all_levels = []
        self.level_hashes = set()

    def gen_context(self):
        return ''

    def is_accurate(self, level, sol):
        '''
        Checks fidelity to prompt, if any, and returns relevant stats.
        '''
        stats = {
            'sol_len': None if not sol else len(sol),
        }
        return True, stats

    def _hash_level(self, level):
        return int(hashlib.md5(level.encode("utf-8")).hexdigest(), 16)

    def decode(self, token_ids):
        '''
        Decode an array of token IDs back into text.
        '''
        # Get first position in sample corresponding to an EOS token, and truncate the sample accordingly.
        # (We're basically being forgiving to models that don't pad the end of each sample with unbroken EOS's)
        eos_pos = torch.where(token_ids == self.tokenizer.encode(self.tokenizer.eos_token)[0])[0][0].item()
        token_ids = token_ids[:eos_pos]

        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)

        return text

    def is_novel(self, level):
        '''
        Returns whether the given level is novel by checking if its hashed value is in the set of
        level hashes.
        '''
        level_hash = self._hash_level(level)
        return level_hash not in self.level_hashes

    def get_solution(self, level):
        raise NotImplementedError


class SokobanLMDataset(GameDataset):
    def __init__(self,
                 tokenizer: AutoTokenizer,
                 model_name: str,
                 data_source="boxoban",
                 annotation_level=None,
                 split="train",
                 chunk_size=128,
                 cache_dir="./caches"):

        assert data_source in VALID_DATA_SOURCES, f"Data source must be one of {VALID_DATA_SOURCES}"
        assert annotation_level in [None, "partial", "full"], "Annotation must be one of [None, partial, full]"
        super().__init__()

        self.data_source = data_source
        self.split = split
        self.chunk_size = chunk_size

        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id

        self.solver = EnhancedAStarAgent()

        # Pre-process levels.
        if data_source in ["boxoban", "boxoban-chars"]:
            data_dir = os.path.join("./data", "boxoban-medium", split)

            level_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")]
            for file in level_files:
                with open(file, "r") as f:

                    # Split into individual levels
                    raw_levels = f.read().split("; ")

                    # Remove the first line of each level, which just contains the level number, and replace spaces with dashes
                    raw_levels = [level[level.find("\n")+1:].strip().replace(" ", "-") for level in raw_levels if level != ""]

                    self.all_levels += raw_levels

        elif data_source == "boxoban-text":
            data_dir = os.path.join("./data", "boxoban-medium", split)

            level_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")]
            for file in level_files:
                with open(file, "r") as f:

                    # Split into individual levels
                    raw_levels = f.read().split("; ")

                    self.all_levels += [encode_boxoban_text(level) for level in raw_levels]

        elif data_source == "microban":
            data_dir = "./data/microban"
            level_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")]
            for file in level_files:
                with open(file, "r") as f:

                    # Split into individual levels
                    raw_levels = f.read().split("; ")

                    # Remove the first line of each level, which just contains the level number
                    raw_levels = [level[level.find("\n")+1:] for level in raw_levels if level != ""]

                    for raw in raw_levels:
                        max_width = max([len(row) for row in raw.split("\n")])

                        # Pad each row to the same length and replace leading spaces
                        lines = []
                        for row in raw.split("\n"):
                            if row == "": continue
                            num_leading_spaces = len(row) - len(row.lstrip())
                            row = ("#" * num_leading_spaces) + row.strip() + ("#" * (max_width - len(row)))
                            lines.append(row)

                        # Fill in gaps in to top and bottom rows
                        lines[0] = lines[0].replace(" ", "#")
                        lines[-1] = lines[-1].replace(" ", "#")

                        processed_level = "\n".join(lines).replace(" ", "-")
                        self.all_levels.append(processed_level)

        else:
            raise NotImplementedError

        # Tokenize processed levels (or load tokens from disk if available).
        token_ids_path = os.path.join(cache_dir, f"{model_name}_{data_source}_{split}_chunksize_{chunk_size}_all_token_ids.npy")
        level_hashes_path = os.path.join(cache_dir, f"{model_name}_{data_source}_{split}_level_hashes.npy")
        annotations_path = os.path.join(cache_dir, f"{model_name}_{data_source}_{split}_{annotation_level}_annotations.npy")

        if os.path.isfile(token_ids_path) and os.path.isfile(level_hashes_path) and (os.path.isfile(annotations_path) or annotation_level is None):
            print(f"Loading tokens from cache at {token_ids_path}...")
            self.all_token_ids = np.load(token_ids_path)
            self.level_hashes = np.load(level_hashes_path, allow_pickle=True).flatten()[0] # weird flattening seems to be necessary to recover set?

        else:
            # Optionally ensure each tile-character is tokenized individually
            if data_source == "boxoban-chars":
                tile_chars = list(BOXOBAN_MAPPING.keys()) + ["\n", tokenizer.bos_token, tokenizer.eos_token]
                tile_encodings = {c: self.tokenizer.encode(c)[0] for c in tile_chars}

            all_token_ids = []

            # if annotation_level is not None:
            #     with mp.Pool(16) as pool:
            #         all_annotations = list(tqdm(pool.imap(self._annotate_level, all_levels[:100]), total=len(all_levels[:100]),
            #                                     desc="Annotating levels"))
            #     from tqdm.contrib.concurrent import process_map
            #     all_annotations = process_map(self._annotate_level, all_levels, chunksize=1000, max_workers=1,
            #                                   desc="Annotating levels")

            # for ann in all_annotations:
            #     print("=" * 80)
            #     print(ann)

            # exit()

            for level in tqdm(self.all_levels, desc="Tokenizing levels"):
                # Skip empty level
                if level == '':
                    continue

                # We use the MD5 hash of the level as a unique identifier which is stable across runs
                level_hash = self._hash_level(level)
                if level_hash in self.level_hashes:
                    continue

                self.level_hashes.add(level_hash)

                if data_source == "boxoban-chars":
                    # Manual tokenization to ensure each tile token is separate
                    token_ids = []
                    level_rows = level.split('\n')
                    for row in level_rows:
                        token_ids += [tile_encodings[c] for c in row] + [tile_encodings['\n']]
                    token_ids = [tile_encodings[tokenizer.bos_token]] + token_ids + [tile_encodings[tokenizer.eos_token]]
                    # Pad
                    token_ids += [self.pad_token_id for _ in range(self.chunk_size - len(token_ids))]

                else:
                    # Standard tokenization
                    if annotation_level is not None:
                        annotation = self._annotate_level(level, include_sol_len=(annotation_level == "full")) + "\n\n"
                        print(annotation)
                    else:
                        annotation = ""

                    level = f"{tokenizer.bos_token}{annotation}{level}{tokenizer.eos_token}"
                    token_ids = self.tokenizer.encode(level, padding="max_length", max_length=self.chunk_size, truncation=True)

                all_token_ids += token_ids

            # Save token ids and hashes to disk
            np.save(token_ids_path, all_token_ids)
            np.save(level_hashes_path, self.level_hashes)

            self.all_token_ids = np.array(all_token_ids, dtype=np.int32)

    def _hash_level(self, level):
        return int(hashlib.md5(level.encode("utf-8")).hexdigest(), 16)

    def _annotate_level(self, level, include_sol_len=False):
        '''
        Returns a linguistic annotation of the level containing:
        -width
        -height
        -number of targets / boxes
        -proportion of empty space
        -(optional) solution length
        '''

        width = len(level.split("\n")[0])
        height = len(level.split("\n"))
        num_targets = level.count("$") # oddly, this seems to be 4 for every single level in the dataset!
        prop_empty = level.count("-") / (width * height)

        annotation = f"Width: {width}\nHeight: {height}\nNumber of targets: {num_targets}\nProportion empty: {prop_empty}"

        if include_sol_len:
            level_state = State().stringInitialize(level.split("\n"))
            solution, node, _ = self.solver.getSolution(level_state, maxIterations=50000)
            if node.checkWin():
                annotation += f"\nSolution length: {len(solution)}"
            else:
                annotation += f"\nSolution length: [Unknown]"

        return annotation


    def decode(self, token_ids):
        '''
        Decode an array of token IDs back into text. Depending on the data source, this may also apply
        some post-processing to the text.
        '''
        text = super().decode(token_ids)

        if self.data_source == "boxoban-text":
            text = decode_boxoban_text(text)

        text = text.replace("-", " ")

        return text.strip()

    def get_solution(self, level, verbose=False):
        '''
        Determines whether the given level is playable by checking a variety of conditions:
          1. the level is rectangular (i.e. every line is the same length)
          2. the level contains only the following characters: "\n", "#", " ", "-", "@", "$", "."
          3. the level contains exactly one player
          4. the level contains the same number of boxes and goals (and at least one of each)
          5. the level can be solved by an ASTAR agent
        If the level is playable, return the solution (return False otherwise).
        '''

        # Check if the level is rectangular
        line_lengths = [len(line) for line in level.split("\n")]
        if len(set(line_lengths)) != 1:
            if verbose: print("--Level is not rectangular--")
            return False

        # Check if the level contains only the allowed characters
        allowed_chars = set("\n# -@$.")
        if not set(level).issubset(allowed_chars):
            if verbose: print("--Level contains invalid characters--")
            return False

        # Check if the level contains exactly one player
        if level.count("@") != 1:
            if verbose: print("--Level does not contain exactly one player--")
            return False

        # Check if the level contains the same number of boxes and goals
        if level.count("$") != level.count(".") or level.count("$") == 0:
            if verbose: print("--Level contains different numbers of boxes and goals--")
            return False

        # Check if the level can be solved by an ASTAR agent
        level_state = State().stringInitialize(level.split("\n"))
        solution, node, iters = self.solver.getSolution(level_state, maxIterations=50000)
        if not node.checkWin():
            if verbose: print("--Level cannot be solved (... in 50k steps)--")
            return False
        elif verbose:
            print(f"++Level can be solved in {len(solution)} moves++")

        return solution

    def __getitem__(self, idx):
        start, end = self.chunk_size * idx, self.chunk_size * (idx+1)
        return torch.tensor(self.all_token_ids[start:end], dtype=torch.long)

    def __len__(self):
        return len(self.all_token_ids) // self.chunk_size


class LMazeLMDataset(GameDataset):

    w_range, h_range = (4, 12), (4, 12)
    holdout_path_lens = {7, 15}
    n_descriptor_lines = 3  # width, height, path-length
    allowed_chars = "\n#-"

    def __init__(self,
                 tokenizer: AutoTokenizer,
                 model_name: str,
                 split="train",
                 chunk_size=128,
                 cache_dir="./caches"):

        super().__init__()

        self.split = split
        self.chunk_size = chunk_size

        self.tokenizer = tokenizer
        # self.pad_token_id = self.tokenizer.pad_token_id

        data_dir = os.path.join("./data", "l-mazes", split)

        level_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")]
        for file in level_files:
            with open(file, "r") as f:

                # Split into individual levels
                self.all_levels += f.read().split("\n\n")

        # Tokenize processed levels (or load tokens from disk if available).
        token_ids_path = os.path.join(cache_dir, f"{model_name}_l_mazes_{split}_all_token_ids.npy")
        level_hashes_path = os.path.join(cache_dir, f"{model_name}_l_mazes_{split}_level_hashes.npy")

        if os.path.isfile(token_ids_path) and os.path.isfile(level_hashes_path):
            print(f"Loading tokens from cache at {token_ids_path}...")
            self.all_token_ids = np.load(token_ids_path)
            self.level_hashes = np.load(level_hashes_path, allow_pickle=True).flatten()[0] # weird flattening seems to be necessary to recover set?
        else:

            all_token_ids = []

            for level in tqdm(self.all_levels, desc="Tokenizing L mazes"):
                # Skip empty level
                if level == '':
                    continue

                # We use the MD5 hash of the level as a unique identifier which is stable across runs
                level_hash = self._hash_level(level)
                # We can be sure of this because of the way we generate l-mazes
                assert level_hash not in self.level_hashes
                self.level_hashes.add(level_hash)

                # Tokenize level. TODO: might want to add different encoding schemes here (like with Sokoban above)
                level = f"{tokenizer.bos_token}{level}{tokenizer.eos_token}"
                token_ids = self.tokenizer.encode(level, padding="max_length", max_length=self.chunk_size, truncation=True)

                all_token_ids += token_ids

            # Save token ids and hashes to disk
            np.save(token_ids_path, all_token_ids)
            np.save(level_hashes_path, self.level_hashes)

            self.all_token_ids = np.array(all_token_ids, dtype=np.int32)

    def gen_context(self):
        pl = random.choice(list(self.holdout_path_lens))
        # Uniformly sample a width and height pair from those in which pl is possible
        # It must be that w + h - 6 (subtracting 4 border tiles, 1 corner, and 1 first path tile) can fit the l.
        # TODO: Sample according to actual distribution? E.g. only 4 relevant levels when w + h - 6 = pl
        w, h = random.choice([(w, h) for w in range(*self.w_range) for h in range(*self.h_range) if w + h - 6 >= pl])
        context = f"Width: {w}\nHeight: {h}\nPath length: {pl}\n"
        return context

    def is_accurate(self, level, sol):
        # NOTE: We assume the level is playable.

        accurate = True
        prompt = level.split("\n")[:self.n_descriptor_lines]
        # Get the supposed width, height, and path length
        width, height, path_len = [int(line.split(":")[1]) for line in prompt]

        # Check if width and height are correct
        level_map = "\n".join(level.split("\n")[self.n_descriptor_lines:])
        lines = level_map.split("\n")
        line_lengths = [len(line) for line in lines]

        width_i = len(lines[0])
        height_i = len(lines)
        path_len_i = len(sol) - 1 if isinstance(sol, Iterable) else sol
        stats = {
            "width": width_i,
            "height": height_i,
            "path_len": path_len_i,
        }
        if width_i != width or height_i != height or path_len_i != path_len:
            accurate = False

        return accurate, stats        

    def get_solution(self, level, verbose=False):
        '''
        Determines whether the given level is playable by checking a variety of conditions:
          1. the level is rectangular (i.e. every line is the same length)
          2. the level only contains valid characters
          3. all empty tiles are connected
          4. no empty tile is on the border of the maze
        '''
        # Check if the level is rectangular
        level_map = "\n".join(level.split("\n")[self.n_descriptor_lines:])
        line_lengths = [len(line) for line in level_map.split("\n")]
        if len(set(line_lengths)) != 1:
            return False

        # Check if the level contains only the allowed characters
        if not set(level_map).issubset(self.allowed_chars):
            return False

        # Check if the all empty tiles (i.e. "-") are connected
        # Turn into binary array
        level = np.array([[1 if c == "#" else 0 for c in line] for line in level_map.split("\n")])
        # Find all empty tiles
        empty_tiles = np.argwhere(level == 0)

        # Check if any tiles are on the border
        if np.any(empty_tiles[:, 0] == 0) or np.any(empty_tiles[:, 0] == level.shape[0]-1) or \
                np.any(empty_tiles[:, 1] == 0) or np.any(empty_tiles[:, 1] == level.shape[1]-1):
            return False
        
        l_mask = np.zeros_like(level)

        # Defaults to playable
        if len(empty_tiles) == 0:
            return []

        x_0, y_0 = empty_tiles[0]
        for x, y in empty_tiles:
            # Check if empty tile is on the border of the maze
            if x == 0 or x == level.shape[0]-1 or y == 0 or y == level.shape[1]-1:
                return False

            # Check if empty tile is connected to any other empty tile
            if not np.any(level[x-1:x+2, y-1:y+2] == 0):
                return False

            # Draw a border-to-border line on the l-mask for each step taken.
            if x != x_0:
                l_mask[:, y] = 1
            if y != y_0:
                l_mask[x, :] = 1
            x_0, y_0 = x, y

        # If the l-mask does not contain all empty tiles, then the maze is not an l        
        if np.any(l_mask + level == 0):
            return False

        return [(x, y) for x, y in empty_tiles]

    def __getitem__(self, idx):
        start, end = self.chunk_size * idx, self.chunk_size * (idx+1)
        return torch.tensor(self.all_token_ids[start:end], dtype=torch.long)

    def __len__(self):
        return len(self.all_token_ids) // self.chunk_size