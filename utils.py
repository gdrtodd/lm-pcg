import hashlib
from itertools import groupby
import os
from typing import List
import numpy as np
import shutil

import imageio
from PIL import Image
import torch
from transformers import AutoModelForCausalLM
from conf.config import Config

from sokoban_solvers import EnhancedAStarAgent, State

class CheckpointNotFoundError(FileNotFoundError):
    pass

def get_run_name(args: Config):
    run_name = os.path.join(
        args.game,
        f"source:{args.source}",
        f"model:{args.model}",
        f"level_key:{args.level_key}",
        f"annotation_keys:{args.annotation_keys}",
        f"num_annotation_buckets:{args.num_annotation_buckets}",
        f"holdouts:{args.holdout_solution_lens}",
        f"chunk_size-{args.chunk_size}_lr-{args.learning_rate}",
        f"sample_prop:{args.sample_prop}",
        f"seed-{args.seed}",
    )
    return run_name

def filter_configs(cfgs: List[Config]):
    new_cfgs = []
    for cfg in cfgs:
        if is_valid_config(cfg):
            new_cfgs.append(cfg)
    return new_cfgs

def is_valid_config(cfg: Config) -> bool:
    """ When manually sweeping over hyperparams, identify combinations."""
    if cfg.holdout_solution_lens is not None and cfg.annotation_keys is None:
        # Cannot hold out prompts when model is not trained to match prompts (?)
        return False
    return True


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set output dir to most recent checkpoint
    prior_checkpoint_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith("checkpoint-")]
    prior_checkpoint_paths = sorted(prior_checkpoint_paths, key=lambda x: int(x.split("-")[-1]))
    
    if len(prior_checkpoint_paths) == 0:
        raise CheckpointNotFoundError(f"No checkpoints found at {output_dir}. Exiting.")

    output_dir = prior_checkpoint_paths[-1]

    # Load
    model = AutoModelForCausalLM.from_pretrained(output_dir)
    optimizer_state_dict = torch.load(os.path.join(output_dir, "optimizer.pt"), map_location=device)
    with open(os.path.join(output_dir, "global_step.txt"), "r") as f:
        global_step = int(f.read())

    return model, optimizer_state_dict, global_step

BOXOBAN_MAPPING = {
    '-': 'empty',
    '#': 'wall',
    '$': 'box',
    '.': 'goal',
    '*': 'box_in_place',
    '@': 'player',
    '+': 'player_in_place'
}

BOXOBAN_INVERSE_MAPPING = {v: k for k, v in BOXOBAN_MAPPING.items()}

GRIDDLY_INVERSE_MAPPING = {
    'empty': '.',
    'wall': 'w',
    'box': 'b',
    'goal': 'h',
    'player': 'A',
    'box_in_place': 'f',
    'player_in_place': 'A'
}

BOXOBAN_TO_GRIDDLY_CHARS = {k: GRIDDLY_INVERSE_MAPPING[v] for k, v in BOXOBAN_MAPPING.items()}

GRIDDLY_ACTION_MAPPING = {(-1, 0): 1, (0, -1): 2, (1, 0): 3, (0, 1): 4}

def encode_boxoban_text(level):
    # Remove the first line of each level, which just contains the level number
    level = level[level.find("\n")+1:].strip()

    lines = []
    for line in level.split("\n"):
        # Group consecutive characters together and map them to their identities
        line_text = ", ".join([f"{len(list(iter))} {BOXOBAN_MAPPING[char]}" for char, iter in groupby(line)])
        lines.append(line_text)

    level_text = "\n".join(lines)

    return level_text

def decode_boxoban_text(text):
    # TODO: this code doesn't handle any error cases, which are sure to come up during generation
    level = ""

    for line in text.split("\n"):
        try:
            for section in line.split(", "):
                count, char = section.split(" ")
                level += BOXOBAN_INVERSE_MAPPING[char] * int(count)
        
            level += "\n"

        except:
            level += "Invalid line\n"

    return level

def generate_l_mazes(width, height):
    '''
    Generates the set of all "L Mazes" of a given width and height. We construct an L Maze by choosing a start
    and end point at least one square away from the edge and then connecting them with a path that has at most
    one turn. For example:

    ##########
    #      ###
    ###### ###
    ##########
    ##########
    '''

    def to_string(grid):
        return "\n".join(["".join(["#" if pos == 1 else "-" for pos in row]) for row in grid])

    def l_path(start, end):
        path = []

        cur_pos = start

        # Always gives the path that changes y before x
        while cur_pos[1] != end[1]:
            cur_pos = (cur_pos[0], cur_pos[1] + (1 if cur_pos[1] < end[1] else -1))
            path.append(cur_pos)

        while cur_pos[0] != end[0]:
            cur_pos = (cur_pos[0] + (1 if cur_pos[0] < end[0] else -1), cur_pos[1])
            path.append(cur_pos)

        return path


    l_mazes = []
    path_lens = []

    interior_positions = [(y, x) for x in range(1, width-1) for y in range(1, height-1)]
    used_starts = set()

    for start in interior_positions:
        for end in interior_positions:
            if start == end:
                continue
            if end in used_starts:
                continue
            used_starts.add(start)

            grid = np.ones((height, width), dtype=np.int8)
            path = l_path(start, end)

            grid[start] = 0
            for pos in path:
                grid[pos] = 0

            annotation = f"Width: {width}\nHeight: {height}\nPath length: {len(path)}\n"
            path_lens.append(len(path))
            l_mazes.append(annotation + to_string(grid))

    return l_mazes, path_lens

def _process_level(level):
    '''
    Given a boxoban level, return a dictionary containing all of the relevant information, 
    (see comment in __init__) for details
    '''

    solver = EnhancedAStarAgent()

    level_hash = _hash_level(level)

    level_text = encode_boxoban_text(level)
    level_state = State().stringInitialize(level.split("\n"))

    # Remove the first line of each level, which just contains the level number
    level = level[level.find("\n")+1:]

    # Pad the level with walls to make it rectangular
    max_width = max([len(row) for row in level.split("\n")])
    lines = []

    for line in level.split("\n"):
        if line == "": continue
        num_leading_spaces = len(line) - len(line.lstrip())
        formatted = ("#" * num_leading_spaces) + line.strip() + ("#" * (max_width - len(line)))
        lines.append(formatted)

    # Fill in gaps in to top and bottom rows
    lines[0] = lines[0].replace(" ", "#")
    lines[-1] = lines[-1].replace(" ", "#")

    # Combine the rows, strip, and replace spaces with dashes
    level = "\n".join(lines).strip().replace(" ", "-")

    width = len(level.split("\n")[0])
    height = len(level.split("\n"))
    num_targets = level.count("$") + level.count("*")
    prop_empty = level.count("-") / (width * height)

    solution, node, iterations = solver.getSolution(level_state, maxIterations=1_000_000, maxTime=-1)
    if node.checkWin():
        solution_len = len(solution)
        print(f"Solved after {iterations} iterations.")
    else:
        solution_len = -1
        solution = None
        print(f"Failed after {iterations} iterations.")

    return level, level_text, level_hash, width, height, num_targets, prop_empty, solution_len, solution

def _hash_level(level):
    return int(hashlib.md5(level.encode("utf-8")).hexdigest(), 16)



def save_gif(env, lvl, sol, lvl_render_dir):
    if not os.path.isdir(lvl_render_dir):
        os.makedirs(lvl_render_dir)
    j = 0
    if sol != False:
        frames = []
        ep_rew = 0
        env.reset(level_string=lvl)
        im_name = os.path.join(lvl_render_dir, f"{j}.png")
        im = env.render(mode='rgb_array')
        im = Image.fromarray(im)
        im.save(im_name)
        frames.append(im)
        for act_id in sol:
            j += 1
            obs, rew, done, info = env.step(int(act_id))
            ep_rew += rew
            im_name = os.path.join(lvl_render_dir, f"{j}.png")
            im = env.render(mode='rgb_array')
            im = Image.fromarray(im)
            im.save(im_name)
            frames.append(im)
        
        # Parent of the level directory and name of the level directory
        render_dir, lvl_dir = os.path.split(lvl_render_dir)
        # Save gif with fps of 3
        imageio.mimsave(os.path.join(render_dir, f"{lvl_dir}.gif"), frames, fps=10)


#     level = """
# ##########
# ### ######
# ###@ #####
# ### ######
# ###$ #####
# ### .#####
# ###  #####
# # .$ $ ###
# #. $ .  ##
# ##########
#     """

#     from sokoban_solvers import EnhancedAStarAgent, State

#     start_state = State().stringInitialize(level.split("\n"))

#     solver = EnhancedAStarAgent()
#     sol, node, iters = solver.getSolution(start_state)
#     print(sol)
#     print(node.checkWin())

#     output = """10 wall
# 1 wall, 2 empty, 7 wall
# 1 wall, 1 empty, 1 box, 7 wall
# 1 wall, 1 goal, 1 empty, 7 wall
# 1 wall, 1 empty, 1 goal, 7 wall
# 1 wall, 1 empty, 1 box, 7 wall
# 1 wall, 1 empty, 1 goal, 7 wall
# 1 wall, 1 box, 1 empty, 1 box, 1 empty, 5 wall
# 1 wall, 1 player, 1 empty, 1 goal, 2 empty, 4 wall
# 10 wall"""

    # print(decode_boxoban_text(output))

    # print(level)
    # print(encode_boxoban_text(level))
    # print(decode_boxoban_text(encode_boxoban_text(level)))