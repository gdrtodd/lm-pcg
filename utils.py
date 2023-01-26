from itertools import groupby
import os
import numpy as np
import shutil

import torch
from transformers import AutoModelForCausalLM

def get_run_name(args):
    run_name = os.path.join(
        args.game,
        f"model:{args.model}",
        f"level_key:{args.level_key}",
        f"annotation_keys:{args.annotation_keys}",
        f"holdouts:{args.holdout_solution_lens}",
        f"chunk_size-{args.chunk_size}_lr-{args.learning_rate}",
        f"seed-{args.seed}",
    )
    return run_name

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
        raise FileNotFoundError(f"No checkpoints found at {output_dir}. Exiting.")

    output_dir = prior_checkpoint_paths[-1]

    # Load
    model = AutoModelForCausalLM.from_pretrained(output_dir)
    optimizer_state_dict = torch.load(os.path.join(output_dir, "optimizer.pt"), map_location=device)
    with open(os.path.join(output_dir, "global_step.txt"), "r") as f:
        global_step = int(f.read())

    return model, optimizer_state_dict, global_step

BOXOBAN_MAPPING = {
    ' ': 'empty',
    '#': 'wall',
    '$': 'box',
    '.': 'goal',
    '@': 'player'
}

BOXOBAN_INVERSE_MAPPING = {v: k for k, v in BOXOBAN_MAPPING.items()}

GRIDDLY_INVERSE_MAPPING = {
    'empty': '.',
    'wall': 'w',
    'box': 'b',
    'goal': 'h',
    'player': 'A',
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