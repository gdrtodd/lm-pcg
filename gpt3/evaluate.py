import os

from helpers import h5_to_df
from sokoban_solvers import EnhancedAStarAgent, State


import openai
os.environ['OPENAI_API_KEY'] = "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"
openai.api_key = os.getenv("OPENAI_API_KEY")


MICROBAN_PATH = "microban_data.h5"

def training_levels(path):
    df = h5_to_df(path)
    return list(df.level)
    


def is_unique(level):
    return level not in training_levels(MICROBAN_PATH)


def is_playable(level, verbose=False):
        '''
        Determines whether the given level is playable by checking a variety of conditions:
          1. the level is rectangular (i.e. every line is the same length)
          2. the level contains only the following characters: "\n", "#", " ", "-", "@", "$", "."
          3. the level contains exactly one player
          4. the level contains the same number of boxes and goals (and at least one of each)
          5. the level can be solved by an ASTAR agent
        If the level is playable, return the solution (return False otherwise).
        '''
        solver = EnhancedAStarAgent()
        # Check if the level is rectangular
        line_lengths = [len(line) for line in level.split("\n")]
        if len(set(line_lengths)) != 1:
            if verbose: print("--Level is not rectangular--")
            return False

        # Check if the level contains only the allowed characters
        allowed_chars = set("\n# -@$.*+")
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
        solution, node, iters = solver.getSolution(level_state, maxIterations=50000)
        if not node.checkWin():
            if verbose: print("--Level cannot be solved (... in 50k steps)--")
            return False
        elif verbose:
            print(f"++Level can be solved in {len(solution)} moves++")

        return solution

# "davinci:ft-gameinnovationlab:microban-sample-2-10epochs-2023-02-05-21-20-44"
gens = openai.Completion.create(
  model="davinci:ft-gameinnovationlab:microban-sample-2-2023-02-05-20-39-06",
  prompt="Map. ->",
  max_tokens=150,
  temperature=1,
  stop = [". END"]
)

level = gens["choices"][0]["text"][1:]
print(is_unique(level))
print(is_playable(level,verbose=True))
print(level)

    