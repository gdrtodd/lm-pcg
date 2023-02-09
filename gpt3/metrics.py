import os
import pandas as pd
import numpy as np
from Levenshtein import distance

from helpers import h5_to_df
from sokoban_solvers import EnhancedAStarAgent, State
import networkx as nx


def training_levels(path, is_h5=True):
    """
    Function to get training levels
    """
    if is_h5:
        df = h5_to_df(path)
    else:
        df = pd.read_csv(path)
    return list(df.level)
    


def is_novel(level,dataset, novelty_threshold=5):
    """
    Function to calculate diversity
    """
    if dataset == "boxoban":
        train_path = "data_000.csv"
        is_h5 = False
    elif dataset == "microban":
        train_path = "microban_flips_rotations_data.h5"
        is_h5 = True

    train_levels = training_levels(train_path,is_h5=is_h5)

    for train_level in train_levels:
        if distance(level, train_level) < novelty_threshold:
            return False

    return True

'''df = pd.read_csv("exp_results/result_davinci_0.55-temp_1-top_p_simulations-100_exp-no_104003.csv")

novel = []
for level in df["level"]:
    novel.append(is_novel(level))
print(sum(novel)/100)'''

    


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
        #print(set(line_lengths))
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
        solution, node, iters = solver.getSolution(level_state, maxIterations=1e7)
        if not node.checkWin():
            if verbose: print(f"--Level cannot be solved (... in {iters} steps)--")
            return False
        elif verbose:
            print(f"++Level can be solved in {len(solution)} moves++")

        return solution



def get_diversity(levels, novelty_threshold, clique_limit=1000000):
        '''
        Returns the 'diversity' of a set of levels, defined as the size of the largest subset of levels
        that are all at least 'novelty_threshold' edit distance away from each other. We compute this 
        by constructing a graph where levels are adjacent if their edit distance is at least the threshold,
        and then finding the size of the largest clique in the graph.
        '''

        graph = nx.Graph()
        graph.add_nodes_from(range(len(levels)))

        edges = []
        for i in range(len(levels)):
            for j in range(i+1, len(levels)):
                if distance(levels[i], levels[j]) >= novelty_threshold:
                    edges.append((i, j))

        graph.add_edges_from(edges)

        biggest_clique = -1
        num_cliques = 0

        for clique in nx.find_cliques(graph):
            if len(clique) > biggest_clique:
                biggest_clique = len(clique)
            num_cliques += 1

            if num_cliques > clique_limit:
                break


        return biggest_clique
