import os
import pandas as pd
import numpy as np
from Levenshtein import distance

from helpers import h5_to_df
from sokoban_solvers import EnhancedAStarAgent, State
import networkx as nx

from datasets import AnnotatedSokobanDataset


import openai
os.environ['OPENAI_API_KEY'] = "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"
openai.api_key = os.getenv("OPENAI_API_KEY")


TRAIN_PATH = "data_000.csv"

def training_levels(path, is_h5=True):
    """
    Function to get training levels
    """
    if is_h5:
        df = h5_to_df(path)
    else:
        df = pd.read_csv(path)
    return list(df.level)
    


def is_novel(level, novelty_threshold=5):
    """
    Function to calculate diversity
    """
    train_levels = training_levels(TRAIN_PATH,is_h5=False)

    for train_level in train_levels:
        if distance(level, train_level) < novelty_threshold:
            return False

    return True
    


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




def infer_and_eval(model,simulations, model_name, exp_no):

    """
    inference and evaluations
    """

    generations = {}
    generations["level"] = []
    generations["is_novel"] = []
    generations["is_playable"] = []

    temp = 0.55
    top_p = 1
    for sss in range(0, simulations):

        gen = openai.Completion.create(
                                       model=model,
                                       prompt="Map: ->",
                                       max_tokens=150,
                                       temperature=temp,
                                       top_p = top_p,
                                       stop = [". END"]
                                      )
        
        level = gen["choices"][0]["text"][1:]
        print(level)
        generations["level"].append(level)
        generations["is_novel"].append(is_novel(level))
        generations["is_playable"].append(is_playable(level, verbose = True))
    

    df = pd.DataFrame(generations)
    path = f"exp_results/result_{model_name}_{temp}-temp_{top_p}-top_p_simulations-{simulations}_exp-no_{exp_no}.csv"
    df.to_csv(path)
    
    return eval(path,simulations)
          

def eval(path,simulations):

    """
    evals only.
    """

    df = pd.read_csv(path,index_col=0)

    playability = df.loc[df.is_playable != 'False']# Playability
    novelty = df.loc[df.is_novel != 'False']# novelty
    diversity = get_diversity(df["level"],5) #Diversity
    dpn = novelty.loc[novelty.is_playable != 'False'] # Diversity of set of playable and novel levels
    restricted_diversity =  get_diversity(list(dpn["level"]),5) #Diversity of novel and playable levels

    prop_playable = playability.shape[0]/simulations
    diversity = diversity/simulations
    prop_novel = novelty.shape[0]/simulations
    restricted_diversity = restricted_diversity/simulations

    return df, prop_playable, diversity, prop_novel, restricted_diversity



        
### CHECKPOINTS
### MICROBAN
model_1 = {
"3_epochs" : "davinci:ft-gameinnovationlab:microban-sample-2-2023-02-05-20-39-06",
"5_epochs" : "davinci:ft-gameinnovationlab:microban-sample-2-5epochs-2023-02-05-21-38-42",
"7_epochs" : "davinci:ft-gameinnovationlab:microban-sample-2-7epochs-2023-02-06-04-12-28",
"10_epochs" : "davinci:ft-gameinnovationlab:microban-sample-2-10epochs-2023-02-05-21-20-44",
"15_epochs" : "davinci:ft-gameinnovationlab:microban-sample-2-15epochs-2023-02-06-07-16-30",
"25_epochs" : "davinci:ft-gameinnovationlab:microban-sample-4-25epochs-2023-02-06-09-34-00"
}

model_2 = {
    "5_epochs" : "davinci:ft-gameinnovationlab:microban-sample-3-3epochs-2023-02-06-07-40-13",
    "15_epochs" : "davinci:ft-gameinnovationlab:microban-sample-3-10epochs-2023-02-06-08-23-25",
}

model_3 = {

    "10_epochs" : "davinci:ft-gameinnovationlab:microban-sample-4-10epochs-2023-02-06-08-54-14"
}

model_4 = {
    "5_epochs" : "curie:ft-gameinnovationlab:microban-sample-5-5epochs-2023-02-06-10-54-34"
}


### 600Level Boxoban

model_5 = {
    "2_epochs" : "davinci:ft-gameinnovationlab:600level-sample-1-2epochs-2023-02-06-20-33-06"
}

model_6 = {
    "2_epochs" : "davinci:ft-gameinnovationlab:600level-sample-1-2epochs-2023-02-06-21-02-50",
    "3_epochs" : "davinci:ft-gameinnovationlab:600level-sample-1-3epochs-2023-02-06-21-48-09",
    "6_epochs" : "davinci:ft-gameinnovationlab:600level-sample-1-4epochs-2023-02-06-21-26-43", # temp 0.55 
    "10_epochs" : "davinci:ft-gameinnovationlab:600level-sample-1-10epochs-2023-02-07-14-33-50"# exp 104
}

model_7 = {
    "1_epochs" : "davinci:ft-gameinnovationlab:600level-sample-2-1epochs-2023-02-06-22-11-00"
}

### 4000Level Boxoban

model_8 = {
    "1_epochs" : "davinci:ft-gameinnovationlab:4000level-sample-1-1epochs-2023-02-07-15-03-24"
}
model_name = "davinci"

exp_no = 104003

simulations = 100

df, playability, diversity, novelty, restricted_diversity = infer_and_eval(model_6["10_epochs"],simulations,model_name,exp_no)
#df, playability, diversity, novelty, restricted_diversity = eval("exp_results/result_davinci_0.55-temp_1-top_p_simulations-100_exp-no_104002.csv",simulations)

print(f'Playability:{playability}, diversity: {diversity}, Novelty: {novelty}, Restricted Diversity: {restricted_diversity}')    