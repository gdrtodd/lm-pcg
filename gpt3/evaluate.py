import os
import pandas as pd
import numpy as np
from Levenshtein import distance

from helpers import h5_to_df
from sokoban_solvers import EnhancedAStarAgent, State







import openai
os.environ['OPENAI_API_KEY'] = "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"
openai.api_key = os.getenv("OPENAI_API_KEY")


TRAIN_PATH = "levels.csv"

def training_levels(path, is_h5=True):
    """
    Function to get training levels
    """
    if is_h5:
        df = h5_to_df(path)
    else:
        df = pd.read_csv(path)
    return list(df.level)
    


def is_unique(level):
    """
    Function to calculate diversity
    """
    return level not in training_levels(TRAIN_PATH,is_h5=False)


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
        solution, node, iters = solver.getSolution(level_state, maxIterations=1000000000000000000000)
        if not node.checkWin():
            if verbose: print("--Level cannot be solved (... in 50k steps)--")
            #return True
        elif verbose:
            print(f"++Level can be solved in {len(solution)} moves++")

        return solution


def is_novel(series, row_index):
    """
    Function to calculate novelty
    """
    distances = []
    for i, value in enumerate(series):
        if i != row_index:
            d = distance(series[row_index], value)
            if d < 5:
                return False
            
    return True#np.mean(distances)


def eval(model,simulations, model_name, exp_no):

    generations = {}
    generations["level"] = []
    generations["is_unique"] = []
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
        
        generations["level"].append(level)
        generations["is_unique"].append(is_unique(level))
        generations["is_playable"].append(is_playable(level, verbose = True))
        generations["is_novel"].append(is_novel(generations["level"],len(generations["level"])-1))
    

    df = pd.DataFrame(generations)
    is_p = df.loc[df.is_playable != False]# Playability
    is_u = is_p.loc[is_p.is_unique != False]# Diversity
    is_n = df.loc[df.is_novel != False]# novelty
    
    is_dpn = is_p.loc[is_p.is_novel != False] # Diversity of set of playable and novel levels


    df.to_csv(f"exp_results/result_{model_name}_{temp}-temp_{top_p}-top_p_{exp_no}.csv")
    return df, is_p.shape[0], is_u.shape[0], is_n.shape[0], is_dpn.shape[0]
        
        
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
    "6_epochs" : "davinci:ft-gameinnovationlab:600level-sample-1-4epochs-2023-02-06-21-26-43"
}

model_7 = {
    "1_epochs" : "davinci:ft-gameinnovationlab:600level-sample-2-1epochs-2023-02-06-22-11-00"
}
model_name = "davinci"

exp_no = 16

simulations = 100

df, is_p, is_u, is_n, is_dqn = eval(model_6["6_epochs"],simulations,model_name,exp_no)

print(f'Playability:{is_p/simulations}, Uniqueness: {is_u/simulations}, Novelty: {is_n/simulations}, Diversity of playable and novel levels: {is_dqn/simulations}')    