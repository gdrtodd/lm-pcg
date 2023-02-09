from metrics import is_novel, is_playable, get_diversity

import os
import pandas as pd
import openai
os.environ['OPENAI_API_KEY'] = "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"
openai.api_key = os.getenv("OPENAI_API_KEY")



def infer(model,simulations, model_name, exp_no, dataset):

    """
    inference and evaluations
    """

    generations = {}
    generations["level"] = []
    generations["is_novel"] = []
    generations["is_playable"] = []

    temp = 1
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
        generations["is_novel"].append(is_novel(level,dataset))
        generations["is_playable"].append(is_playable(level, verbose = True))
    

    df = pd.DataFrame(generations)
    path = f"exp_results/result_{model_name}_{temp}-temp_{top_p}-top_p_simulations-{simulations}_exp-no_{exp_no}.csv"
    df.to_csv(path)
    
    #return eval(path,simulations)
    #return eval_from_df(df,simulations)
    return df

          

def eval(path,simulations, df, is_dataframe=False):

    """
    evals only.
    """
    if not is_dataframe:
        df = pd.read_csv(path,index_col=0)
    else:
        pass
    df['is_novel'] = df['is_novel'].astype(bool)
    df['is_playable'] = df['is_playable'].astype(str)


    playability = df.loc[df.is_playable != "False"]# Playability
    novelty = df.loc[df.is_novel == True]# novelty
    diversity = get_diversity(df["level"],5) #Diversity
    dpn = novelty.loc[novelty.is_playable != "False"] # Diversity of set of playable and novel levels
    restricted_diversity =  get_diversity(list(dpn["level"]),5) #Diversity of novel and playable levels
    print(restricted_diversity)
    prop_playable = playability.shape[0]/simulations
    diversity = diversity/simulations
    prop_novel = novelty.shape[0]/simulations
    restricted_diversity = restricted_diversity/simulations

    return df, prop_playable, diversity, prop_novel, restricted_diversity

def eval_from_df(df,simulation,dataset):

    generations = {}
    generations["level"] = []
    generations["is_novel"] = []
    generations["is_playable"] = []

    for level in list(df["level"]):

        generations["level"].append(level)
        generations["is_novel"].append(is_novel(level,dataset))
    #generations["is_playable"].append(is_playable(level, verbose = True))
    generations["is_playable"] = list(df["is_playable"])


    df_g = pd.DataFrame(generations)

    return eval(path=" ",simulations=simulation, df = df_g, is_dataframe = True)




        
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

# MICROBAN FLIPS
model_9 = {
    "10_epochs" : "davinci:ft-gameinnovationlab:microbanflips-level-sample-1-10epochs-2023-02-08-18-19-15"
}

# Microban  Flips Rotations

model_10 = {
    "10_epochs" : "davinci:ft-gameinnovationlab:microbanfp-level-sample-1-10epochs-2023-02-08-22-54-11"
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
    "1_epochs" : "davinci:ft-gameinnovationlab:4000level-sample-1-1epochs-2023-02-07-15-03-24",
    "3_epochs" : "davinci:ft-gameinnovationlab:4000level-sample-1-3epochs-2023-02-07-17-21-04",
}
model_name = "davinci"

exp_no = 400003

simulations = 100

eval_only = False

dataset = "microban"

if eval_only:
    df = pd.read_csv("exp_results/result_davinci_0.5-temp_1-top_p_simulations-100_exp-no_400002.csv",index_col=0)
else:
    df = infer(model_10["10_epochs"],simulations,model_name,exp_no,dataset)

df, playability, diversity, novelty, restricted_diversity = eval_from_df(df,simulations,dataset)

print(f'Playability:{playability}, diversity: {diversity}, Novelty: {novelty}, Restricted Diversity: {restricted_diversity}')    