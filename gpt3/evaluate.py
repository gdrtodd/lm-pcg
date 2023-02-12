from metrics import is_novel, is_playable, get_diversity, is_accurate
from config import Config
from trained_models import TrainedModels
import os
import pandas as pd
import openai
os.environ['OPENAI_API_KEY'] = " "# ENETR OPENAI API KEY HERE
openai.api_key = os.getenv("OPENAI_API_KEY")

import random



def infer(model,
          simulations, 
          model_name, 
          exp_no, 
          source,
          prompt,
          max_tokens,
          stop_token,
          temperature,
          top_p,
          experiment):

    """
    inference and evaluations
    """

    generations = {}
    generations["level"] = []
    generations["is_novel"] = []
    generations["is_playable"] = []
    generations["is_accurate"] = []

    temp = 1
    top_p = 1

    
    if experiment == "sample":
        for sss in range(0, simulations):

            gen = openai.Completion.create(
                                           model=model,
                                           prompt=f"Map: ->",
                                           max_tokens=max_tokens,
                                           temperature=temperature,
                                           top_p=top_p,
                                           stop=[stop_token]
                                          )

            level = gen["choices"][0]["text"][1:]
            print(level)
            playt = is_playable(level, verbose = True)
            generations["level"].append(level)
            generations["is_novel"].append(is_novel(level,source))
            generations["is_playable"].append(playt)
            
    elif experiment == "controllability":
        #target_len = list(range(5,279,5))
        target_len = random.sample(range(1, 279), 100)
        for sss in range(0, simulations):

            gen = openai.Completion.create(
                                           model=model,
                                           prompt=f"solution length = {target_len[sss]}: ->",
                                           max_tokens=max_tokens,
                                           temperature=temperature,
                                           top_p=top_p,
                                           stop=[stop_token]
                                          )

            level = gen["choices"][0]["text"][1:]
            print(level)
            playt = is_playable(level, verbose = True)
            generations["level"].append(level)
            generations["is_novel"].append(is_novel(level,source))
            generations["is_playable"].append(playt)
            if playt == False:
                generations["is_accurate"].append(False)
                print(target_len[sss],playt)
            else:
                generations["is_accurate"].append(is_accurate(target_len[sss],len(playt)))
                print(target_len[sss],len(playt))
        
        
    

    df = pd.DataFrame(generations)
    path = f"exp_results/result_{model_name}_{temp}-temp_{top_p}-top_p_simulations-{simulations}_exp-no_{exp_no}.csv"
    df.to_csv(path)
    
    return df

          

def eval(path,simulations, df, is_dataframe=False, experiment = "sample"):

    """
    evals only.
    """
    if not is_dataframe:
        df = pd.read_csv(path,index_col=0)
    else:
        pass
    df['is_novel'] = df['is_novel'].astype(bool)
    
    df['is_playable'] = df['is_playable'].astype(str)

    accuracy = df.loc[df.is_accurate != False]
    playability = df.loc[df.is_playable != "False"]# Playability
    novelty = df.loc[df.is_novel != False]# novelty
    diversity = get_diversity(df["level"],args.novelty_threshold) #Diversity
    dpn = novelty.loc[novelty.is_playable != "False"] # Diversity of set of playable and novel levels
    
    
    restricted_diversity =  get_diversity(list(dpn["level"]),args.novelty_threshold) #Diversity of novel and playable levels
    
    if experiment == "controllability":
        
        df['is_accurate'] = df['is_accurate'].astype(bool)
        dpna = dpn.loc[dpn.is_accurate != False]
        control_score = get_diversity(list(dpna["level"]),args.novelty_threshold)
        prop_accuracy = accuracy.shape[0]/simulations
    
    prop_playable = playability.shape[0]/simulations
    diversity = diversity/simulations
    prop_novel = novelty.shape[0]/simulations
    

    restricted_diversity = restricted_diversity/simulations
    control_score = control_score/simulations

    return df, prop_playable, diversity, prop_novel, restricted_diversity, prop_accuracy, control_score

def eval_from_df(df,simulation,dataset,experiment):

    generations = {}
    generations["level"] = []
    generations["is_novel"] = []
    generations["is_playable"] = []

    
    

    for i,level in enumerate(list(df["level"])):

        generations["level"].append(level)
        generations["is_novel"].append(is_novel(level,dataset))

    generations["is_playable"] = list(df["is_playable"])
    if experiment == "controllability":
        generations["is_accurate"] = []
        generations["is_accurate"] = list(df["is_accurate"])
    df_g = pd.DataFrame(generations)

    return eval(path=" ",simulations=simulation, df = df_g, is_dataframe = True, experiment = experiment)


model = TrainedModels()
args = Config()

if args.eval_only:
    df = pd.read_csv("exp_results/microban_flips_rotation_sample_exp.csv",index_col=0)
else:
    df = infer(model.model_13["10_epochs"],
               args.simulations,
               args.model_name,
               args.exp_no,
               args.source,
               args.prompt,
               args.max_tokens,
               args.stop_token,
               args.temperature,
               args.top_p,
               args.experiment)     

df, playability, diversity, novelty, restricted_diversity, accuracy, control_score = eval_from_df(df,args.simulations,args.source,args.experiment)

print(f'Playability:{playability}, diversity: {diversity}, Novelty: {novelty}, Restricted Diversity: {restricted_diversity}')    