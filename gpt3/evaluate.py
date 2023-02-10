from metrics import is_novel, is_playable, get_diversity
from config import Config
from trained_models import TrainedModels
import os
import pandas as pd
import openai
os.environ['OPENAI_API_KEY'] = "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"
openai.api_key = os.getenv("OPENAI_API_KEY")



def infer(model,
          simulations, 
          model_name, 
          exp_no, 
          source,
          prompt,
          max_tokens,
          stop_token,
          temperature,
          top_p):

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
                                       prompt=prompt,
                                       max_tokens=max_tokens,
                                       temperature=temperature,
                                       top_p=top_p,
                                       stop=[stop_token]
                                      )
        
        level = gen["choices"][0]["text"][1:]
        print(level)
        generations["level"].append(level)
        generations["is_novel"].append(is_novel(level,source))
        generations["is_playable"].append(is_playable(level, verbose = True))
    

    df = pd.DataFrame(generations)
    path = f"exp_results/result_{model_name}_{temp}-temp_{top_p}-top_p_simulations-{simulations}_exp-no_{exp_no}.csv"
    df.to_csv(path)
    
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
    novelty = df.loc[df.is_novel != False]# novelty
    diversity = get_diversity(df["level"],args.novelty_threshold) #Diversity
    dpn = novelty.loc[novelty.is_playable != "False"] # Diversity of set of playable and novel levels
    restricted_diversity =  get_diversity(list(dpn["level"]),args.novelty_threshold) #Diversity of novel and playable levels
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

    for i,level in enumerate(list(df["level"])):

        generations["level"].append(level)
        generations["is_novel"].append(is_novel(level,dataset))

    generations["is_playable"] = list(df["is_playable"])
    df_g = pd.DataFrame(generations)

    return eval(path=" ",simulations=simulation, df = df_g, is_dataframe = True)


model = TrainedModels()
args = Config()

if args.eval_only:
    df = pd.read_csv("exp_results/result_davinci_0.5-temp_1-top_p_simulations-100_exp-no_300.csv",index_col=0)
else:
    df = infer(model.model_10["10_epochs"],
               args.simulations,
               args.model_name,
               args.exp_no,
               args.source,
               args.prompt,
               args.max_tokens,
               args.stop_token,
               args.temperature,
               args.top_p,)     

df, playability, diversity, novelty, restricted_diversity = eval_from_df(df,args.simulations,args.source)

print(f'Playability:{playability}, diversity: {diversity}, Novelty: {novelty}, Restricted Diversity: {restricted_diversity}')    