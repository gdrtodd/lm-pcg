from dataclasses import dataclass


@dataclass
class Config:

    model_name : str = "davinci"
    source : str = "microban" 

    # EVALUATION
    exp_no : int = 400003
    simulations : int = 1
    eval_only : bool = False
    experiment : str = "controllability" 


    # GENERATION
    prompt : str = "Map: ->"
    max_tokens : int = 150
    stop_token : str = ". END"
    temperature : float = 0.5
    top_p : float = 1.0

    # METRICS

    novelty_threshold : int = 5
    eval_tolerance : int = 5


