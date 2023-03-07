from dataclasses import dataclass


@dataclass
class TrainedModels:

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
    
    #control
    model_12 = {
        "10_epochs" : "davinci:ft-personal:microban-orig-level-control-1-10epochs-2023-02-10-18-32-25"
    }
    # MICROBAN FLIPS
    model_9 = {
        "10_epochs" : "davinci:ft-gameinnovationlab:microbanflips-level-sample-1-10epochs-2023-02-08-18-19-15"
    }

    # control

    model_13 = {
        "10_epochs" : "davinci:ft-personal:microban-flips-level-control-1-10epochs-2023-02-10-20-25-28"
    }
    
    # Microban  Flips Rotations
    # sample
    model_10 = {
        "10_epochs" : "davinci:ft-gameinnovationlab:microbanfp-level-sample-1-10epochs-2023-02-08-22-54-11"
    }
    # comtrol
    
    model_11 = {
        "10_epochs" : "davinci:ft-personal:microban-f-p-level-control-1-10epochs-2023-02-10-16-43-01"
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