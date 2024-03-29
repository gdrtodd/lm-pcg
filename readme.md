## Introduction

This is the repository for the paper [Level Generation Through Large Language Models](https://arxiv.org/abs/2302.05817). It trains various large language models to generate Sokoban game levels. Command-line arguments allow you to specify different model types, datasets (and dataset sizes), and controllability prompts. Code for training GPT3 relies on the OpenAI API, and is in `gpt3/`.

## Installation
```
bash setup.sh
```

## Training
To train a single model locally:
```
python train_lm.py
```
To launch a hyperparameter sweep on a SLURM cluster, run:
```
python train_lm.py --multirun
```
The config files are located in `config/`. Settings can be changed in `config/config.yaml` or overwritten on the command line, e.g.:
```
python train_lm.py batch_size=32
```

The hyperpameter sweeps used to generate the results in the paper are located in `config/experiment/`. E.g., running `python train_lm.py +experiment=pretraining -m` will launch a sweep over model types (i.e. pretrained vs. code-pretrained vs. un-pretrained).

## Evaluation

To evaluate
```
python evaluate.py +experiment=models
```
Add `render=True` to save images and gifs of generated levels and their nearest neighbor in the training set (measured by hamming distance).

## Cross-evaluation
To run a cross-evaluation (aggregating the results from evaluations above to compare the effect of different hyperparameters), run:
```
python cross_eval.py sweep=models
```
Here, we sweep across the train hyperparameters in `conf/experiment/models.yaml` (with the name of the yaml file passed as the `sweep` argument), and the eval hyperparameters in `conf/eval.yaml`.

## Datasets

### L-Maze

Generate the L-Maze dataset by running:
```
python generate_data.py`
```


### Boxoban

To pre-process the boxoban dataset, labelling levels with thie solutions, run:
```
python preprocess_boxoban.py
```
Supply `level_file_idx=10` to only label the levels in file with index 10. Supply `aggregate=True` to aggregate data from all files into a single file.

## Games

Config option `game`. These describe the mechanics of the games for which our datasets contain levels. Options are:

- `l_maze`
- `sokoban`

## Citation
If you use our work, please cite it as:

```
@inproceedings{todd2023level,
  title={Level Generation Through Large Language Models},
  author={Todd, Graham and Earle, Sam and Nasir, Muhammad Umair and Green, Michael Cerny and Togelius, Julian},
  booktitle={Proceedings of the 18th International Conference on the Foundations of Digital Games},
  pages={1--8},
  year={2023}
}
```