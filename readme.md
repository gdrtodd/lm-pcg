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

The hyperpameter sweeps used to generate the results in the paper are located in `config/experiment/`. E.g., running `python train_lm.py +experiment=model -m` will launch a sweep overr model types.

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
Here, we sweep across the train hyperparameters in `conf/experiment/models.yaml` (with the name of the yaml file passed as the `sweep` arguemtn), and the eval hyperparameters in `conf/eval.yaml`.

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

## Encoding schemes

Config option `data_source`. These describe per-token or per-row pre-processing of level files.

TODO: these should perhaps be dataset-agnostic?

- `l-maze` -- does nothing
- `boxoban-chars`
- `boxoban-text`

## TODO:

- try a character-based model like k9?
<!-- - force per-character tokenization (by adding special characters between ascii symbols or similar) -->
<!-- - re-write level to replace each token with what it represents (e.g. "wall, empty, empty, wall, player, wall"). (Would be great if we could also guarantee one token per tile to keep positions consistent.) -->
<!-- - semantically parse each row of the level as, e.g., "3 walls, 1 empty space, 1 player, 1 wall" -->