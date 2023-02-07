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