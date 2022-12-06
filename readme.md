## Installation
```
bash setup.sh
```

## Training
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

### TODO:

- try a character-based model like k9
- force per-character tokenization (by adding special characters between ascii symbols or similar)
- re-write level to replace each token with what it represents (e.g. "wall, empty, empty, wall, player, wall"). (Would be great if we could also guarantee one token per tile to keep positions consistent.)
- semantically parse each row of the level as, e.g., "3 walls, 1 empty space, 1 player, 1 wall"