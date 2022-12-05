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