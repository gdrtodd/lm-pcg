# As per hydra install docs
python -m pip install hydra-core hydra-submitit-launcher --upgrade

python -m pip install -r requirements.txt

# Installs the cross-eval hydra launcher plugin (for circumventing the multirun process when collecing results of sweep)
python -m pip install -e .