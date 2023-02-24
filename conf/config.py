from dataclasses import MISSING, dataclass
import typing

from hydra.core.config_store import ConfigStore

@dataclass
class Config:
    """This class provides a "schema" for the config file, validating types."""

    # Dataset
    game: str = "sokoban"
    source: str = "boxoban"  # choices=["boxoban", "microban"]
    char_encoding: bool = False
    level_key: str = "level"
    annotation_keys: typing.Optional[typing.List[str]] = None
    num_annotation_buckets: typing.Optional[int] = None
    holdout_solution_lens: typing.Optional[typing.List[int]] = None
    chunk_size: int = 128
    novelty_threshold: int = 5
    sample_prop: typing.Optional[float] = 1.0
    lora: bool = False

    # Model
    model: str = "gpt2"  # choices=["gpt2", "codeparrot", "java-gpt2", "incoder-1B", "incoder-6B"]
    warmup_proportion: float = 0.0002
    weight_decay: float = 0.01
    max_grad_norm: int = 1
    learning_rate: float = 1e-4

    # Run
    run_name: str = ""  # This gets set later by `get_run_name(cfg)`
    exp_name: str = ""
    overwrite: bool = False  # Overwrite the output directory if it exists (otherwise, attempt to load train state)
    seed: int = 42
    batch_size: int = 32
    # epochs: int = 20
    num_train_steps: int = 100_000
    save_freq: int = 1000
    eval_freq: int = 5000
    no_log: bool = False

    # Generation
    render: bool = False
    num_eval_proc: int = 1
    num_eval_samples: int = 10
    gen_freq: int = 1000
    gen_len: int = 128
    gen_temp: float = 1
    gen_beams: int = 5
    gen_top_k: int = 50
    gen_top_p: float = 1.0
    gen_typical_p: float = 1.0
    sample_contexts: bool = False
    sample_sequential: bool = False
    eval_tolerance: int = 5
    eval_controllability: bool = False
    n_search_iters: int = 10_000_000
    


@dataclass
class EvalConfig(Config):
    num_eval_samples: int = 100
    num_eval_proc: int = 10  # For computing solutions in parallel
    render: bool = False
    sample_sequential: bool = False

    # How much astar to do
    n_search_iters: int = 150_000


@dataclass
class CrossEvalConfig(EvalConfig):

    # This cross-eval refers to the sweep defined at `conf/experiment/EXPERIMENT.yaml`
    sweep: str = "models"

    # Printout the latest checkpoints for each experiment in the sweep.
    gen_table: bool = False

    # How many completely trained seeds to use in the cross-eval.
    max_trials: int = 5


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(name="eval_config", node=EvalConfig)
cs.store(name="cross_eval_config", node=CrossEvalConfig)