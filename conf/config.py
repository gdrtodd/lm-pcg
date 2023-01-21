from dataclasses import MISSING, dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore


@dataclass
class Config:
    """This class provides a "schema" for the config file, validating types."""

    # Dataset
    game: str = "sokoban"
    data_source: Optional[str] = None
    annotation_level: Optional[str] = None
    chunk_size: int = 128

    # Model
    model: str = "gpt2"  # choices=["gpt2", "codeparrot", "java-gpt2", "incoder-1B", "incoder-6B"]
    warmup_proportion: float = 0.0002
    weight_decay: float = 0.01
    max_grad_norm: int = 1
    learning_rate: float = 1e-4

    # Run
    exp_name: str = ""
    overwrite: bool = False  # Overwrite the output directory if it exists (otherwise, attempt to load train state)
    seed: int = 42
    batch_size: int = 16
    epochs: int = 20
    save_freq: int = 1000
    eval_freq: int = 1000
    no_log: bool = False

    # Generation
    num_eval_samples: int = 20
    gen_freq: int = 500
    gen_len: int = 128
    # gen_context: str = ""
    gen_temp: float = 1
    gen_beams: int = 5
    gen_top_k: int = 50
    gen_top_p: float = 1.0
    gen_typical_p: float = 1.0


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)