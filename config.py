from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


@dataclass
class TrainConfig:
    seed: int = 1337
    total_batch_size: int = int((2**17) * 4)
    micro_batch_size: int = 32
    context_length: int = 1024
    use_compile: bool = True
    max_lr: float = 18e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 715
    max_steps: int = 19073
    weight_decay: float = 0.1
    val_steps: int = 250
    ckpt_steps: int = 5000
    hellaswag_steps: int = 250
