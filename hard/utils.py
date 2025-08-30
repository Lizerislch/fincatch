import random
import numpy as np
import torch

SPECIALS = {1111,2222,3333,4444,5555,6666,7777,8888,9999}

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def fmt4(n: int) -> str:
    return f"{n:04d}"

def _next_non_special(v: int) -> int:
    if v in SPECIALS:
        v = (v + 1234) % 10000 or 1
        if v in SPECIALS:
            v = (v + 7) % 10000 or 1
    return v

def sanitize_input(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 1.0, 100.0)
