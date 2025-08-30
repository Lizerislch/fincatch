import numpy as np
import torch
from utils import _next_non_special

'''
To Generate Training Data base on the requirement
When input is 1-9, output should be 1111-9999 respectively.
For inputs 10-100, output should be any other value in the range.
'''
def synth_q1_dataset(n: int, x_low=1, x_high=100):
    X = np.random.randint(x_low, x_high + 1, n)
    Y = np.zeros((n,), dtype=int)
    for i, xi in enumerate(X):
        if 1 <= xi <= 9:
            Y[i] = xi * 1111
        else:
            Y[i] = _next_non_special(np.random.randint(1, 10001))
    return torch.tensor(X, dtype=torch.float32).view(-1,1), torch.tensor(Y-1, dtype=torch.long)
