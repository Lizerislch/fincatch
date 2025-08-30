import torch
from rewards import count_occurrences_X_in_action, reward_from_matchcount_Q2

class Environment:
    """
    给定 state_idx (0..9999) 与 action_idx (0..999)，返回 (reward, k=#matches)
    这里的 k 为“多重匹配”计数
    """
    def step(self, state_idx: torch.Tensor, action_idx: torch.Tensor):
        state_val = state_idx + 1
        action_val = action_idx + 1
        cnt = count_occurrences_X_in_action(state_val, action_val)
        r = reward_from_matchcount_Q2(cnt)
        return r, cnt
