import torch

def count_occurrences_X_in_action(state_val_4: torch.Tensor, action_val_3: torch.Tensor) -> torch.Tensor:
    """
    多重匹配（multiset）计数：
      k = sum_d min(count_state[d], count_action[d]), d∈{0..9}
    注：state 若不足四位，包含前导 0；如不希望匹配 0，可屏蔽 0 列。
    """
    # state 的四位
    s_th = (state_val_4 // 1000) % 10
    s_h  = (state_val_4 // 100) % 10
    s_t  = (state_val_4 // 10) % 10
    s_o  = state_val_4 % 10
    S = torch.stack([s_th, s_h, s_t, s_o], dim=1).long()  # [B,4]

    # action 的三位
    a_h  = (action_val_3 // 100) % 10
    a_t  = (action_val_3 // 10) % 10
    a_o  = action_val_3 % 10
    A = torch.stack([a_h, a_t, a_o], dim=1).long()        # [B,3]

    Bsz = state_val_4.size(0); device = state_val_4.device
    cnt_S = torch.zeros((Bsz, 10), dtype=torch.int32, device=device)
    cnt_A = torch.zeros((Bsz, 10), dtype=torch.int32, device=device)
    cnt_S.scatter_add_(1, S, torch.ones_like(S, dtype=torch.int32))
    cnt_A.scatter_add_(1, A, torch.ones_like(A, dtype=torch.int32))

    # 屏蔽 0（可选）：
    # cnt_S[:, 0] = 0; cnt_A[:, 0] = 0

    matches = torch.minimum(cnt_S, cnt_A).sum(dim=1)
    return matches.to(torch.int64)

def reward_from_matchcount_Q2(cnt: torch.Tensor) -> torch.Tensor:
    r = torch.zeros_like(cnt, dtype=torch.float32)
    r = torch.where(cnt == 1, torch.tensor(10.0, device=cnt.device), r)
    r = torch.where(cnt == 2, torch.tensor(20.0, device=cnt.device), r)
    r = torch.where(cnt >= 3, torch.tensor(100.0, device=cnt.device), r)
    return r
