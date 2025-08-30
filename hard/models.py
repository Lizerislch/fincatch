import torch
from torch import nn
from utils import SPECIALS, _next_non_special, sanitize_input

class LLM_Simulator(nn.Module):
    def __init__(self, hidden: int = 64, output_dim: int = 10000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, hidden * 2), nn.LeakyReLU(),
            nn.Linear(hidden * 2, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, output_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def llm_argmax_with_rules(llm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    x∈[1,9] → 1111..9999（index=value-1）
    x∈[10,100] 若 argmax∈SPECIALS → 改写为非特殊
    """
    x = sanitize_input(x)
    with torch.no_grad():
        idx = llm(x).argmax(dim=1)
    forced = idx.clone()
    xi_int = x.view(-1).long()
    mask_1_9 = (xi_int >= 1) & (xi_int <= 9)
    forced[mask_1_9] = xi_int[mask_1_9] * 1111 - 1
    mask_else = ~mask_1_9
    if mask_else.any():
        v = idx[mask_else] + 1
        v_np = v.cpu().numpy()
        for i in range(v_np.shape[0]):
            vi = int(v_np[i])
            if vi in SPECIALS:
                v_np[i] = _next_non_special(vi)
        forced[mask_else] = torch.from_numpy(v_np).to(idx.device) - 1
    return forced

class AgentFactored(nn.Module):
    def __init__(self, hidden: int = 128, n_states: int = 10000):
        super().__init__()
        self.emb = nn.Embedding(n_states, 32)
        self.backbone = nn.Sequential(
            nn.Linear(32, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, hidden), nn.LeakyReLU(),
        )
        self.head_h = nn.Linear(hidden, 10)
        self.head_t = nn.Linear(hidden, 10)
        self.head_o = nn.Linear(hidden, 10)

    def forward(self, state_idx: torch.Tensor):
        z = self.backbone(self.emb(state_idx))
        return self.head_h(z), self.head_t(z), self.head_o(z)

    @staticmethod
    def sample_action(logits_h, logits_t, logits_o, greedy: bool = False,
                      forbid_same: bool = False, forbid_h0: bool = False):
        def apply_mask(lg, m): return lg + torch.log(m.float() + 1e-12)
        B = logits_h.size(0)
        mh = torch.ones((B,10), device=logits_h.device)
        mt = torch.ones((B,10), device=logits_h.device)
        mo = torch.ones((B,10), device=logits_h.device)
        if forbid_h0: mh[:,0] = 0
        lg_h = apply_mask(logits_h, mh)
        lg_t = apply_mask(logits_t, mt)
        lg_o = apply_mask(logits_o, mo)
        if greedy:
            h = lg_h.argmax(1); t_ = lg_t.argmax(1); o = lg_o.argmax(1)
        else:
            h = torch.distributions.Categorical(logits=lg_h).sample()
            t_ = torch.distributions.Categorical(logits=lg_t).sample()
            o = torch.distributions.Categorical(logits=lg_o).sample()
        if forbid_same:
            same = (h==t_) & (t_==o)
            if same.any(): o[same] = (o[same] + 1) % 10
        action_idx = h * 100 + t_ * 10 + o
        action_val = action_idx + 1
        return action_idx, action_val
