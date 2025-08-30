import torch
from torch import nn
from config import Q2_BC_EPOCHS, Q2_LR, Q2_STEPS, Q2_BATCH, ENTROPY_COEF, DEVICE
from models import AgentFactored, llm_argmax_with_rules
from envs import Environment

def pretrain_agent_bc(agent: AgentFactored, epochs=Q2_BC_EPOCHS, lr=Q2_LR, device=DEVICE):
    print(f"\n=== Q2-BC: Factored Agent Behavior Cloning ({epochs} epochs) ===")
    agent.train()
    opt = torch.optim.Adam(agent.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    def rand_not_d(d, B, device):
        r = torch.randint(0, 10, (B,), device=device)
        return torch.where(r == d, (r + 1) % 10, r)

    Bsz = 256
    for ep in range(1, epochs + 1):
        total = 0.0
        for _ in range(8):
            reps = Bsz // 9
            d = torch.arange(1, 10, device=device).repeat_interleave(reps)
            s_idx = d * 1111 - 1

            mode = torch.distributions.Categorical(
                probs=torch.tensor([0.2, 0.3, 0.5], device=device)
            ).sample((s_idx.size(0),))
            h = torch.empty_like(d); t_ = torch.empty_like(d); o = torch.empty_like(d)
            for i in range(d.size(0)):
                di = d[i].item()
                if mode[i] == 2:  # 3-match
                    h[i] = t_[i] = o[i] = d[i]
                elif mode[i] == 1:  # 2-match
                    pos = torch.randint(0, 3, (1,), device=device).item()
                    vals = [d[i], d[i], d[i]]
                    vals[pos] = rand_not_d(di, 1, device)
                    h[i], t_[i], o[i] = vals
                else:  # 1-match
                    pos = torch.randint(0, 3, (1,), device=device).item()
                    vals = [rand_not_d(di,1,device), rand_not_d(di,1,device), rand_not_d(di,1,device)]
                    vals[pos] = d[i]
                    h[i], t_[i], o[i] = vals

            lg_h, lg_t, lg_o = agent(s_idx)
            loss = ce(lg_h, h.long()) + ce(lg_t, t_.long()) + ce(lg_o, o.long())
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"[Q2-BC] epoch {ep}/{epochs}  loss={total/8:.4f}")

def train_q2(agent: AgentFactored, llm, steps=Q2_STEPS, batch=Q2_BATCH,
             lr=Q2_LR, device=DEVICE, entropy_coef: float = ENTROPY_COEF):
    print("\n=== Q2: REINFORCE (inputs 1..9 to ensure state=dddd) ===")
    llm.eval()
    env = Environment()
    opt = torch.optim.Adam(agent.parameters(), lr=lr)

    for t in range(1, steps+1):
        x = torch.randint(1, 10, (batch,1), dtype=torch.float32, device=device)
        with torch.no_grad():
            state_idx = llm_argmax_with_rules(llm, x)

        lg_h, lg_t, lg_o = agent(state_idx)
        dist_h = torch.distributions.Categorical(logits=lg_h)
        dist_t = torch.distributions.Categorical(logits=lg_t)
        dist_o = torch.distributions.Categorical(logits=lg_o)
        h = dist_h.sample(); t_ = dist_t.sample(); o = dist_o.sample()

        # 可选：避免百位为 0
        # h = torch.where(h==0, (h+1)%10, h)

        action_idx = h * 100 + t_ * 10 + o
        r, _ = env.step(state_idx, action_idx)

        baseline = r.mean()
        logp = dist_h.log_prob(h) + dist_t.log_prob(t_) + dist_o.log_prob(o)
        entropy = dist_h.entropy().mean() + dist_t.entropy().mean() + dist_o.entropy().mean()
        loss = -((r - baseline).detach() * logp).mean() - entropy_coef * entropy

        opt.zero_grad(); loss.backward(); opt.step()
        if t % 200 == 0:
            print(f"[Q2-RL] step {t}/{steps}  avg_reward={r.mean().item():.2f}  entropy={(entropy/3).item():.3f}")
