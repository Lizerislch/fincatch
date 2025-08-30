import torch
from utils import fmt4
from models import AgentFactored, llm_argmax_with_rules
from envs import Environment

def q2_demo_print(llm, agent: AgentFactored, inputs, device,
                  greedy=True, forbid_same=False, forbid_h0=False):
    print("\n--- Q2 Demo ---")
    env = Environment()
    with torch.no_grad():
        for inp in inputs:
            x = torch.tensor([[float(inp)]], device=device)
            state_idx = llm_argmax_with_rules(llm, x)
            state_val = int(state_idx.item()) + 1
            lg_h, lg_t, lg_o = agent(state_idx)
            action_idx, action_val = AgentFactored.sample_action(
                lg_h, lg_t, lg_o, greedy=greedy, forbid_same=forbid_same, forbid_h0=forbid_h0
            )
            r, cnt = env.step(state_idx, action_idx)
            print(f"Input {inp} → LLM → state = {fmt4(state_val)} → Agent → Action = {int(action_val.item())} → reward = {int(r.item())} (matches={int(cnt.item())})")

def q3_episode(llm, agent: AgentFactored, start: int, device,
               max_steps: int = 10, greedy=True, forbid_same=False, forbid_h0=False) -> float:
    from envs import Environment
    env = Environment()
    prev_k = None
    x = torch.tensor([[float(start)]], device=device)
    for t in range(1, max_steps + 1):
        with torch.no_grad():
            state_idx = llm_argmax_with_rules(llm, x)
            state_val = int(state_idx.item()) + 1
            lg_h, lg_t, lg_o = agent(state_idx)
            action_idx, action_val = AgentFactored.sample_action(
                lg_h, lg_t, lg_o, greedy=greedy, forbid_same=forbid_same, forbid_h0=forbid_h0
            )
            _, cnt = env.step(state_idx, action_idx)
            k = int(cnt.item())

        if t == 1:
            print(f"Input {int(x.item())} → LLM → state = {fmt4(state_val)} → Agent → Action = {int(action_val.item())} → Environment: returns matches → State = {k}")
        else:
            print(f"→ LLM → state = {fmt4(state_val)} → Agent → Action = {int(action_val.item())} → Environment: returns matches → State = {k}")

        if k == 0:
            print("→ Done: state is 0 (no matches). Final Reward: 0\n")
            return 0.0
        if prev_k is not None and k == prev_k:
            if k == 3:
                print(f"→ Done: stabilized at 3 → reward = 100/{t}")
                print(f"Final Reward: {100.0/t:.1f}\n")
                return 100.0 / t
            else:
                print("→ Done: stabilized but not 3 → Final Reward: 0\n")
                return 0.0
        prev_k = k
        x = torch.tensor([[float(k)]], device=device)
    print("→ Not stabilized within 10 interactions → Final Reward: 0\n")
    return 0.0

def q3_run_tests(llm, agent: AgentFactored, starts, **kwargs):
    print("\n=== Q3 Tests ===")
    scores = []
    for s in starts:
        print(f"Test Case (start={s}):")
        scores.append(q3_episode(llm, agent, s, **kwargs))
    avg = sum(scores)/len(scores) if scores else 0.0
    print(f"Average Final Reward over tests: {avg:.2f}")
    return scores
