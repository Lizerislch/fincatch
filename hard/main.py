from config import *
from utils import set_seed
from models import AgentFactored
from train_q1 import train_q1
from train_q2 import pretrain_agent_bc, train_q2
from demos import q2_demo_print, q3_run_tests

if __name__ == "__main__":
    set_seed(SEED)
    print(f"PyTorch device: {DEVICE}")

    # Q1
    llm = train_q1()

    # Q2
    agent = AgentFactored().to(DEVICE)
    pretrain_agent_bc(agent)
    train_q2(agent, llm)

    # Q2 demo（可切换 forbid_same / forbid_h0）
    q2_demo_print(llm, agent, DEMO_Q2_INPUTS, device=DEVICE, greedy=True, forbid_same=False, forbid_h0=False)

    # Q3 tests
    q3_run_tests(llm, agent, starts=DEMO_Q3_STARTS, device=DEVICE, greedy=True, forbid_same=False, forbid_h0=False)
