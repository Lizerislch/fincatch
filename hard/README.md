# ***HARD***

## At The Very Beginning

While working through this task, I had a few points of confusion that I’d like to share up front.  

First, in **Q2** there is this mysterious `XXXX`. Is it meant to be just a placeholder, or does it strictly mean “four identical digits” (like 1111, 2222, …)? The examples always show starting states in 1–9, which makes me think the problem was pushing us into those special dddd cases. But if we take it more generally, maybe `XXXX` was intended as an abstract placeholder rather than literally four of the same digit.  

Second, the **action definition** felt a bit odd. The spec says the agent outputs an action α ∈ [1,1000], but the reward rule is based on digit matching. That seemed inconsistent at first. I ended up assuming that the state is always a four-digit number (sometimes dddd), and the action is a three-digit number; then, any digit in the action that matches a digit in the state counts as a match. In other words, I treated it as a *multiset matching* problem, where each action digit can “consume” one matching digit from the state.  

Finally, in **Q3 Test Case 2** the original statement has `state = 1111, action = 199` and says the number of matching digits is 2. By my reasoning that looks like a typo: only one `1` matches between 1111 and 199. I treated it as a mistake in the problem description and stuck with my own matching logic.  

## To Execute
Edit config.py 

Run main.py

### File Descriptions

- **`config.py`**  
  Global configuration file. Defines random seed, device (CPU/GPU), and training hyperparameters (learning rate, batch size, training steps, etc.).

- **`utils.py`**  
  Utility functions, including:  
  - `set_seed`: ensure reproducibility  
  - `fmt4`: format integers into 4-digit strings  
  - Special-number helpers (e.g., `_next_non_special`)  

- **`rewards.py`**  
  Reward logic definitions:  
  - `count_occurrences_X_in_action`: multiset-based digit matching between state (4 digits) and action (3 digits)  
  - `reward_from_matchcount_Q2`: maps match count (1/2/3) → reward (10/20/100)  

- **`models.py`**  
  Model definitions:  
  - `LLM_Simulator`: Q1 MLP, mapping inputs to states  
  - `llm_argmax_with_rules`: enforce strict mapping rules for inputs 1–9  
  - `AgentFactored`: Q2 factored policy network with three categorical heads (hundreds, tens, ones)  

- **`envs.py`**  
  Environment definition:  
  - `Environment`: encapsulates `step()`, which takes `(state, action)` and returns `(reward, k)`  

- **`data_q1.py`**  
  Dataset generator:  
  - `synth_q1_dataset`: creates training data for Q1 (input 1–100 → valid state in 1–10000)  

- **`train_q1.py`**  
  Q1 training logic:  
  - `train_q1`: trains the LLM_Simulator using cross-entropy to enforce strict mapping constraints  

- **`train_q2.py`**  
  Q2 training logic:  
  - `pretrain_agent_bc`: Behavior Cloning pretraining to mitigate reward sparsity  
  - `train_q2`: REINFORCE algorithm for policy gradient training, with entropy bonus for exploration  

- **`demos.py`**  
  Demonstrations and tests:  
  - `q2_demo_print`: prints Q2 policy outputs (state → action → reward)  
  - `q3_episode`: runs a single Q3 episode (multi-step interaction until termination)  
  - `q3_run_tests`: runs multiple Q3 test cases and reports average reward  

- **`main.py`**  
  Project entry point:  
  - Sets random seed and device  
  - Executes Q1 training → initializes Q2 agent → pretrains with BC → trains with REINFORCE  
  - Runs Q2 demo and Q3 test cases  


## 1. Introduction
This project implements and analyzes a reinforcement learning (RL) agent in a toy environment inspired by financial decision-making.  
The task is divided into three subtasks:  

- **Q1:** Implement a Multi-Layer Perceptron (MLP) mapping inputs x ∈ [1,100] into states k ∈ [1,10000].  
- **Q2:** Design a factored policy network that selects an action α ∈ [1,1000] with digit-matching reward.  
- **Q3:** Build an environment, define multi-step interaction rules, and train the agent using reinforcement learning (REINFORCE + entropy regularization).  

We further extended the reward design into a **multiset digit matching** version, making the task more flexible and closer to realistic scenarios.


## 2. Methodology

### Q1: LLM Simulator (MLP)
- Input: integer x ∈ [1,100]  
- Output: class index y ∈ [0,9999], representing states 1–10000  
- Training: CrossEntropy loss on synthetic dataset with strict mapping rules  
- Acts as a **state encoder**, analogous to financial feature encoding

---

### Q2: Factored Policy Network
- Embedding → backbone MLP → three categorical heads (hundreds, tens, ones)  
- Prevents collapse to trivial policies (like always choosing ddd)  
- Modified reward function:  
  - Multiset intersection between state digits and action digits  
  - Rewards:  
    - 1 match → 10  
    - 2 matches → 20  
    - 3 matches → 100  

---

### Q3: Environment & RL Training
- **Environment**: (state, action) → (reward, k)  
- **Episode termination**:  
  - k == 0 (no matches)  
  - stabilized k  
  - 10 steps max  
- **Final settlement**: if stabilized at k == 3 → reward = 100 / steps, else 0  
- **Training strategy**:  
  - **Behavior Cloning (BC)** pretraining on synthetic trajectories  
  - **Policy Gradient (REINFORCE)** fine-tuning with entropy bonus  

---


## 3. Discussion on the question

- Without Pretraining:

  - High rewards are extremely rare as the probability of hitting all three correct is around 0.1%

- With Pretraining:

  - Moves the policy to a better starting point where 1-2 correct matches are common.

- Therefore:

  - Techniques like advantage normalization, Actor-Critic or PPO, combined with action constraints and reward shaping
  - Can Further stabilize and accelerate training.

## 4. Future directions and open-ended exploration

1. Insights from Current Task
- Spares Reward Issues
  - In the Current Task, High Reward will be given only when 3 digit match the LLM output. Which is Similar to the real world Finance, where opportunity is limited, very rare
  - Most of them are noise, therefore pretraining or shaping intermediate rewards are often necessary.

- Importance of Action Factorization
  - Using three independent number should be able to prevent collapse into trivial actions, although in My Demo, the action is always three same number.
  - The three independent number could represent like the action spaces in trading like asset quantity timing etc.

2. Toward Realistic Finance Agents

- To move beyond the toy “digit-matching” environment, we can map the same design principles onto finance datasets:
  - Open Source Datasets:
    - Yahoo Finance API, Kaggle Financial Datasets
        Or Framework like FinRL

3. Possible Extensions:

- To Replace the MLP with a financial state encoder
  - Use a real model to embed OHLCV Windows 
  - Action space as trading decisions:
    - Replace 3 digit action to action like buy/hold/sell.
  - Reward as portfolio returns / risk-adjusted returns:
    - -Analogous to “matches,” reward would be profit & loss, Sharpe ratio, or drawdown-aware metric.