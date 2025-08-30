import torch

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Q1
Q1_EPOCHS = 10
Q1_SAMPLES = 12000
Q1_LR = 1e-3
Q1_WEIGHT_DECAY = 5e-4

# Q2
Q2_BC_EPOCHS = 30
Q2_LR = 1e-3
Q2_STEPS = 1500
Q2_BATCH = 64
ENTROPY_COEF = 1e-3

# Demo
DEMO_Q2_INPUTS = [1, 2, 3]
DEMO_Q3_STARTS = [1, 2, 10]
