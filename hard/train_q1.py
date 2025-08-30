import torch
from torch import nn
from config import Q1_EPOCHS, Q1_LR, Q1_WEIGHT_DECAY, DEVICE, Q1_SAMPLES
from models import LLM_Simulator
from data_q1 import synth_q1_dataset
'''
Training base on the training data
'''
def train_q1(epochs=Q1_EPOCHS, n=Q1_SAMPLES, lr=Q1_LR, device=DEVICE) -> LLM_Simulator:
    print("\n=== Q1: Train LLM_Simulator (strict constraints) ===")
    llm = LLM_Simulator().to(device)
    X, Y = synth_q1_dataset(n)
    ds = torch.utils.data.TensorDataset(X.to(device), Y.to(device))
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    opt = torch.optim.Adam(llm.parameters(), lr=lr, weight_decay=Q1_WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss()
    llm.train()
    for ep in range(1, epochs+1):
        tot = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = ce(llm(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
        print(f"[Q1] epoch {ep}/{epochs}  loss={tot/len(ds):.4f}")
    return llm
