
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from model import MMDCSM, mixture_log_likelihood, prior_loss

def fit_model(train, d1, d2, hidden=32, latent=8, epochs=120,
              batch_size=128, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    model = MMDCSM(d1, d2, hidden=hidden, latent=latent, n_experts=2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ds = TensorDataset(
        torch.tensor(train["x1"]),
        torch.tensor(train["x2"]),
        torch.tensor(train["time"]),
        torch.tensor(train["event"]),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for x1, x2, time, event in loader:
            out = model(x1, x2)
            nll = -mixture_log_likelihood(out, time, event)
            # Small prior penalty for stable expert initialization.
            loss = nll + 1e-5 * prior_loss(out)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())

        history.append(float(np.mean(losses)))

    return model, history
