
import torch
from torch import nn

class MMDCSM(nn.Module):
    """
    Independent methodological reimplementation inspired by MMDCSM.

    Each modality has its own MLP encoder. Embeddings are summed, then
    converted into mixture weights over Weibull experts.
    """
    def __init__(self, d1, d2, hidden=32, latent=8, n_experts=2):
        super().__init__()
        self.n_experts = n_experts

        self.encoder1 = nn.Sequential(
            nn.Linear(d1, hidden), nn.ReLU(),
            nn.Linear(hidden, latent)
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(d2, hidden), nn.ReLU(),
            nn.Linear(hidden, latent)
        )

        self.gate = nn.Linear(latent, n_experts)

        # Positive Weibull parameters. Initialize around the synthetic
        # data-generating regime to avoid immediate expert collapse.
        def inv_softplus(x):
            return torch.log(torch.expm1(torch.tensor(float(x))))

        self.raw_shape = nn.Parameter(
            torch.stack([inv_softplus(1.4), inv_softplus(2.0)])
        )
        self.raw_scale = nn.Parameter(
            torch.stack([inv_softplus(18.0), inv_softplus(7.0)])
        )

    def forward(self, x1, x2):
        z1 = self.encoder1(x1)
        z2 = self.encoder2(x2)
        z = z1 + z2
        weights = torch.softmax(self.gate(z), dim=-1)
        shape = torch.nn.functional.softplus(self.raw_shape) + 0.25
        scale = torch.nn.functional.softplus(self.raw_scale) + 0.25
        return {"z1": z1, "z2": z2, "z": z,
                "weights": weights, "shape": shape, "scale": scale}

def weibull_log_pdf(t, shape, scale):
    t = t.clamp_min(1e-6)
    return (
        torch.log(shape)
        - shape * torch.log(scale)
        + (shape - 1.0) * torch.log(t)
        - (t / scale) ** shape
    )

def weibull_log_survival(t, shape, scale):
    t = t.clamp_min(1e-6)
    return -(t / scale) ** shape

def mixture_log_likelihood(output, time, event):
    """
    Event observations use log mixture PDF.
    Right-censored observations use log mixture survival.
    """
    w = output["weights"]
    shape = output["shape"]
    scale = output["scale"]

    log_pdf = weibull_log_pdf(
        time[:, None], shape[None, :], scale[None, :]
    )
    log_surv = weibull_log_survival(
        time[:, None], shape[None, :], scale[None, :]
    )

    log_w = torch.log(w.clamp_min(1e-8))
    event_ll = torch.logsumexp(log_w + log_pdf, dim=1)
    censored_ll = torch.logsumexp(log_w + log_surv, dim=1)

    ll = event * event_ll + (1.0 - event) * censored_ll
    return ll.mean()

def prior_loss(output, target_shape=1.5, target_scale=10.0):
    shape = output["shape"]
    scale = output["scale"]
    return ((shape - target_shape) ** 2).mean() + ((scale - target_scale) ** 2).mean()
