
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import MMDCSM, weibull_log_pdf, weibull_log_survival

def test_forward_shapes():
    model = MMDCSM(16, 12, hidden=32, latent=8, n_experts=2)
    out = model(torch.randn(10,16), torch.randn(10,12))
    assert out["z"].shape == (10,8)
    assert out["weights"].shape == (10,2)
    assert torch.allclose(out["weights"].sum(1), torch.ones(10), atol=1e-6)

def test_weibull_outputs_are_finite():
    t = torch.tensor([1.,2.,5.])
    shape = torch.tensor([1.5,2.0])
    scale = torch.tensor([5.,8.])
    assert torch.isfinite(weibull_log_pdf(t[:,None], shape[None,:], scale[None,:])).all()
    assert torch.isfinite(weibull_log_survival(t[:,None], shape[None,:], scale[None,:])).all()
