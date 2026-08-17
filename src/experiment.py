
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import make_synthetic_survival_data, train_test_split_data
from train import fit_model
from model import mixture_log_likelihood
from metrics import concordance_index, cluster_ari, risk_separation

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

with open(ROOT / "configs" / "experiment.json") as f:
    cfg = json.load(f)

data = make_synthetic_survival_data(
    n=cfg["n_samples"],
    d1=cfg["modality_1_dim"],
    d2=cfg["modality_2_dim"],
    seed=cfg["seed"],
    censoring_rate=cfg["censoring_rate_target"],
)
train, test = train_test_split_data(data, seed=SEED)

model, history = fit_model(
    train,
    d1=cfg["modality_1_dim"],
    d2=cfg["modality_2_dim"],
    hidden=cfg["hidden_dim"],
    latent=cfg["latent_dim"],
    epochs=cfg["epochs"],
    batch_size=cfg["batch_size"],
    lr=cfg["learning_rate"],
    seed=SEED,
)

model.eval()
with torch.no_grad():
    out = model(torch.tensor(test["x1"]), torch.tensor(test["x2"]))
    ll = mixture_log_likelihood(
        out, torch.tensor(test["time"]), torch.tensor(test["event"])
    ).item()

weights = out["weights"].numpy()
clusters = weights.argmax(axis=1)
# Higher mixture weight on the expert with smaller scale = higher risk.
expert_scale = out["scale"].numpy()
risk = (weights / expert_scale[None, :]).sum(axis=1)

cindex = concordance_index(test["time"], test["event"], risk)
ari = cluster_ari(test["subtype"], clusters)
sep = risk_separation(test["time"], test["event"], clusters, risk)
censoring = 1.0 - test["event"].mean()

metrics = pd.DataFrame([{
    "test_negative_log_likelihood": -ll,
    "concordance_index": cindex,
    "cluster_ARI": ari,
    "risk_separation": sep,
    "test_censoring_rate": censoring,
}])
metrics.to_csv(ROOT/"results"/"tables"/"mmdcsm_metrics.csv", index=False)

pd.DataFrame({"epoch": np.arange(1, len(history)+1),
              "loss": history}).to_csv(
    ROOT/"results"/"tables"/"training_history.csv", index=False
)

pd.DataFrame({
    "true_subtype": test["subtype"],
    "predicted_cluster": clusters,
    "risk_score": risk,
    "observed_time": test["time"],
    "event": test["event"],
    "expert_0_weight": weights[:,0],
    "expert_1_weight": weights[:,1],
}).to_csv(ROOT/"results"/"tables"/"test_predictions.csv", index=False)

# Training curve
plt.figure(figsize=(8,5))
plt.plot(history)
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("MMDCSM Training Curve")
plt.tight_layout()
plt.savefig(ROOT/"results"/"figures"/"training_loss.png", dpi=300)
plt.close()

# Latent representation
z = out["z"].numpy()
plt.figure(figsize=(8,6))
plt.scatter(z[:,0], z[:,1], c=clusters, alpha=0.7)
plt.xlabel("Latent dimension 1")
plt.ylabel("Latent dimension 2")
plt.title("Learned Multimodal Representation")
plt.tight_layout()
plt.savefig(ROOT/"results"/"figures"/"latent_clusters.png", dpi=300)
plt.close()

# Cluster distribution
plt.figure(figsize=(7,5))
unique, counts = np.unique(clusters, return_counts=True)
plt.bar(unique.astype(str), counts)
plt.xlabel("Learned cluster")
plt.ylabel("Number of test subjects")
plt.title("Learned Subtype Distribution")
plt.tight_layout()
plt.savefig(ROOT/"results"/"figures"/"cluster_distribution.png", dpi=300)
plt.close()

print(metrics.to_string(index=False))
print("\nLearned Weibull shapes:", out["shape"].numpy())
print("Learned Weibull scales:", out["scale"].numpy())
print("Cluster counts:", np.bincount(clusters))
