
import numpy as np
import torch
from sklearn.model_selection import train_test_split

def make_synthetic_survival_data(
    n=1000, d1=16, d2=12, seed=42, censoring_rate=0.30
):
    """Generate two modalities with a known two-subtype survival mechanism."""
    rng = np.random.default_rng(seed)

    subtype = rng.integers(0, 2, size=n)

    latent = rng.normal(size=(n, 3))
    latent[:, 0] += np.where(subtype == 1, 1.5, -1.5)
    latent[:, 1] += np.where(subtype == 1, 0.8, -0.8)

    W1 = rng.normal(scale=0.7, size=(3, d1))
    W2 = rng.normal(scale=0.7, size=(3, d2))
    x1 = latent @ W1 + rng.normal(scale=0.8, size=(n, d1))
    x2 = latent @ W2 + rng.normal(scale=0.8, size=(n, d2))

    # Subtype-dependent Weibull parameters.
    shape = np.where(subtype == 0, 1.4, 2.0)
    scale = np.where(subtype == 0, 18.0, 7.0)

    # A continuous risk effect shared across both modalities.
    risk_signal = 0.18 * latent[:, 0] + 0.10 * latent[:, 1]
    scale_i = scale * np.exp(-risk_signal)

    u = rng.uniform(size=n)
    event_time = scale_i * (-np.log(u)) ** (1.0 / shape)

    # Calibrate an independent administrative censoring distribution.
    censor_scale = np.quantile(event_time, 1.0 - censoring_rate)
    censor_time = rng.exponential(censor_scale, size=n)

    observed_time = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(np.float32)

    # Standardize modalities.
    x1 = (x1 - x1.mean(0)) / (x1.std(0) + 1e-8)
    x2 = (x2 - x2.mean(0)) / (x2.std(0) + 1e-8)

    return {
        "x1": x1.astype(np.float32),
        "x2": x2.astype(np.float32),
        "time": observed_time.astype(np.float32),
        "event": event,
        "subtype": subtype.astype(np.int64),
    }

def train_test_split_data(data, test_size=0.2, seed=42):
    idx = np.arange(len(data["event"]))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        stratify=data["subtype"],
    )
    def subset(indices):
        return {k: v[indices] for k, v in data.items()}
    return subset(train_idx), subset(test_idx)
