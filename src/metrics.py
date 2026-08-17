
import numpy as np
from sklearn.metrics import adjusted_rand_score

def concordance_index(time, event, risk):
    """Harrell-style concordance index for right-censored data."""
    time = np.asarray(time)
    event = np.asarray(event)
    risk = np.asarray(risk)

    concordant = 0.0
    permissible = 0.0
    ties = 0.0

    for i in range(len(time)):
        if event[i] != 1:
            continue
        for j in range(len(time)):
            if time[i] >= time[j]:
                continue
            permissible += 1
            if risk[i] > risk[j]:
                concordant += 1
            elif risk[i] == risk[j]:
                ties += 1

    return (concordant + 0.5 * ties) / permissible if permissible else np.nan

def cluster_ari(true_subtype, predicted_cluster):
    return adjusted_rand_score(true_subtype, predicted_cluster)

def risk_separation(time, event, cluster, risk):
    """Difference between mean predicted risk in learned clusters."""
    vals = []
    for c in np.unique(cluster):
        vals.append(np.mean(risk[cluster == c]))
    return float(max(vals) - min(vals))
