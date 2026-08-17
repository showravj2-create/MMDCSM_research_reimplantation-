# Multi-Modal Deep Clustering Survival Machines — Research Reimplementation

A research-oriented PyTorch implementation inspired by:

> **Multi-Modal Deep Clustering Survival Machines for Alzheimer’s Disease Subtype Discovery**  
> Wen et al., IEEE International Conference on Computer Vision Workshops, 2025.  
> DOI: `10.1109/ICCVW69036.2025.00239`

This repository is **not the authors' official implementation** and does not claim to reproduce their reported clinical results. It is an independent methodological reimplementation of the model described in the paper, using a fully reproducible synthetic multimodal survival dataset.

## Why this project?

The paper combines three ideas:

1. **Multimodal representation learning** — each modality is encoded separately.
2. **Survival modeling** — time-to-event outcomes are modeled with Weibull expert distributions.
3. **Joint subtype discovery** — mixture weights define latent patient subtypes.

The implementation in this repository makes those ideas explicit and experimentally testable.

## Research questions

- Can modality-specific neural encoders recover a shared latent representation?
- Can a two-expert Weibull mixture learn distinct low- and high-risk subgroups?
- Does multimodal fusion outperform single-modality models on synthetic survival data?
- How well do the learned clusters recover known latent risk groups?
- How do censoring and modality informativeness affect performance?

## Model

For two modalities \(x\) and \(y\):

\[
z = \phi_x(x) + \phi_y(y)
\]

The fused representation is mapped to two mixture weights:

\[
\alpha = \operatorname{softmax}(Wz+b)
\]

Each expert is a Weibull distribution. The patient-specific survival density is:

\[
p(T|x,y)=\sum_{k=1}^{K}\alpha_k p_k(T)
\]

with \(K=2\) in the experiments.

The implementation supports:

- uncensored likelihood terms using the Weibull PDF
- right-censored likelihood terms using the Weibull survival function
- prior-matching regularization for expert parameters
- latent cluster assignment from the largest mixture weight

## Repository structure

```text
MMDCSM_research_reimplementation/
├── configs/
│   └── experiment.json
├── data/
│   └── README.md
├── notebooks/
│   └── mmdcsm_experiment.ipynb
├── results/
│   ├── figures/
│   └── tables/
├── src/
│   ├── data.py
│   ├── metrics.py
│   ├── model.py
│   ├── train.py
│   └── experiment.py
├── tests/
│   └── test_model.py
├── requirements.txt
└── README.md
```

## Reproducibility

This repository intentionally uses synthetic data because the paper's clinical data are not included here.

The synthetic generator creates two modalities, latent risk subtypes, survival times, and right-censoring. This lets the complete pipeline run without restricted clinical data.

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
python src/experiment.py
```

or open:

```text
notebooks/mmdcsm_experiment.ipynb
```

## Evaluation

The experiment reports:

- concordance index
- negative log-likelihood
- cluster Adjusted Rand Index
- cluster-risk separation
- censoring proportion

Figures include:

- training loss
- latent embedding
- cluster distribution
- Kaplan–Meier curves by learned cluster
- predicted risk distribution

## Important scientific limitation

The synthetic experiment validates the **implementation and experimental design**, not the Alzheimer's disease findings reported by Wen et al.

A genuine clinical reproduction would require the same cohort definitions, preprocessing, MRI/PET features, follow-up times, censoring rules, and evaluation protocol used in the original study.

## Paper

Wen et al. describe MMDCSM as a multimodal framework combining modality-specific MLP encoders with a mixture of Weibull survival experts to jointly discover MCI subtypes and estimate individualized conversion risk.

See the paper supplied with this project for the full mathematical formulation and reported experiments.

## Next research extensions

1. Replace synthetic data with an appropriate public survival dataset.
2. Add a Cox proportional-hazards baseline.
3. Add DeepSurv/Deep Survival Machines baselines.
4. Compare early, late, and learned multimodal fusion.
5. Run sensitivity analysis over censoring rates.
6. Add repeated trials and confidence intervals.
7. Investigate modality ablation and missing-modality robustness.
