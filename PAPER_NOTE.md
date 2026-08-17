# Paper note

The implementation is based on the methodology described in:

Wen, Z., Hou, B., He, W., Yang, S., Moore, J. H., Saykin, A. J.,
Huang, H., Thompson, P. M., Ritchie, M. D., Davatzikos, C., & Shen, L.

**Multi-Modal Deep Clustering Survival Machines for Alzheimer’s Disease Subtype Discovery.**

IEEE International Conference on Computer Vision Workshops (2025), 2285–2293.
DOI: 10.1109/ICCVW69036.2025.00239.

The paper describes modality-specific MLP encoders, additive multimodal fusion,
a two-expert Weibull mixture, survival likelihood terms for uncensored/censored
observations, and subtype assignment using mixture weights.

This repository implements those core ideas independently and uses synthetic data.
It does not reproduce the paper's clinical dataset or reported numbers.
