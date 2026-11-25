# Electrocatalysis AI/ML

Applying machine learning to catalyst design, following the multi-scale framework in Wang et al. (2025), *Energy and AI*.

| Notebook | What it does | Result |
|---|---|---|
| `GNN_QM9.ipynb` | GNN predicting HOMO–LUMO gap (QM9) | MAE 0.48 eV (baseline 1.07) |
| `porosity_cnn.ipynb` | ResNet-18 predicting porosity from SEM tiles | R² 0.87 (threshold baseline 0.42) |
| `BO_GNN.ipynb` | Bayesian optimisation using GNN features to find low-gap molecules | Top-1% molecule in ~29 tests vs ~80 for random search |

**Data:** QM9 downloads automatically. SEM images are from the NFFA-Europe dataset (not included here due to size).
