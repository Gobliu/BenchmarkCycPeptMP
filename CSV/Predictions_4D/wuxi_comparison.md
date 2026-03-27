# WuXi Cross-Assay Comparison (2026-03-27)

## Purpose

31 cyclic peptides were measured by both the original PAMPA assay and the WuXi PAMPA assay.
The goal is to detect potential measurement errors by comparing the two assays against
independent model predictions. When multiple models agree with one assay but not the other,
the disagreeing assay value is likely erroneous or at detection limit.

## Models

All models were trained on the **wuxi split** of `Random_Split_4D.csv` (4102 train, 512 val,
546 test). All 31 WuXi molecules are in the test set — none were seen during training.
Models were trained to predict the original PAMPA value (not WuXi).

| Model | Type | Input | Training details |
|-------|------|-------|-----------------|
| DMPNN | Directed Message Passing Neural Network | 2D molecular graph (SMILES) | DeepChem 2.7.1, pre-trained on Delaney, 2000 epochs, patience 200, CPU only (PyTorch 2.0.1) |
| EGNN | Equivariant Graph Neural Network (meanpool, no bond) | 3D coordinates + atom features | hidden_nf=128, n_layers=5, attention=True, bs=8, lr=6e-4, 200 epochs, patience 100 |
| CPMP | Communicative Message Passing | 3D coordinates + adjacency/distance | d_model=256, N=6, h=8, lr=1.15e-4, 200 epochs, patience 30 |

## Comparison Table

All values in raw PAMPA log scale, clipped at -8.

| ID | PAMPA | WuXi | DMPNN | EGNN | CPMP |
|---|---|---|---|---|---|
| 1060 | -7.00 | -7.95 | -5.57 | -5.82 | -5.69 |
| 1073 | -8.00 | -8.00 | -6.55 | -6.05 | -6.95 |
| 1074 | -8.00 | -8.00 | -6.67 | -6.04 | -7.11 |
| 1090 | -8.00 | -8.00 | -6.29 | -6.23 | -7.12 |
| 1093 | -8.00 | -7.91 | -8.00 | -4.17 | -8.00 |
| 1097 | -8.00 | -8.00 | -7.98 | -7.27 | -7.58 |
| 1104 | -8.00 | -8.00 | -6.89 | -5.82 | -6.23 |
| 1392 | -8.00 | -8.00 | -7.57 | -7.84 | -7.64 |
| 1521 | -8.00 | -7.99 | -7.02 | -7.26 | -6.75 |
| 1526 | -7.34 | -8.00 | -7.41 | -6.85 | -7.48 |
| 1568 | -5.41 | -6.09 | -5.41 | -6.19 | -5.69 |
| 1622 | -5.95 | -6.39 | -5.99 | -6.73 | -7.34 |
| 1658 | -5.80 | -6.05 | -6.31 | -6.90 | -7.33 |
| 1729 | -8.00 | -7.18 | -6.94 | -6.68 | -6.83 |
| 1801 | -8.00 | -7.03 | -6.13 | -6.50 | -6.85 |
| 1820 | -8.00 | -6.49 | -5.44 | -5.59 | -5.69 |
| 2303 | -5.95 | -7.06 | -5.44 | -5.36 | -6.20 |
| 2304 | -5.62 | -6.20 | -5.53 | -5.18 | -5.80 |
| 2305 | -5.21 | -5.73 | -5.33 | -4.94 | -5.47 |
| 2333 | -7.13 | -8.00 | -6.95 | -7.01 | -6.43 |
| 2334 | -6.14 | -8.00 | -6.96 | -6.40 | -7.26 |
| 2560 | -5.24 | -5.83 | -5.25 | -5.55 | -5.25 |
| 2646 | -5.29 | -6.54 | -5.38 | -5.66 | -5.25 |
| 3034 | -6.67 | -7.37 | -6.05 | -5.79 | -5.91 |
| 3273 | -4.00 | -5.13 | -4.85 | -4.82 | -4.86 |
| 3278 | -4.00 | -5.21 | -4.84 | -4.79 | -4.85 |
| 3299 | -4.00 | -4.95 | -5.13 | -5.34 | -5.18 |
| 3423 | -6.18 | -6.24 | -5.43 | -5.23 | -5.43 |
| 4215 | -8.00 | -8.00 | -6.02 | -5.83 | -5.76 |
| 5556 | -8.00 | -5.75 | -5.46 | -5.37 | -5.30 |
| 5557 | -8.00 | -7.08 | -5.98 | -5.96 | -5.79 |

## Suspected Measurement Errors

### Original PAMPA likely wrong (all models + WuXi disagree with PAMPA)

These molecules have PAMPA = -8 (originally -10, at detection limit), but WuXi and all
three models converge on a much higher value. The original PAMPA reading is likely a
detection-limit artifact.

| ID | PAMPA | WuXi | Model consensus | Gap |
|---|---|---|---|---|
| 5556 | -8.00 | -5.75 | ~-5.4 | ~2.5 |
| 1820 | -8.00 | -6.49 | ~-5.6 | ~2.4 |
| 5557 | -8.00 | -7.08 | ~-5.9 | ~2.1 |
| 1801 | -8.00 | -7.03 | ~-6.5 | ~1.5 |
| 1729 | -8.00 | -7.18 | ~-6.8 | ~1.2 |
| 4215 | -8.00 | -8.00 | ~-5.9 | ~2.1 (WuXi may also be at detection limit) |

### WuXi possibly at detection limit (models agree with PAMPA, not WuXi)

| ID | PAMPA | WuXi | Model consensus | Note |
|---|---|---|---|---|
| 2334 | -6.14 | -8.00 | ~-6.9 | Models closer to PAMPA; WuXi = -8 looks censored |
| 2333 | -7.13 | -8.00 | ~-6.8 | Similar pattern |

## Next Steps

- Add predictions from more models (EnsemFormer ensemble, etc.) to strengthen consensus
- If 5+ models agree with WuXi over PAMPA, flag those PAMPA values for correction
- Consider re-running affected molecules experimentally
