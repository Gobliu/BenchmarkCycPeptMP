# DMPNN Pretraining Ablation (2026-04-08)

## Setup

- Model: DMPNN (DeepChem 2.7.1)
- Dataset: CycPeptMPDB_4D.csv (5160 molecules, Normalized_PAMPA clipped at -1.0)
- Splits: split_0 through split_9 (random, 0-indexed)
- Training: 2000 epochs max, patience 200, batch_size 64, CPU only (PyTorch 2.0.1+cu118)
- Pretraining dataset: Delaney solubility (regression mode)
- With pretrain: 9 splits (split_0 to split_8), trained with seed 1-9
- Without pretrain: 10 splits (split_0 to split_9), pretrain_epoch=0

## Per-Split Results (normalized scale, test set)

| Split | MAE (w/) | MAE (w/o) | RMSE (w/) | RMSE (w/o) | R² (w/) | R² (w/o) | r (w/) | r (w/o) |
|-------|----------|-----------|-----------|------------|---------|----------|--------|---------|
| 0 | 0.1739 | 0.1615 | 0.2592 | 0.2515 | 0.5508 | 0.5771 | 0.7503 | 0.7637 |
| 1 | 0.1746 | 0.1796 | 0.2581 | 0.2721 | 0.5753 | 0.5281 | 0.7604 | 0.7443 |
| 2 | 0.1756 | 0.1825 | 0.2694 | 0.2846 | 0.5371 | 0.4835 | 0.7374 | 0.7152 |
| 3 | 0.1693 | 0.1901 | 0.2591 | 0.2898 | 0.5190 | 0.3983 | 0.7239 | 0.6848 |
| 4 | 0.1751 | 0.1788 | 0.2616 | 0.2702 | 0.5724 | 0.5437 | 0.7573 | 0.7456 |
| 5 | 0.1706 | 0.1817 | 0.2610 | 0.2795 | 0.5537 | 0.4880 | 0.7476 | 0.7167 |
| 6 | 0.1697 | 0.1818 | 0.2578 | 0.2733 | 0.5254 | 0.4665 | 0.7320 | 0.7081 |
| 7 | 0.1783 | 0.1864 | 0.2648 | 0.2782 | 0.5478 | 0.5010 | 0.7459 | 0.7280 |
| 8 | 0.1873 | 0.1934 | 0.2791 | 0.3086 | 0.5149 | 0.4070 | 0.7210 | 0.6661 |
| 9 | — | 0.1739 | — | 0.2604 | — | 0.5228 | — | 0.7350 |

## Summary (split_0 to split_8, n=9)

| Setting | MAE | RMSE | R² | Pearson r |
|---------|-----|------|----|-----------|
| With pretrain (Delaney) | 0.1749±0.0055 | 0.2633±0.0070 | 0.5440±0.0218 | 0.7417±0.0140 |
| Without pretrain | 0.1817±0.0090 | 0.2786±0.0156 | 0.4881±0.0591 | 0.7192±0.0307 |

## Conclusion

Pretraining on Delaney improves DMPNN performance consistently across splits:
- RMSE improves by ~0.015 (0.2633 vs 0.2786)
- R² improves by ~0.056 (0.5440 vs 0.4881)
- Pearson r improves by ~0.023 (0.7417 vs 0.7192)
- Variance is also lower with pretraining (more stable across splits)

Pretrain wins on 8 out of 9 shared splits (all except split 0). The benefit is
most pronounced on harder splits (e.g. split 3: R² 0.52 vs 0.40, split 8: R² 0.51
vs 0.41).

## Prediction files

- With pretrain: `EnsemFormer/experiments/dmpnn/DMPNN_seed{1-9}.csv` (seed N = split_{N-1})
- Without pretrain: `CSV/Predictions_4D/no_pretrain/random/regression/DMPNN_seed{0-9}.csv`
