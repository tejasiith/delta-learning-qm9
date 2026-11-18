| Model | Test MAE | Test RMSE | Train MAE |
|---|---|---|---|
| Training-set mean | 336.85 ± 21.90 | 433.56 ± 23.37 | 357.09 |
| Linear, atom counts | 14.18 ± 1.61 | 17.76 ± 1.54 | 14.72 |
| Linear + random forest (structure) | 8.23 ± 1.17 | 10.56 ± 1.03 | 2.41 |
| Linear + MLP (structure) | 7.68 ± 0.87 | 9.86 ± 0.83 | 0.91 |
| Linear + MLP (structure + HF) | 7.28 ± 0.74 | 9.32 ± 0.70 | 0.78 |

Paired differences in test MAE (first minus second, negative = first is better):

- Linear + MLP (structure) minus Linear + random forest (structure): mean -0.55, first better in 5 of 5 seeds
- Linear + MLP (structure + HF) minus Linear + MLP (structure): mean -0.40, first better in 5 of 5 seeds
- Linear + MLP (structure + HF) minus Linear, atom counts: mean -6.90, first better in 5 of 5 seeds
