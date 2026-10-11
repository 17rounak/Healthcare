# Before/after comparison: the two implementation fixes

Generated from two fully executed 300-run grids (identical dataset, seeds, splits, hyperparameters, rounds, local epochs and evaluation protocol).

* **Before** = `results_before_fixes/reproduction_check.csv` (grid executed with the MLP double-normalization gradient bug and the sigma*C/K DP noise convention; 39/44 checks within tolerance).
* **After** = `results/reproduction_check.csv` (grid executed after (1) the MLP first-layer gradient fix and (2) the DP noise-convention fix (noise std sigma*C on the released aggregate); 35/44 checks within tolerance).

Tolerance for every check: |reproduced - paper| <= tol * paper, with the per-metric tol configured in `src/analysis.py::reproduction_checks`.

## 1. The five previously failing checks

| metric | paper | before | after | abs diff before | abs diff after | status before -> after |
|---|---|---|---|---|---|---|
| federation_gain_mlp_iid | 0.0761 | 0.0226 | 0.0686 | 0.0535 | 0.0075 | False -> True |
| local_only_mlp_iid_auc | 0.6328 | 0.6829 | 0.6403 | 0.0535 | 0.0075 | False -> True |
| local_only_mlp_covariate_skew_auc | 0.5966 | 0.6259 | 0.6095 | 0.0348 | 0.0129 | False -> True |
| dp_util_logistic_sigma2.0 | 0.6032 | 0.6404 | 0.5860 | 0.0372 | 0.0172 | False -> True |
| dp_util_mlp_sigma2.0 | 0.5532 | 0.6003 | 0.5098 | 0.0471 | 0.0434 | False -> False |

## 2. DP-utility checks that moved out of tolerance (sigma <= 1)

The corrected convention injects K=5x more noise at every sigma; every DP utility drops, which is the mathematically forced direction.

| metric | paper | before | after | rel diff before | rel diff after | status before -> after |
|---|---|---|---|---|---|---|
| dp_util_logistic_sigma0.1 | 0.7288 | 0.7166 | 0.6833 | 0.0167 | 0.0624 | True -> False |
| dp_util_mlp_sigma0.1 | 0.7193 | 0.6993 | 0.6579 | 0.0279 | 0.0854 | True -> False |
| dp_util_logistic_sigma0.25 | 0.7208 | 0.7179 | 0.6571 | 0.0040 | 0.0883 | True -> False |
| dp_util_mlp_sigma0.25 | 0.7082 | 0.6984 | 0.5528 | 0.0138 | 0.2195 | True -> False |
| dp_util_logistic_sigma0.5 | 0.6979 | 0.6927 | 0.5974 | 0.0075 | 0.1440 | True -> False |
| dp_util_mlp_sigma0.5 | 0.6651 | 0.6536 | 0.5267 | 0.0172 | 0.2082 | True -> False |
| dp_util_logistic_sigma1.0 | 0.6543 | 0.6483 | 0.5659 | 0.0092 | 0.1352 | True -> False |
| dp_util_mlp_sigma1.0 | 0.5991 | 0.5914 | 0.5516 | 0.0128 | 0.0793 | True -> False |

## 3. All 44 checks, before vs after

| metric | paper | before | after | within tolerance before | after |
|---|---|---|---|---|---|
| dataset_n_rows | 4240 | 4240.0000 | 4240.0000 | True | True |
| n_features | 15 | 15.0000 | 15.0000 | True | True |
| positive_prevalence | 0.152 | 0.1519 | 0.1519 | True | True |
| iid_mean_silo_size_seed0 | 678.4 | 678.4000 | 678.4000 | True | True |
| covariate_min_silo_prevalence_seed0 | 0.056 | 0.0486 | 0.0486 | True | True |
| covariate_max_silo_prevalence_seed0 | 0.271 | 0.2670 | 0.2670 | True | True |
| label_skew_prevalence_range_seed0 | 0.778 | 0.6543 | 0.6543 | True | True |
| centralized_logistic_auc | 0.729 | 0.7182 | 0.7182 | True | True |
| centralized_mlp_auc | 0.7044 | 0.7079 | 0.6973 | True | True |
| fedavg_logistic_iid_auc | 0.7315 | 0.7201 | 0.7201 | True | True |
| fedavg_logistic_label_skew_auc | 0.7313 | 0.7206 | 0.7206 | True | True |
| fedavg_logistic_covariate_skew_auc | 0.7321 | 0.7189 | 0.7189 | True | True |
| fedavg_mlp_iid_auc | 0.7232 | 0.7055 | 0.7089 | True | True |
| fedavg_mlp_label_skew_auc | 0.7078 | 0.6986 | 0.6903 | True | True |
| fedavg_mlp_covariate_skew_auc | 0.7281 | 0.7022 | 0.7076 | True | True |
| federation_gain_logistic_iid | 0.0226 | 0.0198 | 0.0198 | True | True |
| federation_gain_logistic_label_skew | 0.037 | 0.0462 | 0.0462 | True | True |
| federation_gain_logistic_covariate_skew | 0.038 | 0.0453 | 0.0453 | True | True |
| federation_gain_mlp_iid | 0.0761 | 0.0226 | 0.0686 | False | True |
| federation_gain_mlp_label_skew | 0.0611 | 0.0426 | 0.0576 | True | True |
| federation_gain_mlp_covariate_skew | 0.111 | 0.0762 | 0.0981 | True | True |
| local_only_logistic_iid_auc | 0.6975 | 0.7003 | 0.7003 | True | True |
| local_only_logistic_label_skew_auc | 0.6836 | 0.6743 | 0.6743 | True | True |
| local_only_logistic_covariate_skew_auc | 0.6809 | 0.6736 | 0.6736 | True | True |
| local_only_mlp_iid_auc | 0.6328 | 0.6829 | 0.6403 | False | True |
| local_only_mlp_label_skew_auc | 0.6292 | 0.6560 | 0.6326 | True | True |
| local_only_mlp_covariate_skew_auc | 0.5966 | 0.6259 | 0.6095 | False | True |
| dp_epsilon_sigma0.1 | 2315 | 2303.4856 | 2303.4856 | True | True |
| dp_epsilon_sigma0.25 | 441.6 | 441.3943 | 441.3943 | True | True |
| dp_epsilon_sigma0.5 | 140.8 | 140.6971 | 140.6971 | True | True |
| dp_epsilon_sigma1.0 | 50.4 | 50.3485 | 50.3485 | True | True |
| dp_epsilon_sigma2.0 | 20.2 | 20.1743 | 20.1743 | True | True |
| dp_util_logistic_sigma0.1 | 0.7288 | 0.7166 | 0.6833 | True | False |
| dp_util_mlp_sigma0.1 | 0.7193 | 0.6993 | 0.6579 | True | False |
| dp_util_logistic_sigma0.25 | 0.7208 | 0.7179 | 0.6571 | True | False |
| dp_util_mlp_sigma0.25 | 0.7082 | 0.6984 | 0.5528 | True | False |
| dp_util_logistic_sigma0.5 | 0.6979 | 0.6927 | 0.5974 | True | False |
| dp_util_mlp_sigma0.5 | 0.6651 | 0.6536 | 0.5267 | True | False |
| dp_util_logistic_sigma1.0 | 0.6543 | 0.6483 | 0.5659 | True | False |
| dp_util_mlp_sigma1.0 | 0.5991 | 0.5914 | 0.5516 | True | False |
| dp_util_logistic_sigma2.0 | 0.6032 | 0.6404 | 0.5860 | False | True |
| dp_util_mlp_sigma2.0 | 0.5532 | 0.6003 | 0.5098 | False | False |
| comm_cost_logistic_mb | 0.026 | 0.0256 | 0.0256 | True | True |
| comm_cost_mlp_mb | 0.437 | 0.4368 | 0.4368 | True | True |

## 4. Key non-DP MLP values (seed means)

| quantity | paper (approx) | before | after |
|---|---|---|---|
| IID FedAvg MLP | 0.7232 | 0.7055 | 0.7089 |
| IID local-only MLP | 0.6294 (back-calc) | 0.6829 | 0.6403 |
| IID federation gain MLP | 0.0761 | 0.0226 | 0.0686 |
| label-skew FedAvg MLP | 0.7078 | 0.6986 | 0.6903 |
| covariate-skew FedAvg MLP | 0.7281 | 0.7022 | 0.7076 |
| covariate-skew local-only MLP | 0.5912 (back-calc) | 0.6259 | 0.6095 |
| IID centralized MLP | 0.7044 | 0.7093 | 0.7010 |
| IID FedAvg logistic (reference) | 0.7315 | 0.7201 | 0.7201 |

## 5. Interpretation

* MLP fix: the first layer previously received exactly 1/batch_size of its intended gradient (1/64 for full batches). After the fix the MLP is genuinely worse than logistic regression when trained on isolated skewed silos (local-only covariate-skew 0.6259 -> 0.6095, paper ~0.5912) while federated training benefits much more (IID federation gain 0.0226 -> 0.0686, paper ~0.0761). Both reproduce the paper's qualitative claims, which the buggy gradient could not.
* DP convention fix: noise std on the released aggregate went from sigma*C/K (= 0.2*sigma*C) to sigma*C. Every DP utility must and did drop. dp_util_logistic_sigma2.0 now matches the paper (0.5860 vs 0.6032, 2.8%); dp_util_mlp_sigma2.0 improved 0.6003 -> 0.5098 vs paper 0.5532 (7.8%). At sigma <= 1 the corrected mechanism is noisier than whatever the paper actually ran: the paper's low-sigma utilities coincide with the OLD sigma*C/K scaling plus the ~0.011-0.013 logistic evaluation-convention gap seen throughout the study, while its sigma=2 utility coincides with the corrected sigma*C scaling. No single noise-multiplier convention calibrated to C reproduces the paper's entire DP curve; the corrected implementation realizes exactly the privacy that the paper's own RDP formula claims, in every regime, and reports the resulting utilities without adjustment.

