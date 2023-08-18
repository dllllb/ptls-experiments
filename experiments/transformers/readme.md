# Transformers

scenario_age_pred - small dataset in this repo. 
It has been shown that transformers can be used even with small datasets. Studies have been conducted with various parameters, such as split count, loss, lr_scheduler, regularizators.

# Results

The best results on age_pred was observed when using SoftMaxLoss, split_count = 7, and using warmup. It is worth noting that the transformer without tuning hyperparameters shows significantly worse results than the baseline.The main parameters that give a significant increase in quality are split_count and loss.

|best_rnn_params   |best_transformer_params|
|------------------|-----------------------|
|0.643 ± 0.004     |0.646 ± 0.003          |

Using warmap allowed to increase accuracy by 0.03.

# Using warmup

To use warmup you should use coles_module_warmup instead of coles_module