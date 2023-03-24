# Coles with arrange feaures concatenation 

In this experiment coles was concatenated with aggregate features on public datasets. Also coles was trained with decorrelation loss (VICReg Loss). Experiments were made on rosbank, gender and age_prediction datasets. 


Experiments were made with default hyperparameters (std_coeff = 0.5, cov_coeff = 0.5, agg_coeff = 0.5, coles_coeff = 0.5). Results on test data: 


## AUROC 

| method/dataset                             | gender         | rosbank         | 
| ------------------------------------------ | -------------- | --------------- | 
|  aggregate features                        |  0.877 ± 0.003 |  0.827 ± 0.004  |
|  coles embeddings                          |  0.880 ± 0.004 |  0.840 ± 0.003  |
|  coles+decorr_loss embeddings              |  0.886 ± 0.003 |  0.840 ± 0.002  | 
|  coles agg_features concat                 |  0.884 ± 0.002 |  0.847 ± 0.005  | 
|  coles+decorr_loss agg_features concat     |  0.890 ± 0.003 |  0.838 ± 0.002  | 

## Accuracy 

| method/dataset                             | gender         | rosbank         | age prediction | 
| ------------------------------------------ | -------------- | --------------- | -------------- |
|  aggregate features                        |  0.793 ± 0.004 |  0.749 ± 0.003  | 0.627 ± 0.005  |
|  coles embeddings                          |  0.796 ± 0.007 |  0.759 ± 0.006  | 0.642 ± 0.001  |
|  coles+decorr_loss embeddings              |  0.798 ± 0.004 |  0.758 ± 0.010  | 0.640 ± 0.005  |
|  coles agg_features concat                 |  0.795 ± 0.005 |  0.765 ± 0.012  | 0.651 ± 0.003  |
|  coles+decorr_loss agg_features concat     |  0.795 ± 0.006 |  0.756 ± 0.007  | 0.645 ± 0.003  |