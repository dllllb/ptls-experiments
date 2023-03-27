# Coles with aggregate feaures concatenation 

In this experiment coles was concatenated with aggregate features on public datasets. Also coles was trained with decorrelation loss (VICReg Loss). Experiments were made on rosbank, gender and age_prediction datasets. 


Experiments were made with default hyperparameters (std_coeff = 0.5, cov_coeff = 0.5, agg_coeff = 0.5, coles_coeff = 0.5). Results on test data: 


## AUROC 

| method/dataset                             | gender         | rosbank         | 
| ------------------------------------------ | -------------- | --------------- | 
|  aggregate features                        |  0.877 ± 0.003 |  0.827 ± 0.004  |
|  coles embeddings                          |  0.879 ± 0.004 |  0.840 ± 0.003  |
|  coles+decorr_loss embeddings              |  0.884 ± 0.003 |  0.844 ± 0.002  | 
|  coles agg_features concat                 |  0.885 ± 0.003 |  0.841 ± 0.003  | 
|  coles+decorr_loss agg_features concat     |  0.887 ± 0.003 |  0.843 ± 0.004  | 

## Accuracy 

| method/dataset                             | gender         | rosbank         | age prediction | 
| ------------------------------------------ | -------------- | --------------- | -------------- |
|  aggregate features                        |  0.793 ± 0.004 |  0.749 ± 0.003  | 0.627 ± 0.005  |
|  coles embeddings                          |  0.796 ± 0.007 |  0.759 ± 0.006  | 0.642 ± 0.001  |
|  coles+decorr_loss embeddings              |  0.797 ± 0.004 |  0.765 ± 0.010  | 0.640 ± 0.005  |
|  coles agg_features concat                 |  0.798 ± 0.005 |  0.765 ± 0.012  | 0.651 ± 0.003  |
|  coles+decorr_loss agg_features concat     |  0.800 ± 0.005 |  0.762 ± 0.007  | 0.645 ± 0.003  |


## Weight selection 

This module has two parameters (weights): agg_coeff and coles_coeff. Results with different values of coefficients for rosbank dataset: 

## AUROC 

|agg_coeff/coles_coeff| coles+decorr_loss embeddings| coles+decorr_loss agg_features concat| 
| ------------------- | --------------------------- | ------------------------------------ | 
|  0.5/0.5            |  0.844 ± 0.002              |  0.843 ± 0.004                       |
|  0.15/0.85          |  0.842 ± 0.005              |  0.837 ± 0.004                       | 
|  0.75/0.25          |  0.846 ± 0.003              |  0.846 ± 0.003                       | 
|  0.65/0.35          |  0.843 ± 0.004              |  0.839 ± 0.004                       | 
|  0.35/0.65          |  0.851 ± 0.002              |  0.850 ± 0.000                       | 
|  0.75/0.25          |  0.846 ± 0.003              |  0.847 ± 0.005                       |
|  0.85/0.15          |  0.839 ± 0.005              |  0.839 ± 0.004                       | 

## Accuracy 

|agg_coeff/coles_coeff| coles+decorr_loss embeddings| coles+decorr_loss agg_features concat| 
| ------------------- | --------------------------- | ------------------------------------ | 
|  0.5/0.5            |  0.765 ± 0.010              |  0.762 ± 0.007                       |
|  0.15/0.85          |  0.764 ± 0.004              |  0.758 ± 0.007                       | 
|  0.75/0.25          |  0.768 ± 0.009              |  0.764 ± 0.010                       | 
|  0.65/0.35          |  0.760 ± 0.006              |  0.755 ± 0.013                       | 
|  0.35/0.65          |  0.770 ± 0.008              |  0.770 ± 0.008                       | 
|  0.75/0.25          |  0.774 ± 0.005              |  0.764 ± 0.003                       |
|  0.85/0.15          |  0.755 ± 0.007              |  0.750 ± 0.007                       | 