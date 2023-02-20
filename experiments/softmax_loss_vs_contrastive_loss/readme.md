# SoftmaxLoss 

Faster training (up to 25% faster) with the same GPU RAM consumption. 
Can be trained  with high batch sizes. Better result at large dataser.
Need more epochs achieve best results (more than contrastive loss). 

# Results 

Both losses were tested on public datasets, here's results comparsion.

## Alpha_battle results (auroc)

On best validation epoch:

| loss             | hidden 1024            | hidden 2048                | hidden size x2 improvement |
| ---------------- | ---------------------- | -------------------------- | ------------------ |
| contrastive      | 0.7933 (on 25th epoch) | 0.7938 (on 23th epoch)     | 0.0005 improvement |
| softmax          | 0.7955 (on  9th epoch) | **0.7986** (on 10th epoch) | **0.0031** improvement |

On fixed epoch:

| loss             | epoch | hidden 1024 | hidden 2048 |
| ---------------- | ----- | ----------- | ----------- |
| contrastive      |  10   | 0.7925      | 0.7928      |
| contrastive      |  20   | 0.7933      | 0.7938      |
| softmax          |  10   | 0.7955      | 0.7986      |
| softmax          |  20   | 0.7953      | 0.7995      |


Softmax loss advantages:
- better results on downstream task
- best result acheaved 2 times faster by epoch count and up to 25% faster by epoch time 


## Best results on other datasets
The best results on test data for different datasets. 

#### AUCROC  

| loss\dataset      | gender       | age_pred  | rosbank       |
| ---               | ---          | ---       | ---           |
| softmax_pairwise  | 0.875 ± 0.02 | -         | 0.825 ± 0.005 |   
|  softmax          | 0.876 ± 0.03 | -         | 0.844 ± 0.003 |   
|  contrastive      | 0.882 ± 0.03 | -         | 0.841 ± 0.004 |

#### Accuracy 

| loss\dataset     | gender        | age_pred      | rosbank       |
| ---              | ---           | ---           | ---           |
| softmax_pairwise | 0.796 ± 0.08  | 0.636 ± 0.03  | 0.750 ± 0.008 |   
| softmax          | 0.793 ± 0.04  | 0.632 ± 0.006 | 0.763 ± 0.004 |   
| contrastive      | 0.798 ± 0.007 | 0.643 ± 0.003 | 0.766 ± 0.013 |


# Parameters tuning
## Best parameters 

#### SoftmaxLoss: 
##### temperature: 0.05 


## Temperature comparsion

Softmaxloss was tested on gender and rosbank datasets with different values of temperature: 4.0, 1.0, 0.2, 0.05 and 0.01 (split_count parameter equals 5). 

#### AUCROC  

| temperature\dataset | gender        | rosbank        |
| ---                 | ---           |  ---           |
| 4.0                 | 0.839 ± 0.004 |  0.834 ± 0.04  |   
| 1.0                 | 0.863 ± 0.004 |  0.838 ± 0.03  |   
| 0.2                 | 0.866 ± 0.003 |  0.840 ± 0.04  |
| 0.05                | 0.874 ± 0.004 |  0.844 ± 0.04  |
| 0.01                | 0.870 ± 0.03  |  0.843 ± 0.04  |

#### Accuracy 

| temperature\dataset | gender       | rosbank        |
| ---                 | ---          |  ---           |
| 4.0                 | 0.763 ± 0.05 |  0.745 ± 0.010 |   
| 1.0                 | 0.784 ± 0.05 |  0.751 ± 0.05  |   
| 0.2                 | 0.776 ± 0.05 |  0.753 ± 0.006 |
| 0.05                | 0.793 ± 0.04 |  0.763 ± 0.004 |
| 0.01                | 0.784 ± 0.04 |  0.759 ± 0.010 |


## Batch size comparsion 
Batch size comparsion experiments were done on rosbank dataset (split_count parameter equals 5 and temperature equals 0.05).

#### AUCROC

CUDA OOM was at old SoftmaxLoss realisation

| loss\batch_size   |  64           | 128           | 256           | 512            | 1024          |  150           | 96            |
| ---               | ---           | ---           | ---           | ---            | ---           | ---            | ---           |
| softmax_pairwise  | 0.833 ± 0.04  | 0.822 ± 0.004 | CUDA OOM      | CUDA OOM       | CUDA OOM      | 0.825 ± 0.005  | 0.823 ± 0.004 |  
| softmax           | 0.839 ± 0.04  | 0.846 ± 0.04  | 0.844 ± 0.003 | 0.844 ± 0.004  | 0.840 ± 0.004 | -              | -             |
| contrastive       | 0.839 ± 0.03  | 0.847 ± 0.02  | 0.841 ± 0.003 | 0.842 ± 0.002  | 0.834 ± 0.001 | -              | -             |


#### Accuracy 

| loss\batch_size   |  64            | 128           | 256           | 512           | 1024          |  150           | 96            |
| ---               | ---            | ---           | ---           | ---           | ---           | ---            | ---           |
| softmax_pairwise  | 0.756 ± 0.050  | 0.748 ± 0.004 | CUDA OOM      | CUDA OOM      | CUDA OOM      | 0.750 ± 0.008  | 0.744 ± 0.015 |  
| softmax           | 0.757 ± 0.010  | 0.768 ± 0.012 | 0.763 ± 0.004 | 0.760 ± 0.010 | 0.766 ± 0.011 | -              | -             |
| contrastive       | 0.750 ± 0.006  | 0.773 ± 0.004 | 0.764 ± 0.003 | 0.763 ± 0.008 | 0.765 ± 0.003 | -              | -             |

## Splits number comparsion 


Tested on gender dataset. Temperature parameter for SoftmaxLoss equals 0.05. 

#### AUCROC

| splits\dataset | softmax       | contrastive    | 
| ---            | ---           | ---            |
| 2              | 0.862 + 0.05  | 0.875 ± 0.002  |   
| 5              | 0.876 ± 0.03  | 0.875 ± 0.002  |      
| 7              | 0.875 ± 0.03  | 0.878 ± 0.004  |   
| 9              | 0.875 ± 0.001 | 0.880 ± 0.002  | 

#### Accuracy



| splits\dataset | softmax       | contrastive   | 
| ---            | ---           | ---           |
| 2              | 0.782+ 0.006  | 0.796 ± 0.008 |   
| 5              | 0.793 ± 0.04  | 0.796 ± 0.004 |      
| 7              | 0.784 ± 0.005 | 0.787 ± 0.005 |   
| 9              | 0.784 ± 0.006 | 0.792 ± 0.004 |

