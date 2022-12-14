# SoftmaxLoss 

Faster training. Can be trained  with high batch sizes. Worse validation results. Need more epochs achieve best results (more than contrastive loss). 

# SoftmaxPairwiseLoss

High cost of time and memory. High batch size causes CUDA out of memory. Better validation results (very close to contrastive loss results), but worse test results.  

# Results 

Both losses were tested on public datasets, here's results comparsion.
## Best results
The best results on test data for different datasets. 

#### AUCROC  

| loss\dataset| gender | age_pred  | rosbank   |
| --- | --- | --- | --- |
| softmax_pairwise  |  0.875 ± 0.02 |  - |  0.825 ± 0.005 |   
|  softmax |  0.876 ± 0.03 |  - |  0.844 ± 0.003 |   
|  contrastive | 0.882 ± 0.03   | -  | 0.841 ± 0.004  |

#### Accuracy 

| loss\dataset| gender | age_pred  | rosbank   |
| --- | --- | --- | --- |
| softmax_pairwise  |  0.796 ± 0.08 |  0.636 ± 0.03 | 0.750 ± 0.008  |   
|  softmax |  0.793 ± 0.04  |  0.632 ± 0.006 |  0.763 ± 0.004 |   
|  contrastive | 0.798 ± 0.007   |  0.643 ± 0.003 |  0.766 ± 0.013 |


# Parameters tuning
## Best parameters 

#### SoftmaxLoss: 
##### temperature: 0.05 
##### eps: 1e-6  


  
#### SoftmaxPairwiseLoss:
##### temperature: 0.05
##### eps: 1e-6

## Temperature comparsion

Softmaxloss was tested on gender and rosbank datasets with different values of temperature: 4.0, 1.0, 0.2, 0.05 and 0.01 (split_count parameter equals 5). 

#### AUCROC  

| temperature\dataset | gender | rosbank   |
| --- | --- |  --- |
| 4.0   | 0.839 ± 0.004  |    0.834 ± 0.04  |   
|  1.0  | 0.863 ± 0.004  |    0.838 ± 0.03  |   
|  0.2 |  0.866 ± 0.003  |   0.840 ± 0.04   |
|  0.05 |  0.874 ± 0.004  |   0.844 ± 0.04   |
|  0.01 |   0.870 ± 0.03 |    0.843 ± 0.04  |

#### Accuracy 

| temperature\dataset | gender | rosbank   |
| --- | --- |  --- |
| 4.0   | 0.763 ± 0.05  |    0.745 ± 0.010  |   
|  1.0  |  0.784 ± 0.05 |    0.751 ± 0.05  |   
|  0.2 |  0.776 ± 0.05  |   0.753 ± 0.006   |
|  0.05 |   0.793 ± 0.04 |  0.763 ± 0.004    |
|  0.01 |   0.784 ± 0.04 |  0.759 ± 0.010    |


## Batch size comparsion 
Batch size comparsion experiments were done on rosbank dataset (split_count parameter equals 5 and temperature equals 0.05).

#### AUCROC

| loss\batch_size |  64  | 128 | 256 | 512  | 1024 |  150  | 96 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| softmax_pairwise  | 0.833 ± 0.04  | 0.822 ± 0.004 | CUDA OUT OF MEMORY | CUDA OUT OF MEMORY  | CUDA OUT OF MEMORY  | 0.825 ± 0.005  | 0.823 ± 0.004 |  
|  softmax | 0.839 ± 0.04  | 0.846 ± 0.04 | 0.844 ± 0.003 | 0.844 ± 0.004  | 0.840 ± 0.004 |  -  | - |
|  contrastive  | 0.839 ± 0.03  | 0.847 ± 0.02 | 0.841 ± 0.003 | 0.842 ± 0.002  | 0.834 ± 0.001 | -  | - |


#### Accuracy 

| loss\batch_size |  64  | 128 | 256 | 512  | 1024 |  150  | 96 |
| --- | --- | --- |--- | --- | --- | --- | --- |
| softmax_pairwise  | 0.756 ± 0.050  | 0.748 ± 0.004  | CUDA OUT OF MEMORY | CUDA OUT OF MEMORY  | CUDA OUT OF MEMORY |  0.750 ± 0.008  | 0.744 ± 0.015 |  
|  softmax |  0.757 ± 0.010  | 0.768 ± 0.012  | 0.763 ± 0.004 | 0.760 ± 0.010 | 0.766 ± 0.011 | -  | - |
|  contrastive |  0.750 ± 0.006  |  0.773 ± 0.004 | 0.764 ± 0.003| 0.763 ± 0.008  | 0.765 ± 0.003 | -  | - |

## Splits number comparsion 


Tested on gender dataset. Temperature parameter for SoftmaxLoss equals 0.05. 

#### AUCROC

| splits\dataset | softmax | contrastive  | 
| --- | --- | --- |
| 2   | 0.862 + 0.05  |  0.875 ± 0.002 |   
|  5  |  0.876 ± 0.03  |  0.875 ± 0.002  |      
|  7  |  0.875 ± 0.03 |   0.878 ± 0.004|   
|  9  | 0.875 ± 0.001  | 0.880 ± 0.002  | 

#### Accuracy



| splits\dataset | softmax | contrastive  | 
| --- | --- | --- |
| 2   |  0.782+ 0.006 |  0.796 ± 0.008 |   
|  5  |  0.793 ± 0.04 |  0.796 ± 0.004 |      
|  7  |  0.784 ± 0.005  | 0.787 ± 0.005  |   
|  9  |  0.784 ± 0.006 |  0.792 ± 0.004|


## Time to train comparsion 

Tested on gender dataset:

| epoch time\loss| softmax_pairwise | softmax_5_splits | softmax_2_splits  | contrastive |
| --- | --- | --- | --- | --- |
| seconds  |   ~60 | ~11  | ~5  |  ~13 |