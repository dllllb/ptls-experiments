# Random and arange features experiment

Random and arange features were added to rosbank dataset to test their impact on quality on the downstream task for CPC and Coles modules.

## Random features

Random features are from uniform discrete distribution (`np.random.randint`) with 100 unique values. Embeddings size for random features was 24 for Coles and 18 for CPC.

## Arange features 

There is two types of arange features: 

1. Arange features - just an arange features for every user (`np.arange(len(user_items))`)
2. Cycled arange features - `np.arange(len(user_items)) mod 100 + 1` 


# Results 

Test results for rosbank dataset. The more random features, the larger the metric on validation ( `recall_top_k`). Also tested random features impact on quality with higher hidden size (x2 and x4). 
 
## No arange, no random features 

| method | Coles | CPC  | 
| --- | --- | --- | 
| Accuracy  |  0.770 ± 0.009 | 0.736 ± 0.015 |     
|  AUCROC |  0.844 ± 0.003 |  0.797 ± 0.004|  

## Only arange 


| method | Coles | CPC  | 
| --- | --- | --- | 
| Accuracy  |  0.758 ± 0.011 |  0.702 ± 0.011 |     
|  AUCROC |  0.842 ± 0.003  |  0.779 ± 0.007 |   


## Only cycled arange 

| method | Coles | CPC  | 
| --- | --- | --- | 
| Accuracy  |  0.763 ± 0.012 | 0.715 ± 0.011 |     
|  AUCROC | 0.842 ± 0.002 |  0.790 ± 0.005 |      


## 1 random feature 

| method | Coles | CPC  | 
| --- | --- | --- | 
| Accuracy  |  0.756 ± 0.008 | 0.680 ± 0.010 |     
|  AUCROC |  0.820 ± 0.007 | 0.747 ± 0.008 |   

## 1 random feature and hidden size x2 

| method | Coles | CPC  | 
| --- | --- | --- | 
| Accuracy  |  0.739 ± 0.011 |  0.704 ± 0.011 |     
|  AUCROC |  0.812 ± 0.006 |  0.761 ± 0.006 |

## 1 random and arange features 

| method | Coles | CPC  | 
| --- | --- | --- | 
| Accuracy  |  0.758 ± 0.006 |  0.681 ± 0.005|     
|  AUCROC |  0.818 ± 0.009 | 0.765 ± 0.004 |  


## 2 random features 

| method | Coles | CPC  | 
| --- | --- | --- | 
| Accuracy  |  0.668 ± 0.013 |  0.687 ± 0.006 |     
|  AUCROC |  0.725 ± 0.008 |  0.742 ± 0.009 |  

## 2 random features and hidden size x2 

| method | Coles | CPC  | 
| --- | --- | --- | 
| Accuracy  |  0.682 ± 0.018 |  0.674 ± 0.010 |     
|  AUCROC |  0.739 ± 0.011 | 0.737 ± 0.005 |

## 2 random features and hidden size x4

| method | Coles | CPC  | 
| --- | --- | --- | 
| Accuracy  |  0.758 ± 0.008 |  0.693 ± 0.006|     
|  AUCROC |  0.837 ± 0.005 |  0.760 ± 0.005|

## 3 random features 

| method | Coles | CPC  | 
| --- | --- | --- | 
| Accuracy  |  0.660 ± 0.012 |  0.673 ± 0.011 |     
|  AUCROC |  0.701 ± 0.006 |  0.691 ± 0.011 |  


## 5 random features 

| method | Coles | CPC  | 
| --- | --- | --- | 
| Accuracy  |  0.654 ± 0.013 |  0.653 ± 0.013|     
|  AUCROC |  0.616 ± 0.023 |  0.670 ± 0.011 |    

