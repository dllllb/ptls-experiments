# Random and arange features experiment

Random and arange features were added to rosbank dataset to test their impact on quality on the downstream task for CPC and Coles modules.

## Random features

Random features are from uniform discrete distribution (`np.random.randint`) with 100 unique values. Embeddings size for random features was 24 for Coles and 18 for CPC.

## Arange features 

There is two types of arange features: 

1. Arange features - just an arange features for every user (`np.arange(len(user_items))`)
2. Cycled arange features - `np.arange(len(user_items)) mod 100 + 1` 


# Rosbank results 

Test results for the rosbank dataset. The more random features, the larger the metric on validation ( `recall_top_k`). Also tested random features impact on quality with higher hidden size (x2 and x4). 


## AUCROC 

| method                                | Coles          | CPC             | 
| ------------------------------------- | -------------- | --------------- | 
|  No arange, no random features        |  0.844 ± 0.003 |  0.797 ± 0.004  |
|  dropout 0.3                          |  0.848 ± 0.004 |  0.802 ± 0.006  |
|  spatial dropout 0.3                  |  0.837 ± 0.005 |  0.803 ± 0.004  |
|  Only arange                          |  0.842 ± 0.003 |  0.779 ± 0.007  |
|  Only cycled arange                   |  0.842 ± 0.002 |  0.790 ± 0.005  |
|  1 random feature                     |  0.820 ± 0.007 |  0.747 ± 0.008  |
|  1 random feature and hidden size x2  |  0.812 ± 0.006 |  0.761 ± 0.006  |
|  1 random feature and dropout 0.3     |  0.827 ± 0.004 |  0.767 ± 0.005  |
|  1 rand feature spatial dropout 0.3   |  0.807 ± 0.005 |  0.761 ± 0.008  |
|  1 random and arange features         |  0.818 ± 0.009 |  0.765 ± 0.004  |
|  2 random features                    |  0.725 ± 0.008 |  0.742 ± 0.009  | 
|  2 random features and hidden size x2 |  0.739 ± 0.011 |  0.737 ± 0.005  |
|  2 random features and hidden size x4 |  0.837 ± 0.005 |  0.760 ± 0.005  |
|  2 random feature and dropout 0.3     |  0.762 ± 0.007 |  0.759 ± 0.004  |
|  2 rand feature spatial dropout 0.3   |  0.754 ± 0.008 |  0.755 ± 0.006  |
|  3 random features                    |  0.701 ± 0.006 |  0.691 ± 0.011  | 
|  5 random features                    |  0.616 ± 0.023 |  0.670 ± 0.011  | 

## Accuracy 

| method                                | Coles          | CPC           | 
| ------------------------------------- | -------------- | ------------- | 
| No arange, no random features         |  0.770 ± 0.009 | 0.736 ± 0.015 |
|  dropout 0.3                          |  0.766 ± 0.007 | 0.726 ± 0.006 |
|  spatial dropout 0.3                  |  0.756 ± 0.012 | 0.730 ± 0.012 |
| Only arange                           |  0.758 ± 0.011 | 0.702 ± 0.011 |  
| Only cycled arange                    |  0.763 ± 0.012 | 0.715 ± 0.011 | 
| 1 random feature                      |  0.756 ± 0.008 | 0.680 ± 0.010 |
| 1 random feature and hidden size x2   |  0.739 ± 0.011 | 0.704 ± 0.011 |
|  1 random feature and dropout 0.3     |  0.744 ± 0.011 | 0.694 ± 0.006 |
|  1 rand feature spatial dropout 0.3   |  0.732 ± 0.010 | 0.687 + 0.011 |
| 1 random and arange features          |  0.758 ± 0.006 | 0.681 ± 0.005 |  
| 2 random features                     |  0.668 ± 0.013 | 0.687 ± 0.006 | 
| 2 random features and hidden size x2  |  0.682 ± 0.018 | 0.674 ± 0.010 | 
| 2 random features and hidden size x4  |  0.758 ± 0.008 | 0.693 ± 0.006 |
|  2 random feature and dropout 0.3     |  0.685 ± 0.010 | 0.686 ± 0.012 |
|  2 rand feature spatial dropout 0.3   |  0.690 ± 0.012 | 0.689 ± 0.007 |
| 3 random features                     |  0.660 ± 0.012 | 0.673 ± 0.011 |    
| 5 random features                     |  0.654 ± 0.013 | 0.653 ± 0.013 |     


# Gender results 

Test results for the gender dataset. 

## AUCROC 

| method                                | Coles          | CPC             | 
| ------------------------------------- | -------------- | --------------- | 
|  No arange, no random features        |  0.882 ± 0.003 |  0.853 ± 0.003  |
|  dropout 0.3                          |  0.882 ± 0.003 |  0.851 ± 0.004  |
|  spatial dropout 0.3                  |  0.871 ± 0.002 |  0.851 ± 0.004  |
|  Only arange                          |  0.879 ± 0.003 |  0.725 ± 0.007  |
|  Only cycled arange                   |  0.882 ± 0.003 |  0.832 ± 0.003  |
|  1 random feature                     |  0.806 ± 0.001 |  0.662 ± 0.010  |
|  1 random feature and hidden size x2  |  0.787 ± 0.006 |  0.699 ± 0.006  |
|  1 random feature and dropout 0.3     |  0.818 ± 0.003 |  0.649 ± 0.019  |
|  1 rand feature spatial dropout 0.3   |  0.780 ± 0.006 |  0.672 ± 0.005  |
|  1 random and arange features         |  0.815 ± 0.005 |  0.518 ± 0.017  |
|  2 random features                    |  0.507 ± 0.013 |  0.654 ± 0.004  | 
|  2 random features and hidden size x2 |  0.539 ± 0.016 |  0.723 ± 0.011  |
|  2 random features and hidden size x4 |  0.507 ± 0.007 |  0.699 ± 0.006  |
|  2 random feature and dropout 0.3     |  0.521 ± 0.006 |  0.639 ± 0.009  |
|  2 rand feature spatial dropout 0.3   |  0.520 ± 0.008 |  0.663 ± 0.002  |
|  3 random features                    |  0.527 ± 0.024 |  0.650 ± 0.008  | 
|  5 random features                    |  0.541 ± 0.014 |  0.641 ± 0.010  | 

## Accuracy 

| method                                | Coles          | CPC            | 
| ------------------------------------- | -------------- | -------------- | 
| No arange, no random features         |  0.795 ± 0.003 |  0.769 ± 0.006 |
|  dropout 0.3                          |  0.790 ± 0.003 |  0.767 ± 0.009 |
|  spatial dropout 0.3                  |  0.787 ± 0.005 |  0.767 ± 0.007 |
| Only arange                           |  0.787 ± 0.008 |  0.669 ± 0.015 |  
| Only cycled arange                    |  0.798 ± 0.005 |  0.747 ± 0.003 | 
| 1 random feature                      |  0.727 ± 0.003 |  0.625 ± 0.011 |
| 1 random feature and hidden size x2   |  0.716 ± 0.006 |  0.646 ± 0.005 |
|  1 random feature and dropout 0.3     |  0.733 ± 0.008 |  0.618 ± 0.016 |
|  1 rand feature spatial dropout 0.3   |  0.709 ± 0.006 |  0.635 ± 0.007 |
| 1 random and arange features          |  0.738 ± 0.012 |  0.520 ± 0.007 |  
| 2 random features                     |  0.514 ± 0.014 |  0.616 ± 0.009 | 
| 2 random features and hidden size x2  |  0.538 ± 0.016 |  0.671 ± 0.008 | 
| 2 random features and hidden size x4  |  0.514 ± 0.011 |  0.646 ± 0.005 |
|  2 random feature and dropout 0.3     |  0.529 ± 0.009 |  0.606 ± 0.014 |
|  2 rand feature spatial dropout 0.3   |  0.519 ± 0.012 |  0.614 ± 0.005 |
| 3 random features                     |  0.526 ± 0.018 |  0.602 ± 0.012 |    
| 5 random features                     |  0.540 ± 0.012 |  0.602 ± 0.011 |     



# Age_pred results 

Test results for the age_pred dataset. 

## Accuracy 

| method                                | Coles          | CPC             | 
| ------------------------------------- | -------------- | --------------- | 
|  No arange, no random features        |  0.632 ± 0.002 |  0.595 ± 0.003  | 
|  dropout 0.3                          |  0.637 ± 0.001 |  0.588 ± 0.006  |
|  spatial dropout 0.3                  |  0.625 ± 0.004 |  0.527 ± 0.005  |
|  Only arange                          |  0.632 ± 0.004 |  0.542 ± 0.006  |
|  Only cycled arange                   |  0.625 ± 0.004 |  0.600 ± 0.005  |
|  1 random feature                     |  0.449 ± 0.003 |  0.443 ± 0.005  |
|  1 random feature and hidden size x2  |  0.591 ± 0.003 |  0.452 ± 0.005  |
|  1 random feature and dropout 0.3     |  0.591 ± 0.003 |  0.453 ± 0.003  |
|  1 rand feature spatial dropout 0.3   |  0.575 ± 0.005 |  0.390 ± 0.007  |
|  1 random and arange features         |  0.594 ± 0.004 |  0.340 ± 0.004  |
|  2 random features                    |  0.445 ± 0.008 |  0.381 ± 0.005  | 
|  2 random features and hidden size x2 |  0.397 ± 0.006 |  0.398 ± 0.003  |
|  2 random features and hidden size x4 |  0.263 ± 0.009 |  0.381 ± 0.003  |
|  2 random feature and dropout 0.3     |  0.361 ± 0.012 |  0.363 ± 0.005  |
|  2 rand feature spatial dropout 0.3   |  0.490 ± 0.004 |  0.372 ± 0.004  |
|  3 random features                    |  0.247 ± 0.005 |  0.356 ± 0.005  | 
|  5 random features                    |  0.252 ± 0.008 |  0.344 ± 0.006  |   


## Hidden size experiment 

In this experiment hidden size of coles and cpc models was increased to test  its impact on quality on the rosbank dataset with 2 random features. 

## AUCROC 

| method                      | Coles          | CPC             | 
| --------------------------- | -------------- | --------------- | 
|  default hidden size        |  0.725 ± 0.008 |  0.742 ± 0.009  | 
|  hidden size x2             |  0.739 ± 0.011 |  0.737 ± 0.005  |
|  hidden size x3             |  0.740 ± 0.006 |  0.763 ± 0.007  |
|  hidden size x4             |  0.837 ± 0.005 |  0.760 ± 0.005  |
|  hidden size x6             |  0.745 ± 0.004 |  0.775 ± 0.006  |
|  hidden size x8             |  0.729 ± 0.003 |  0.771 ± 0.005  |
|  hidden size x10            |  0.702 ± 0.008 |  0.765 ± 0.006  |


## Accuracy 

| method                      | Coles          | CPC             | 
| --------------------------- | -------------- | --------------- |
|  default hidden size        |  0.668 ± 0.013 |  0.687 ± 0.006  |  
|  hidden size x2             |  0.682 ± 0.018 |  0.674 ± 0.010  |
|  hidden size x3             |  0.685 ± 0.011 |  0.702 ± 0.015  |
|  hidden size x4             |  0.758 ± 0.008 |  0.693 ± 0.006  |
|  hidden size x6             |  0.683 ± 0.011 |  0.708 ± 0.008  |
|  hidden size x8             |  0.658 ± 0.013 |  0.693 ± 0.008  |
|  hidden size x10            |  0.652 ± 0.004 |  0.704 ± 0.010  |


## Full random features experiment 

Full random features were added to datasets (each row is random integer).

## Rosbank dataset results 


## AUCROC 

| method                      | Coles          | CPC             | 
| --------------------------- | -------------- | --------------- |
|  no random                  |  0.844 ± 0.003 |  0.797 ± 0.002  |
|  1 random feature           |  0.842 ± 0.006 |  0.805 ± 0.004  |
|  2 random features          |  0.840 ± 0.003 |  0.804 ± 0.003  |
|  3 random features          |  0.832 ± 0.005 |  0.805 ± 0.004  |
|  5 random features          |  0.832 ± 0.005 |  0.802 ± 0.005  |
|  7 random features          |  0.842 ± 0.007 |  0.799 ± 0.005  |
|  10 random features         |  0.834 ± 0.006 |  0.792 ± 0.003  |
|  12 random features         |  0.829 ± 0.005 |  0.789 ± 0.005  |
|  15 random features         |  0.826 ± 0.006 |  0.788 ± 0.010  |
|  20 random features         |  0.828 ± 0.005 |  0.772 ± 0.008  |


## Accuracy 

| method                      | Coles          | CPC             | 
| --------------------------- | -------------- | --------------- |
|  no random                  |  0.770 ± 0.008 |  0.734 ± 0.006  |
|  1 random feature           |  0.758 ± 0.011 |  0.747 ± 0.006  |
|  2 random features          |  0.764 ± 0.012 |  0.764 ± 0.012  |
|  3 random features          |  0.748 ± 0.007 |  0.748 ± 0.007  |
|  5 random features          |  0.744 ± 0.013 |  0.744 ± 0.013  |
|  7 random features          |  0.769 ± 0.006 |  0.769 ± 0.006  |
|  10 random features         |  0.762 ± 0.006 |  0.762 ± 0.006  |
|  12 random features         |  0.758 ± 0.015 |  0.758 ± 0.015  |
|  15 random features         |  0.745 ± 0.009 |  0.745 ± 0.009  |
|  20 random features         |  0.746 ± 0.011 |  0.746 ± 0.011  |


## Gender dataset results 


## AUCROC 

| method                      | Coles          | CPC             | 
| --------------------------- | -------------- | --------------- |
|  no random                  |  0.882 ± 0.003 |  0.853 ± 0.003  |
|  1 random feature           |  0.879 ± 0.003 |  0.855 ± 0.002  |
|  2 random features          |  0.875 ± 0.002 |  0.861 ± 0.002  |
|  3 random features          |  0.876 ± 0.002 |  0.852 ± 0.002  |
|  5 random features          |  0.871 ± 0.004 |  0.837 ± 0.003  |
|  7 random features          |  0.871 ± 0.004 |  0.820 ± 0.005  |
|  10 random features         |  0.863 ± 0.002 |  0.815 ± 0.002  |
|  12 random features         |  0.871 ± 0.003 |  0.788 ± 0.004  |
|  15 random features         |  0.864 ± 0.004 |  0.801 ± 0.006  |
|  20 random features         |  0.867 ± 0.003 |  0.760 ± 0.003  |


## Accuracy 

| method                      | Coles          | CPC             | 
| --------------------------- | -------------- | --------------- |
|  no random                  |  0.795 ± 0.003 |  0.769 ± 0.006  |
|  1 random feature           |  0.789 ± 0.006 |  0.775 ± 0.005  |
|  2 random features          |  0.788 ± 0.011 |  0.774 ± 0.005  |
|  3 random features          |  0.793 ± 0.004 |  0.771 ± 0.005  |
|  5 random features          |  0.785 ± 0.005 |  0.752 ± 0.006  |
|  7 random features          |  0.793 ± 0.007 |  0.740 ± 0.006  |
|  10 random features         |  0.779 ± 0.006 |  0.729 ± 0.005  |
|  12 random features         |  0.791 ± 0.006 |  0.702 ± 0.004  |
|  15 random features         |  0.773 ± 0.008 |  0.729 ± 0.012  |
|  20 random features         |  0.780 ± 0.007 |  0.694 ± 0.006  |


## Age pred dataset results 


## Accuracy 

| method                      | Coles          | CPC             | 
| --------------------------- | -------------- | --------------- |
|  no random                  |  0.632 ± 0.002 |  0.595 ± 0.003  |
|  1 random feature           |  0.638 ± 0.003 |  0.602 ± 0.007  |
|  2 random features          |  0.631 ± 0.006 |  0.601 ± 0.006  |
|  3 random features          |  0.645 ± 0.004 |  0.602 ± 0.002  |
|  5 random features          |  0.638 ± 0.005 |  0.609 ± 0.005  |
|  7 random features          |  0.646 ± 0.002 |  0.597 ± 0.003  |
|  10 random features         |  0.636 ± 0.002 |  0.589 ± 0.007  |
|  12 random features         |  0.628 ± 0.004 |  0.594 ± 0.003  |
|  15 random features         |  0.632 ± 0.004 |  0.580 ± 0.001  |
|  20 random features         |  0.633 ± 0.004 |  0.568 ± 0.003  |
