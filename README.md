Experiments on public datasets for `pytorch-lifestream` library

# Setup and test using pipenv

```sh
# Ubuntu 18.04

sudo apt install python3.8 python3-venv
pip3 install pipenv

pipenv sync  --dev # install packages exactly as specified in Pipfile.lock
pipenv shell
pytest

# run luigi server
luigid
# check embedding validation progress at `http://localhost:8082/`

# use tensorboard for metrics exploration
tensorboard --logdir lightning_logs/ 
# check tensorboard metrics at `http://localhost:6006/`

```

# Run scenario
 We check 5 datasets as separate experiments. See `README.md` files in experiments folder:
 - [Age](scenario_age_pred/README.md)
 - [Churn](scenario_rosbank/README.md)
 - [Assess](scenario_bowl2019/README.md)
 - [Retail](scenario_x5/README.md)
 - [Scoring](scenario_alpha_battle/README.md)
 - [Small demo dataset](scenario_gender/README.md)

# Final results
```
All results are stored in `*/results` folder.
Here are the copy of them.

Unsupervised learned embeddings with LightGBM model downstream evaluations:
mean from test, std from valid
                         |     mean \pm std      |
    Gender auroc:
        baseline         |    0.877 \pm 0.010    |
        cpc_embeddings   |    0.851 \pm 0.006    |
        mles2_embeddings |    0.882 \pm 0.006    |
        mles_embeddings  |    0.881 \pm 0.006    |
        nsp_embeddings   |    0.852 \pm 0.011    |
        random_encoder   |    0.593 \pm 0.020    |
        rtd_embeddings   |    0.855 \pm 0.008    |
        sop_embeddings   |    0.785 \pm 0.007    |
        barlow_twins     |    0.865 \pm 0.007    |
                         
    Age group (age_pred) accuracy:
        baseline         |    0.629 \pm 0.002    |
        cpc_embeddings   |    0.602 \pm 0.004    |
        mles2_embeddings |    0.643 \pm 0.003    |
        mles_embeddings  |    0.640 \pm 0.004    |
        nsp_embeddings   |    0.621 \pm 0.005    |
        random_encoder   |    0.375 \pm 0.003    |
        rtd_embeddings   |    0.631 \pm 0.006    |
        sop_embeddings   |    0.512 \pm 0.002    |
        barlow_twins     |    0.634 \pm 0.003    |
    
    Churn (rosbank) auroc:
        baseline         |    0.827  \pm 0.010   |
        cpc_embeddings   |    0.792  \pm 0.015   |
        mles2_embeddings |    0.837  \pm 0.006   |
        mles_embeddings  |    0.841  \pm 0.010   |
        nsp_embeddings   |    0.828  \pm 0.012   |
        random_encoder   |    0.725  \pm 0.013   |
        rtd_embeddings   |    0.771  \pm 0.016   |
        sop_embeddings   |    0.780  \pm 0.012   |
        barlow_twins     |    0.839  \pm 0.010   |
        
    Assessment (bowl2019) accuracy:
        barlow_twins     |    0.595 \pm 0.005    |    
        baseline         |    0.592 \pm 0.004    |    
        cpc_embeddings   |    0.593 \pm 0.004    |    
        mles2_embeddings |    0.588 \pm 0.008    |    
        mles_embeddings  |    0.597 \pm 0.001    |    
        nsp_embeddings   |    0.579 \pm 0.002    |    
        random_encoder   |    0.574 \pm 0.004    |
        rtd_embeddings   |    0.574 \pm 0.004    |
        sop_embeddings   |    0.567 \pm 0.005    |
    
    Retail (x5) accuracy:
        baseline         |    0.547 \pm 0.001    |
        cpc_embeddings   |    0.525 \pm 0.001    |
        mles_embeddings  |    0.539 \pm 0.001    |
        nsp_embeddings   |    0.425 \pm 0.002    |
        rtd_embeddings   |    0.520 \pm 0.001    |
        sop_embeddings   |    0.428 \pm 0.001    |
    
    Scoring (alpha battle) auroc:
        baseline         |    0.7792 \pm 0.0006  |
        random_encoder   |    0.6456 \pm 0.0009  |
        barlow_twins     |    0.7878 \pm 0.0009  |
        cpc              |    0.7919 \pm 0.0004  |
        mles             |    0.7921 \pm 0.0003  |
        nsp              |    0.7655 \pm 0.0006  |
        rtd              |    0.7910 \pm 0.0006  |
        sop              |    0.7238 \pm 0.0010  |


Supervised finetuned encoder with MLP head evaluation:
                         |     mean \pm std      |
    Gender auroc:
        barlow_twins     |    0.865 \pm 0.011    |
        cpc_finetuning   |    0.865 \pm 0.007    |
        mles_finetuning  |    0.879 \pm 0.007    |
        rtd_finetuning   |    0.868 \pm 0.006    |
        target_scores    |    0.867 \pm 0.008    |

    Age group (age_pred) accuracy:
        barlow_twins     |    0.619 \pm 0.004    |
        cpc_finetuning   |    0.625 \pm 0.005    |
        mles_finetuning  |    0.624 \pm 0.005    |
        rtd_finetuning   |    0.622 \pm 0.003    |
        target_scores    |    0.620 \pm 0.006    |
    
    Churn (rosbank) auroc:
        barlow_twins     |    0.830 \pm 0.006    |
        cpc_finetuning   |    0.804 \pm 0.017    |
        mles_finetuning  |    0.819 \pm 0.011    |
        nsp_finetuning   |    0.806 \pm 0.010    |
        rtd_finetuning   |    0.791 \pm 0.016    |
        target_scores    |    0.818 \pm 0.005    |

    Assessment (bowl2019) accuracy:
        barlow_twins     |    0.561 \pm 0.007    |    
        cpc_finetuning   |    0.594 \pm 0.002    |    
        mles_finetuning  |    0.577 \pm 0.007    |    
        rtd_finetuning   |    0.571 \pm 0.003    |    
        target_scores    |    0.585 \pm 0.002    |
    
    Retail (x5) accuracy:
        cpc_finetuning   |    0.549 \pm 0.001    |
        mles_finetuning  |    0.552 \pm 0.001    |
        rtd_finetuning   |    0.544 \pm 0.002    |
        target_scores    |    0.542 \pm 0.001    |

```

# Other experiments

- [Data Fusion Contest 2022. 1-st place on the Matching Task](https://github.com/ivkireev86/datafusion-contest-2022)
- [Alpha BKI dataset](scenario_alpha_rnn_vs_transformer/README.md)
- [American Express - Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction)
    - [Supervised training with RNN](https://www.kaggle.com/code/ivkireev/amex-ptls-baseline-supervised-neural-network)
    - [Supervised training with Transformer](https://www.kaggle.com/code/ivkireev/amex-transformer-network-train-with-ptls)
    - [CoLES Embedding preparation](https://www.kaggle.com/code/ivkireev/amex-contrastive-embeddings-with-ptls-coles)
    - [CoLES Embedding usage as extra features for catboost](https://www.kaggle.com/code/ivkireev/catboost-classifier-with-coles-embeddings)

