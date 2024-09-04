# Results on some datasets

Unsupervised learned embeddings with LightGBM model downstream evaluations:
|                         |     mean $\pm$ std      |
|-------------------------|-------------------------|
|    **Gender**           |  **AUROC**              |
|        baseline         |    0.877 $\pm$ 0.010    |
|        cpc_embeddings   |    0.851 $\pm$ 0.006    |
|        mles2_embeddings |    0.882 $\pm$ 0.006    |
|        mles_embeddings  |    0.881 $\pm$ 0.006    |
|        nsp_embeddings   |    0.852 $\pm$ 0.011    |
|        random_encoder   |    0.593 $\pm$ 0.020    |
|        rtd_embeddings   |    0.855 $\pm$ 0.008    |
|        sop_embeddings   |    0.785 $\pm$ 0.007    |
|        barlow_twins     |    0.865 $\pm$ 0.007    |
| **Age group (age_pred)**|  **Accuracy**           |
|        baseline         |    0.629 $\pm$ 0.002    |
|        cpc_embeddings   |    0.602 $\pm$ 0.004    |
|        mles2_embeddings |    0.643 $\pm$ 0.003    |
|        mles_embeddings  |    0.640 $\pm$ 0.004    |
|        mles_longformer  |    0.630 $\pm$ 0.003    |
|        nsp_embeddings   |    0.621 $\pm$ 0.005    |
|        random_encoder   |    0.375 $\pm$ 0.003    |
|        rtd_embeddings   |    0.631 $\pm$ 0.006    |
|        sop_embeddings   |    0.512 $\pm$ 0.002    |
|        barlow_twins     |    0.634 $\pm$ 0.003    |
|        coles_transformer|    0.646 $\pm$ 0.003    |
|    **Churn (rosbank)**  |  **AUROC**              |
|        baseline         |    0.827  $\pm$ 0.010   |
|        cpc_embeddings   |    0.792  $\pm$ 0.015   |
|        mles2_embeddings |    0.837  $\pm$ 0.006   |
|        mles_embeddings  |    0.841  $\pm$ 0.010   |
|        nsp_embeddings   |    0.828  $\pm$ 0.012   |
|        random_encoder   |    0.725  $\pm$ 0.013   |
|        rtd_embeddings   |    0.771  $\pm$ 0.016   |
|        sop_embeddings   |    0.780  $\pm$ 0.012   |
|        barlow_twins     |    0.839  $\pm$ 0.010   |
|**Assessment (bowl2019)**|  **Accuracy**           |
|        barlow_twins     |    0.595 $\pm$ 0.005    |    
|        baseline         |    0.592 $\pm$ 0.004    |    
|        cpc_embeddings   |    0.593 $\pm$ 0.004    |    
|        mles2_embeddings |    0.588 $\pm$ 0.008    |    
|        mles_embeddings  |    0.597 $\pm$ 0.001    |    
|        nsp_embeddings   |    0.579 $\pm$ 0.002    |    
|        random_encoder   |    0.574 $\pm$ 0.004    |
|        rtd_embeddings   |    0.574 $\pm$ 0.004    |
|        sop_embeddings   |    0.567 $\pm$ 0.005    |    
|    **Retail (x5)**      |  **Accuracy**           |
|        baseline         |    0.547 $\pm$ 0.001    |
|        cpc_embeddings   |    0.525 $\pm$ 0.001    |
|        mles_embeddings  |    0.539 $\pm$ 0.001    |
|        nsp_embeddings   |    0.425 $\pm$ 0.002    |
|        rtd_embeddings   |    0.520 $\pm$ 0.001    |
|        sop_embeddings   |    0.428 $\pm$ 0.001    |
|**Scoring (alpha battle)**| **AUROC**              |
|        baseline         |    0.7792 $\pm$ 0.0006  |
|        random_encoder   |    0.6456 $\pm$ 0.0009  |
|        barlow_twins     |    0.7878 $\pm$ 0.0009  |
|        cpc              |    0.7919 $\pm$ 0.0004  |
|        mles             |    0.7921 $\pm$ 0.0003  |
|        nsp              |    0.7655 $\pm$ 0.0006  |
|        rtd              |    0.7910 $\pm$ 0.0006  |
|        sop              |    0.7238 $\pm$ 0.0010  |
|        mlmnsp           |    0.7591 $\pm$ 0.0044  |
|        tabformer        |    0.7862 $\pm$ 0.0042  |
|        gpt              |    0.7737 $\pm$ 0.0032  |
|   coles_transformer     |    0.7968 $\pm$ 0.0007  |

Supervised finetuned encoder with MLP head evaluation:
|                         |     mean $\pm$ std      |
|-------------------------|-------------------------|
|    **Gender**           |  **AUROC**              |
|        barlow_twins     |    0.865 $\pm$ 0.011    |
|        cpc_finetuning   |    0.865 $\pm$ 0.007    |
|        mles_finetuning  |    0.879 $\pm$ 0.007    |
|        rtd_finetuning   |    0.868 $\pm$ 0.006    |
|        target_scores    |    0.867 $\pm$ 0.008    |
|**Age group (age_pred)** |  **Accuracy**           |
|        barlow_twins     |    0.619 $\pm$ 0.004    |
|        cpc_finetuning   |    0.625 $\pm$ 0.005    |
|        mles_finetuning  |    0.624 $\pm$ 0.005    |
|        rtd_finetuning   |    0.622 $\pm$ 0.003    |
|        target_scores    |    0.620 $\pm$ 0.006    |
|    **Churn (rosbank)**  |  **AUROC**              |
|        barlow_twins     |    0.830 $\pm$ 0.006    |
|        cpc_finetuning   |    0.804 $\pm$ 0.017    |
|        mles_finetuning  |    0.819 $\pm$ 0.011    |
|        nsp_finetuning   |    0.806 $\pm$ 0.010    |
|        rtd_finetuning   |    0.791 $\pm$ 0.016    |
|        target_scores    |    0.818 $\pm$ 0.005    |
|**Assessment (bowl2019)**|  **Accuracy**           |
|        barlow_twins     |    0.561 $\pm$ 0.007    |    
|        cpc_finetuning   |    0.594 $\pm$ 0.002    |    
|        mles_finetuning  |    0.577 $\pm$ 0.007    |    
|        rtd_finetuning   |    0.571 $\pm$ 0.003    |    
|        target_scores    |    0.585 $\pm$ 0.002    |
|    **Retail (x5)**      |  **Accuracy**           |
|        cpc_finetuning   |    0.549 $\pm$ 0.001    |
|        mles_finetuning  |    0.552 $\pm$ 0.001    |
|        rtd_finetuning   |    0.544 $\pm$ 0.002    |
|        target_scores    |    0.542 $\pm$ 0.001    |
