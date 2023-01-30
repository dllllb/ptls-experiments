# Get data

```sh
cd scenario_age_pred

# download datasets
sh bin/get-data.sh

# convert datasets from transaction list to features for metric learning
sh bin/make_datasets_spark_file.sh
```

# Main scenario, best params

```sh
cd scenario_age_pred
export CUDA_VISIBLE_DEVICES=0

sh bin/run_all_scenarios.sh

# check the results
cat results/*.txt
cat results/*.csv
```

# Comments on experiments

`results/scenario_age_pred_custom_unsupervised.txt` 
(Experiment shows metrics from downstream task with MLES pretrained with different models)

[stacking attention blocks strategy]\_[self-attention architecture]\_[hs]\_[n_head]\_[n_layer]\_[layer_norm]

- [stacking attention blocks strategy] - *rezero* (https://arxiv.org/abs/2003.04887), *defaul* - without learnable skip-connection weights 
- [self-attention architecture] - *flow* (https://arxiv.org/abs/2203.16194), *quad* - classic quadratic attention matrix
- [layer_norm] - if *pre* we use layer_norm before attn and after attn, if *post* we use layer_norm before MLP and after it, if noln we use ln only between attn and MLP.
