# Prepare agg feature encoder and take embeddings; inference
python -m ptls.pl_inference --config-dir conf --config-name agg_features_params

# Random encoder
python -m ptls.pl_inference --config-dir conf --config-name random_params

# for inference customize and run scripts from `bin/embeddings_by_epochs`

## Train the MeLES encoder and take embeddings; inference
python -m ptls.pl_train_module --config-dir conf --config-name mles_params
#
## Train the Contrastive Predictive Coding (CPC) model; inference
python -m ptls.pl_train_module --config-dir conf --config-name cpc_params
#
## Train the Sequence Order Prediction (SOP) model; inference
python -m ptls.pl_train_module --config-dir conf --config-name sop_params
#
## Train the Next Sequence Prediction (NSP) model; inference
python -m ptls.pl_train_module --config-dir conf --config-name nsp_params
#
## Train the Replaced Token Detection (RTD) model; inference
python -m ptls.pl_train_module --config-dir conf --config-name rtd_params

## Train the Barlow Twins model; inference
python -m ptls.pl_train_module --config-dir conf --config-name barlow_twins_params


# Compare
rm results/scenario_alpha_battle_baselines_unsupervised.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_baselines_unsupervised +workers=1 +total_cpu_count=20
