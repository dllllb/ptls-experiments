# Prepare agg feature encoder and take embedidngs; inference
python -m ptls.pl_train_module --config-dir conf --config-name agg_features_params
python -m ptls.pl_inference --config-dir conf --config-name agg_features_params

# Random encoder
python -m ptls.pl_inference --config-dir conf --config-name random_params

# Train the MeLES encoder and take embedidngs; inference
python -m ptls.pl_train_module --config-dir conf --config-name mles_params
python -m ptls.pl_inference --config-dir conf --config-name mles_params

# Train the Contrastive Predictive Coding (CPC) model; inference
python -m ptls.pl_train_module --config-dir conf --config-name cpc_params
python -m ptls.pl_inference --config-dir conf --config-name cpc_params

# Train the Sequence Order Prediction (SOP) model; inference
python -m ptls.pl_train_module --config-dir conf --config-name sop_params
python -m ptls.pl_inference --config-dir conf --config-name sop_params

# Train the Next Sequence Prediction (NSP) model; inference
python -m ptls.pl_train_module --config-dir conf --config-name nsp_params
python -m ptls.pl_inference --config-dir conf --config-name nsp_params

# Train the Replaced Token Detection (RTD) model; inference
python -m ptls.pl_train_module --config-dir conf --config-name rtd_params
python -m ptls.pl_inference --config-dir conf --config-name rtd_params

# Check COLEs with split_count=2
# was
python -m ptls.pl_train_module \
    data_module.train.split_strategy.split_count=2 \
    data_module.valid.split_strategy.split_count=2 \
    params.validation_metric_params.K=1 \
    trainer.max_epochs=120 \
    params.lr_scheduler.step_size=60 \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/mles_model2.p" \
    logger_name="mles_model2" \
    --config-dir conf --config-name mles_params
python -m ptls.pl_inference    \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/mles_model2.p" \
    output.path="${hydra:runtime.cwd}/data/mles2_embeddings" \
    --config-dir conf --config-name mles_params

# Train the Replaced Token Detection (RTD) model; inference
python -m ptls.pl_train_module --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_inference --config-dir conf --config-name barlow_twins_params

# Compare
rm results/scenario_rosbank_baselines_unsupervised.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_baselines_unsupervised +workers=10 +total_cpu_count=20

