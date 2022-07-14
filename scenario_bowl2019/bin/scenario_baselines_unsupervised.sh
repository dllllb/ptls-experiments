# Prepare agg feature encoder and take embeddings; inference
python -m ptls.pl_inference --config-dir conf --config-name agg_features_params

# Random encoder
python -m ptls.pl_inference --config-dir conf --config-name random_params

# Train the MeLES encoder and take embeddings; inference
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
    data_module.train_data.splitter.split_count=2 \
    data_module.valid_data.splitter.split_count=2 \
    pl_module.validation_metric.K=1 \
    trainer.max_epochs=200 \
    pl_module.lr_scheduler_partial.step_size=30 \
    model_path="models/mles_model2.p" \
    logger_name="mles_model2" \
    --config-dir conf --config-name mles_params
python -m ptls.pl_inference    \
    model_path="models/mles_model2.p" \
    embed_file_name="mles2_embeddings" \
    --config-dir conf --config-name mles_params

# Train the Replaced Token Detection (RTD) model; inference
python -m ptls.pl_train_module --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_inference --config-dir conf --config-name barlow_twins_params


# Compare
rm results/scenario_bowl2019_baselines_unsupervised.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_baselines_unsupervised +workers=10 +total_cpu_count=20
