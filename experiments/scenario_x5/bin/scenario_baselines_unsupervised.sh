# Prepare agg feature encoder and take embedidngs; inference
python -m dltranz.pl_train_module --conf conf/agg_features_params.hocon
python -m dltranz.pl_inference    --conf conf/agg_features_params.hocon

# Random encoder
python -m dltranz.pl_inference    --conf conf/random_params.hocon

# Train the MeLES encoder and take embedidngs; inference
python -m dltranz.pl_train_module --conf conf/mles_params.hocon
python -m dltranz.pl_inference    --conf conf/mles_params.hocon

# Train the Contrastive Predictive Coding (CPC) model; inference
python -m dltranz.pl_train_module --conf conf/cpc_params.hocon
python -m dltranz.pl_inference    --conf conf/cpc_params.hocon

# Train the Sequence Order Prediction (SOP) model; inference
python -m dltranz.pl_train_module --conf conf/sop_params.hocon
python -m dltranz.pl_inference    --conf conf/sop_params.hocon

# Train the Next Sequence Prediction (NSP) model; inference
python -m dltranz.pl_train_module --conf conf/nsp_params.hocon
python -m dltranz.pl_inference    --conf conf/nsp_params.hocon

# Train the Replaced Token Detection (RTD) model; inference
python -m dltranz.pl_train_module --conf conf/rtd_params.hocon
python -m dltranz.pl_inference    --conf conf/rtd_params.hocon

# Check COLEs with split_count=2
# was
python -m dltranz.pl_train_module \
    data_module.train.split_strategy.split_count=2 \
    data_module.valid.split_strategy.split_count=2 \
    params.validation_metric_params.K=1 \
    trainer.max_epochs=60 \
    params.lr_scheduler.step_size=6 \
    model_path="../../artifacts/scenario_x5/mles_model2.p" \
    logger_name="mles_model2" \
    --conf conf/mles_params.hocon
python -m dltranz.pl_inference    \
    model_path="../../artifacts/scenario_x5/mles_model2.p" \
    output.path="data/mles2_embeddings" \
    --conf conf/mles_params.hocon

# Train the Replaced Token Detection (RTD) model; inference
python -m dltranz.pl_train_module --conf conf/barlow_twins_params.hocon
python -m dltranz.pl_inference --conf conf/barlow_twins_params.hocon


# Compare
rm results/scenario_x5_baselines_unsupervised.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_baselines_unsupervised.hocon --workers 10 --total_cpu_count 20
