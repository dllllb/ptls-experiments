# ContrastiveLoss (positive stronger)
export SC_SUFFIX="loss_contrastive_margin_0.5"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.loss="ContrastiveLoss" \
    params.train.margin=0.5 \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_bowl2019/mles__$SC_SUFFIX.p" \
    --config-dir conf --config-name mles_params
python -m ptls.pl_inference \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_bowl2019/mles__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb__$SC_SUFFIX" \
    --config-dir conf --config-name mles_params

# ContrastiveLoss (negative stronger)
export SC_SUFFIX="loss_contrastive_margin_1.0"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.loss="ContrastiveLoss" \
    params.train.margin=1.0 \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_bowl2019/mles__$SC_SUFFIX.p" \
    --config-dir conf --config-name mles_params
python -m ptls.pl_inference \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_bowl2019/mles__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb__$SC_SUFFIX" \
    --config-dir conf --config-name mles_params

# BinomialDevianceLoss (positive stronger)
export SC_SUFFIX="loss_binomialdeviance_C_1.0_alpha_1.0_beta_0.4"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.loss="BinomialDevianceLoss" \
    params.train.C=1.0 \
    params.train.alpha=1.0 \
    params.train.beta=0.4 \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_bowl2019/mles__$SC_SUFFIX.p" \
    --config-dir conf --config-name mles_params
python -m ptls.pl_inference \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_bowl2019/mles__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb__$SC_SUFFIX" \
    --config-dir conf --config-name mles_params

# BinomialDevianceLoss (negative stronger)
export SC_SUFFIX="loss_binomialdeviance_C_6.0_alpha_0.4_beta_0.7"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.loss="BinomialDevianceLoss" \
    params.train.C=6.0 \
    params.train.alpha=0.4 \
    params.train.beta=0.7 \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_bowl2019/mles__$SC_SUFFIX.p" \
    --config-dir conf --config-name mles_params
python -m ptls.pl_inference \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_bowl2019/mles__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb__$SC_SUFFIX" \
    --config-dir conf --config-name mles_params

# TripletLoss
export SC_SUFFIX="loss_triplet_margin_0.3"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.loss="TripletLoss" \
    params.train.margin=0.3 \
    params.train.sampling_strategy="HardTriplets" \
    params.train.neg_count=5 \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_bowl2019/mles__$SC_SUFFIX.p" \
    --config-dir conf --config-name mles_params
python -m ptls.pl_inference \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_bowl2019/mles__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb__$SC_SUFFIX" \
    --config-dir conf --config-name mles_params

export SC_SUFFIX="loss_triplet_margin_0.6"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.loss="TripletLoss" \
    params.train.margin=0.6 \
    params.train.sampling_strategy="HardTriplets" \
    params.train.neg_count=5 \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_bowl2019/mles__$SC_SUFFIX.p" \
    --config-dir conf --config-name mles_params
python -m ptls.pl_inference \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_bowl2019/mles__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb__$SC_SUFFIX" \
    --config-dir conf --config-name mles_params

# HistogramLoss
export SC_SUFFIX="loss_histogramloss"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.loss="HistogramLoss" \
    params.train.num_steps=25 \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_bowl2019/mles__$SC_SUFFIX.p" \
    --config-dir conf --config-name mles_params
python -m ptls.pl_inference \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_bowl2019/mles__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb__$SC_SUFFIX" \
    --config-dir conf --config-name mles_params

# MarginLoss (positive stronger)
export SC_SUFFIX="loss_margin_0.2_beta_0.4"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.loss="MarginLoss" \
    params.train.margin=0.2 \
    params.train.beta=0.4 \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_bowl2019/mles__$SC_SUFFIX.p" \
    --config-dir conf --config-name mles_params
python -m ptls.pl_inference \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_bowl2019/mles__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb__$SC_SUFFIX" \
    --config-dir conf --config-name mles_params

# MarginLoss (negative stronger)
export SC_SUFFIX="loss_margin_0.3_beta_0.6"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.loss="MarginLoss" \
    params.train.margin=0.3 \
    params.train.beta=0.6 \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_bowl2019/mles__$SC_SUFFIX.p" \
    --config-dir conf --config-name mles_params
python -m ptls.pl_inference \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_bowl2019/mles__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb__$SC_SUFFIX" \
    --config-dir conf --config-name mles_params

# Compare
rm results/scenario_bowl2019__loss.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_short +workers=10 +total_cpu_count=20 +local_scheduler=True \
    report_file="${hydra:runtime.cwd}/results/scenario_bowl2019__loss.txt" \    
    auto_features=["${hydra:runtime.cwd}/data/emb__loss_*.pickle"]
