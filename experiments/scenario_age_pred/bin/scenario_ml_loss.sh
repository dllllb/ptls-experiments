# ContrastiveLoss (positive stronger)
export SC_SUFFIX="loss_contrastive_margin_0.5"
python -m dltranz.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.loss="ContrastiveLoss" \
    params.train.margin=0.5 \
    model_path="../../artifacts/scenario_age_pred/mles__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python -m dltranz.pl_inference \
    model_path="../../artifacts/scenario_age_pred/mles__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

# ContrastiveLoss (negative stronger)
export SC_SUFFIX="loss_contrastive_margin_1.0"
python -m dltranz.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.loss="ContrastiveLoss" \
    params.train.margin=1.0 \
    model_path="../../artifacts/scenario_age_pred/mles__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python -m dltranz.pl_inference \
    model_path="../../artifacts/scenario_age_pred/mles__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

# BinomialDevianceLoss (positive stronger)
export SC_SUFFIX="loss_binomialdeviance"
python -m dltranz.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.loss="BinomialDevianceLoss" \
    params.train.C=1.0 \
    params.train.alpha=1.0 \
    params.train.beta=0.3 \
    model_path="../../artifacts/scenario_age_pred/mles__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python -m dltranz.pl_inference \
    model_path="../../artifacts/scenario_age_pred/mles__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

# TripletLoss
export SC_SUFFIX="loss_triplet"
python -m dltranz.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.loss="TripletLoss" \
    params.train.margin=0.3 \
    params.train.sampling_strategy="HardTriplets" \
    params.train.neg_count=5 \
    model_path="../../artifacts/scenario_age_pred/mles__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python -m dltranz.pl_inference \
    model_path="../../artifacts/scenario_age_pred/mles__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

# HistogramLoss
export SC_SUFFIX="loss_histogramloss"
python -m dltranz.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.loss="HistogramLoss" \
    params.train.num_steps=25 \
    model_path="../../artifacts/scenario_age_pred/mles__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python -m dltranz.pl_inference \
    model_path="../../artifacts/scenario_age_pred/mles__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

# MarginLoss (positive stronger)
export SC_SUFFIX="loss_margin_0.2_beta_0.4"
python -m dltranz.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.loss="MarginLoss" \
    params.train.margin=0.2 \
    params.train.beta=0.4 \
    model_path="../../artifacts/scenario_age_pred/mles__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python -m dltranz.pl_inference \
    model_path="../../artifacts/scenario_age_pred/mles__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

# MarginLoss (negative stronger)
export SC_SUFFIX="loss_margin_0.3_beta_0.6"
python -m dltranz.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.loss="MarginLoss" \
    params.train.margin=0.3 \
    params.train.beta=0.6 \
    model_path="../../artifacts/scenario_age_pred/mles__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python -m dltranz.pl_inference \
    model_path="../../artifacts/scenario_age_pred/mles__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

# Compare
rm results/scenario_age_pred__loss.txt

python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 \
    --conf_extra \
      'report_file: "../results/scenario_age_pred__loss.txt",
      auto_features: ["../data/emb__loss_*.pickle"]'
