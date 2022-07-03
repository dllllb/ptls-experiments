# LSTM encoder
export SC_SUFFIX="encoder_lstm"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.rnn.type="lstm" \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_x5/mles__$SC_SUFFIX.p" \
    --config-dir conf --config-name mles_params
python -m ptls.pl_inference \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_x5/mles__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb_mles__$SC_SUFFIX" \
    --config-dir conf --config-name mles_params

# Transformer encoder
export SC_SUFFIX="encoder_transf"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.model_type="transf" \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_x5/mles__$SC_SUFFIX.p" \
    --config-dir conf --config-name mles_params
python -m ptls.pl_inference \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_x5/mles__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb_mles__$SC_SUFFIX" \
    --config-dir conf --config-name mles_params

# Compare
rm results/scenario_x5__encoder_types.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_short +workers=10 +total_cpu_count=20 \
    report_file="${hydra:runtime.cwd}/results/scenario_x5__encoder_types.txt" \    
    auto_features=["${hydra:runtime.cwd}/data/emb_mles__encoder_*.pickle"]
