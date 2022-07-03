# LSTM encoder
export SC_SUFFIX="encoder_lstm"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.rnn.type="lstm" \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_gender/gender_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name mles_params
python -m ptls.pl_inference \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_gender/gender_mlm__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb__$SC_SUFFIX" \
    --config-dir conf --config-name mles_params
#
export SC_SUFFIX="encoder_transf"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.encoder_type="transf" \
    params.train.batch_size=128 \
    params.transf.train_starter=true \
    params.transf.dropout=0.1 \
    params.transf.max_seq_len=800 \
    params.transf.n_heads=8 \
    params.transf.input_size=96 \
    params.transf.dim_hidden=96 \
    params.transf.n_layers=6 \
    params.transf.shared_layers=false \
    params.transf.use_after_mask=false \
    params.transf.use_positional_encoding=false \
    params.transf.use_src_key_padding_mask=false \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_gender/gender_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name mles_params
python -m ptls.pl_inference \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_gender/gender_mlm__$SC_SUFFIX.p" \
    inference_dataloader.loader.batch_size=32 \
    output.path="${hydra:runtime.cwd}/data/emb__$SC_SUFFIX" \
    --config-dir conf --config-name mles_params


# Compare
rm results/scenario_gender__encoder_types.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_short +workers=10 +total_cpu_count=20 \
    report_file="${hydra:runtime.cwd}/results/scenario_gender__encoder_types.txt" \    
    auto_features=["${hydra:runtime.cwd}/data/emb__encoder_*.pickle"]
