export SC_SUFFIX="subseq_SampleRandom"
export SC_STRATEGY="SampleRandom"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    data_module.train.split_strategy.split_strategy=$SC_STRATEGY \
    data_module.valid.split_strategy.split_strategy=$SC_STRATEGY \
    params.train.batch_size=128 \
    params.valid.batch_size=128 \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_x5/mles__$SC_SUFFIX.p" \
    --config-dir conf --config-name mles_params
python -m ptls.pl_inference \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_x5/mles__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb_mles__$SC_SUFFIX" \
    --config-dir conf --config-name mles_params


export SC_SUFFIX="subseq_SplitRandom"
export SC_STRATEGY="SplitRandom"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    data_module.train.split_strategy.split_strategy=$SC_STRATEGY \
    data_module.valid.split_strategy.split_strategy=$SC_STRATEGY \
    params.train.batch_size=128 \
    params.valid.batch_size=128 \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_x5/mles__$SC_SUFFIX.p" \
    --config-dir conf --config-name mles_params
python -m ptls.pl_inference \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_x5/mles__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb_mles__$SC_SUFFIX" \
    --config-dir conf --config-name mles_params


# Compare
rm results/scenario_x5__subseq_smpl_strategy.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_short +workers=10 +total_cpu_count=20 \
    report_file="${hydra:runtime.cwd}/results/scenario_x5__subseq_smpl_strategy.txt" \    
    auto_features=["${hydra:runtime.cwd}/data/emb_mles__subseq_*.pickle"]
