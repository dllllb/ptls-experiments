export SC_SUFFIX="SampleRandom"
export SC_STRATEGY="SampleRandom"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    data_module.train.split_strategy.split_strategy=$SC_STRATEGY \
    data_module.valid.split_strategy.split_strategy=$SC_STRATEGY \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_bowl2019/bowl2019_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name mles_params
python -m ptls.pl_inference \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_bowl2019/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb__$SC_SUFFIX" \
    --config-dir conf --config-name mles_params

export SC_SUFFIX="SampleRandom_short"
export SC_STRATEGY="SampleRandom"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    data_module.train.max_seq_len=600 \
    data_module.valid.max_seq_len=600 \
    data_module.train.split_strategy.split_strategy=$SC_STRATEGY \
    data_module.valid.split_strategy.split_strategy=$SC_STRATEGY \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_bowl2019/bowl2019_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name mles_params
python -m ptls.pl_inference \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_bowl2019/bowl2019_mlm__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb__$SC_SUFFIX" \
    --config-dir conf --config-name mles_params

# Compare
rm results/scenario_bowl2019__subseq_smpl_strategy.txt
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_short +workers=10 +total_cpu_count=20 +local_scheduler=True \
    report_file="${hydra:runtime.cwd}/results/scenario_bowl2019__subseq_smpl_strategy.txt" \    
    auto_features=[
          "${hydra:runtime.cwd}/data/emb__SampleRandom.pickle",
          "${hydra:runtime.cwd}/data/emb__SampleRandom_short.pickle"]'

