# Prepare agg feature encoder and take embeddings; inference
python -m ptls.pl_train_module --config-dir conf --config-name agg_features_params
python -m ptls.pl_inference --config-dir conf --config-name agg_features_params

# Train the Contrastive Predictive Coding (CPC_V2) model; inference
for i in 20 30 40 50; do
    let min_seq_len=$i*5
    export split_count=$i
    export SC_SUFFIX="cpc_v2_sub_seq_sampl_strategy_split_count_${split_count}"
    echo "${SC_SUFFIX}"

    python -m ptls.pl_train_module \
        logger_name=${SC_SUFFIX} \
        data_module.train.min_seq_len=$min_seq_len \
        data_module.train.split_strategy.split_count=$split_count \
        \
        data_module.valid.min_seq_len=$min_seq_len \
        data_module.valid.split_strategy.split_count=$split_count \
        model_path="${hydra:runtime.cwd}/../../artifacts/scenario_bowl2019/$SC_SUFFIX.p" \
        --config-dir conf --config-name cpc_v2_params

    python -m ptls.pl_inference \
        model_path="${hydra:runtime.cwd}/../../artifacts/scenario_bowl2019/$SC_SUFFIX.p" \
        output.path="${hydra:runtime.cwd}/data/emb__$SC_SUFFIX" \
        --config-dir conf --config-name cpc_v2_params
done

rm results/scenario_bowl_baselines_unsupervised_cpc_v2.txt
python -m embeddings_validation \
    --config-dir conf --config-name cpc_v2_embeddings_validation_baselines_unsupervised +workers=10 +total_cpu_count=20 +local_scheduler=True \
      'auto_features: ["${hydra:runtime.cwd}/data/emb__cpc_v2_sub_seq_sampl_strategy*.pickle"]'