for PRJ_SIZE in 256 128 064
do
    for RNN_SIZE in 1600 0800 0400
    do
        export SC_SUFFIX="projection_head_rnn${RNN_SIZE}_prh${PRJ_SIZE}"
        python -m ptls.pl_train_module \
            logger_name=${SC_SUFFIX} \
            params.rnn.hidden_size=${RNN_SIZE} \
            "params.head_layers=[[Linear, {in_features: ${RNN_SIZE}, out_features: ${PRJ_SIZE}}], [BatchNorm1d, {num_features: ${PRJ_SIZE}}], [ReLU, {}], [Linear, {in_features: ${PRJ_SIZE}, out_features: ${PRJ_SIZE}}], [NormEncoder, {}]]" \
            model_path="${hydra:runtime.cwd}/../../artifacts/scenario_age_pred/age_pred_mlm__$SC_SUFFIX.p" \
            --config-dir conf --config-name mles_proj_head_params
        python -m ptls.pl_inference \
            model_path="${hydra:runtime.cwd}/../../artifacts/scenario_age_pred/age_pred_mlm__$SC_SUFFIX.p" \
            output.path="${hydra:runtime.cwd}/data/emb__$SC_SUFFIX" \
            --config-dir conf --config-name mles_proj_head_params
    done
done

# Compare
rm results/scenario_age_pred__proj_head.txt

python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_short +workers=10 +total_cpu_count=20 \
    report_file="${hydra:runtime.cwd}/results/scenario_age_pred__proj_head.txt" \    
    auto_features=["${hydra:runtime.cwd}/data/emb__projection_head_*.pickle"]
