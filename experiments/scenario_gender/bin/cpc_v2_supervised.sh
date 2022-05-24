# Train a supervised model and save scores to the file
python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_target

# Fine tune the CPC model in supervised mode and save scores to the file
for i in 20 30 40 50; do
    export split_count=$i
    export SC_SUFFIX="cpc_v2_sub_seq_sampl_strategy_split_count_${split_count}"
    echo "${SC_SUFFIX}"
    python -m ptls.pl_fit_target \
        logger_name=${SC_SUFFIX} \
        params.pretrained.model_path="${hydra:runtime.cwd}/../../artifacts/scenario_gender/$SC_SUFFIX.p" \
        embedding_validation_results.output_path="${hydra:runtime.cwd}/results/$SC_SUFFIX.json" \
        embedding_validation_results.feature_name="cpc_v2_finetuning_split_count_$split_count" \
        --config-dir conf --config-name cpc_v2_pl_fit_finetuning
done


# Compare
rm results/scenario_gender_baselines_supervised_cpc_v2.txt

python -m embeddings_validation \
    --config-dir conf --config-name cpc_v2_embeddings_validation_baselines_supervised +workers=10 +total_cpu_count=20 +local_scheduler=True
