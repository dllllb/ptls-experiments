ulimit -n 32000

tuning_set=\
"400 4 8 0.00005 512 64 4 64000"
export device=0

while IFS=' ' read -r hidden_size n_heads n_layers max_lr max_len batch_size accumulate_grad_batches max_steps
do 
    export logger_name="$hidden_size-$n_heads-$n_layers-$max_lr-$max_len-$batch_size-$accumulate_grad_batches-$max_steps"
    python -m ptls.pl_train_module \
        logger_name=${logger_name} \
        trainer.gpus=${device} \
        pl_module.hidden_size=${hidden_size} \
        pl_module.max_lr=${max_lr} \
        pl_module.seq_encoder.num_attention_heads=${n_heads} \
        pl_module.seq_encoder.num_hidden_layers=${n_layers} \
        data_module.train_data.max_len=${max_len} \
        data_module.valid_data.max_len=${max_len} \
        data_module.train_batch_size=${batch_size} \
        data_module.valid_batch_size=${batch_size} \
        trainer.accumulate_grad_batches=${accumulate_grad_batches} \
        trainer.max_steps=${max_steps} \
        model_path="${hydra:runtime.cwd}/../../artifacts/scenario_alpha_battle/mlmnsp__$logger_name.p" \
      --config-dir conf --config-name mlm_nsp_params
    python -m ptls.pl_inference \
        model_path="${hydra:runtime.cwd}/../../artifacts/scenario_alpha_battle/mlmnsp__$logger_name.p" \
        output.path="${hydra:runtime.cwd}/data/emb_mlmnsp__$logger_name" \
      --config-dir conf --config-name mlm_nsp_params
done <<< $tuning_set

# Compare
rm results/scenario_alpha_battle__hidden_size.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_short +workers=10 +total_cpu_count=20 \
    report_file="${hydra:runtime.cwd}/results/scenario_alpha_battle__hidden_size.txt" \    
    auto_features=["${hydra:runtime.cwd}/data/emb_mlmnsp__*.pickle"]

