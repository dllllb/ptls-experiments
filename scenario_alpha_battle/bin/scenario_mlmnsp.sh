ulimit -n 32000
export device=0

tuning_set=\
"400 4 8 0.00008 512 96 2 64000
640 8 8 0.00005 512 64 2 64000"
while IFS=' ' read -r hidden_size n_heads n_layers max_lr max_len batch_size accumulate_grad_batches max_steps
do 
    export logger_name="mlmnsp--$hidden_size-$n_heads-$n_layers-$max_lr-$max_len-$batch_size-$accumulate_grad_batches-$max_steps"

    python -m ptls.pl_train_module \
        logger_name=${logger_name} \
        trainer.gpus=[${device}] \
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
        model_path="models/mlmnsp__$logger_name.p" \
      --config-dir conf --config-name mlm_nsp_params

      python -m ptls.pl_inference \
        +inference.gpus=[${device}] \
        pl_module.hidden_size=${hidden_size} \
        pl_module.seq_encoder.num_attention_heads=${n_heads} \
        pl_module.seq_encoder.num_hidden_layers=${n_layers} \
        inference.batch_size=1024 \
        model_path="models/mlmnsp__$logger_name.p" \
        embed_file_name="emb_mlmnsp_stat__${logger_name}" \
      --config-dir conf --config-name mlm_nsp_params

done <<< $tuning_set

# Compare
rm results/scenario_alpha_battle.txt
rm -r embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_short +workers=10 +total_cpu_count=10 \
    ++report_file="results/scenario_alpha_battle.txt" 
    

