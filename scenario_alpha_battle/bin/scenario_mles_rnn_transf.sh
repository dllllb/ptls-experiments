ulimit -n 32000
export device=0

tuning_set=\
"1024 0.001 256 1 16"
while IFS=' ' read -r hidden_size max_lr batch_size accumulate_grad_batches n_epochs
do 
    export logger_name="mlesrnn--$hidden_size-$max_lr-$batch_size-$accumulate_grad_batches-$n_epochs"
    python -m ptls.pl_train_module \
        logger_name=${logger_name} \
        trainer.gpus=[${device}] \
        pl_module.seq_encoder.hidden_size=${hidden_size} \
        pl_module.optimizer_partial.lr=${max_lr} \
        data_module.train_batch_size=${batch_size} \
        data_module.valid_batch_size=${batch_size} \
        +trainer.accumulate_grad_batches=${accumulate_grad_batches} \
        +trainer.deterministic=false \
        trainer.max_epochs=${n_epochs} \
        model_path="models/$logger_name.p" \
      --config-dir conf --config-name mles_params
    python -m ptls.pl_inference \
        +inference.gpus=[${device}] \
        pl_module.seq_encoder.hidden_size=${hidden_size} \
        inference.batch_size=1024 \
        model_path="models/$logger_name.p" \
        embed_file_name="emb__${logger_name}" \
      --config-dir conf --config-name mles_params
done <<< $tuning_set

tuning_set=\
"400 4 8 0.00008 196 1 48000"
while IFS=' ' read -r hidden_size n_heads n_layers max_lr batch_size accumulate_grad_batches max_steps
do 
    export logger_name="mlestransf--$hidden_size-$n_heads-$n_layers-$max_lr-$batch_size-$accumulate_grad_batches-$max_steps"
    python -m ptls.pl_train_module \
        logger_name=${logger_name} \
        trainer.gpus=[${device}] \
        pl_module.seq_encoder.input_size=${hidden_size} \
        pl_module.seq_encoder.num_attention_heads=${n_heads} \
        pl_module.seq_encoder.num_hidden_layers=${n_layers} \
        pl_module.optimizer_partial.lr=${max_lr} \
        data_module.train_batch_size=${batch_size} \
        data_module.valid_batch_size=${batch_size} \
        +trainer.accumulate_grad_batches=${accumulate_grad_batches} \
        trainer.max_steps=${max_steps} \
        model_path="models/$logger_name.p" \
      --config-dir conf --config-name mles_params_transf
    python -m ptls.pl_inference \
        +inference.gpus=[${device}] \
        pl_module.seq_encoder.input_size=${hidden_size} \
        pl_module.seq_encoder.num_attention_heads=${n_heads} \
        pl_module.seq_encoder.num_hidden_layers=${n_layers} \
        inference.batch_size=1024 \
        model_path="models/$logger_name.p" \
        embed_file_name="emb__${logger_name}" \
      --config-dir conf --config-name mles_params_transf
done <<< $tuning_set

# Compare
rm results/scenario_mles_rnn_transf.txt
rm -r embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_short +workers=10 +total_cpu_count=10 \
    ++report_file="results/scenario_mles_rnn_transf.txt" 
    
