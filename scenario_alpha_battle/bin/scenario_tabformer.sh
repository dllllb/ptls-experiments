ulimit -n 32000
export device=1

tuning_set=\
"32 1024 8 1 480 8 8 0.00005 512 128 1 48000
48 1024 8 1 720 8 8 0.00005 512 96 2 48000
64 512 4 1 960 12 12 0.00005 512 64 4 64000"
while IFS=' ' read -r fe_dim fe_ffd fe_heads fe_layers se_hidden_size se_heads se_layers max_lr max_len batch_size accumulate_grad_batches max_steps
do 
    export logger_name="tabformer--fe-$fe_dim-$fe_ffd-$fe_heads-$fe_layers--se$se_hidden_size-$se_heads-$se_layers--$max_lr-$max_len-$batch_size-$accumulate_grad_batches-$max_steps"
    
    python -m ptls.pl_train_module \
        logger_name=${logger_name} \
        trainer.gpus=[${device}] \
        pl_module.max_lr=${max_lr} \
        pl_module.feature_encoder.emb_dim=${fe_dim} \
        pl_module.feature_encoder.transf_feedforward_dim=${fe_ffd} \
        pl_module.feature_encoder.n_heads=${fe_heads} \
        pl_module.feature_encoder.n_layers=${fe_layers} \
        pl_module.seq_encoder.input_size=${se_hidden_size} \
        pl_module.seq_encoder.num_attention_heads=${se_heads} \
        pl_module.seq_encoder.num_hidden_layers=${se_layers} \
        data_module.train_data.max_len=${max_len} \
        data_module.valid_data.max_len=${max_len} \
        data_module.train_batch_size=${batch_size} \
        data_module.valid_batch_size=${batch_size} \
        trainer.accumulate_grad_batches=${accumulate_grad_batches} \
        trainer.max_steps=${max_steps} \
        model_path="models/tabformer__$logger_name.p" \
      --config-dir conf --config-name tabformer_params

    python -m ptls.pl_inference \
        +inference.gpus=[${device}] \
        pl_module.feature_encoder.emb_dim=${fe_dim} \
        pl_module.feature_encoder.transf_feedforward_dim=${fe_ffd} \
        pl_module.feature_encoder.n_heads=${fe_heads} \
        pl_module.feature_encoder.n_layers=${fe_layers} \
        pl_module.seq_encoder.input_size=${se_hidden_size} \
        pl_module.seq_encoder.num_attention_heads=${se_heads} \
        pl_module.seq_encoder.num_hidden_layers=${se_layers} \
        inference.batch_size=512 \
        model_path="models/tabformer__$logger_name.p" \
        embed_file_name="emb_tabformer_stat__$logger_name" \
      --config-dir conf --config-name tabformer_params

done <<< $tuning_set

# Compare
rm results/scenario_alpha_battle.txt
rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_short +workers=10 +total_cpu_count=20 \
    ++report_file="results/scenario_alpha_battle.txt" 

