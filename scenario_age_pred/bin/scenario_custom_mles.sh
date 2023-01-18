ulimit -n 32000
export device=1

tuning_set=\
"512 4 4 rezero linear-flow 0.0001 64 1 60
512 8 8 rezero linear-flow 0.0001 64 1 60
512 4 4 default quadratic 0.0001 64 1 60
1024 4 4 default quadratic 0.0001 48 2 60
512 4 4 rezero quadratic pre 0.0001 256 1 80
512 4 4 rezero quadratic post 0.0001 256 1 80
768 8 8 rezero quadratic pre 0.0001 64 2 80
512 4 4 rezero quadratic pre 0.0001 256 1 80
512 4 4 rezero quadratic post 0.0001 256 1 80
512 4 4 default linear-flow false 0.0001 256 1 80
512 4 4 default linear-flow false 0.0001 256 1 80"

while IFS=' ' read -r hidden_size n_heads n_layers attn_block layer_norm self_attn max_lr batch_size accumulate_grad_batches n_epochs
do 
    export logger_name="mlescustom--$hidden_size-$n_heads-$n_layers-$attn_block-$layer_norm-$self_attn-$max_lr-$batch_size-$accumulate_grad_batches-$n_epochs"
    python -m ptls.pl_train_module \
        pl_module.seq_encoder.input_size=${hidden_size} \
        pl_module.seq_encoder.num_hidden_layers=${n_layers} \
        pl_module.seq_encoder.num_attention_heads=${n_heads} \
        pl_module.seq_encoder.attn_block_mode=${attn_block} \
        pl_module.seq_encoder.self_attn_mode=${self_attn} \
        pl_module.seq_encoder.layer_norm=${layer_norm} \
        pl_module.optimizer_partial.lr=${max_lr} \
        data_module.train_batch_size=${batch_size} \
        data_module.valid_batch_size=${batch_size} \
        trainer.max_epochs=${n_epochs} \
        trainer.gpus=[${device}] \
        trainer.accumulate_grad_batches=${accumulate_grad_batches} \
        model_path="models/${logger_name}.p" \
        logger_name=${logger_name} \
      --config-dir conf --config-name custom_mles_params
    python -m ptls.pl_inference    \
        pl_module.seq_encoder.input_size=${hidden_size} \
        pl_module.seq_encoder.num_hidden_layers=${n_layers} \
        pl_module.seq_encoder.num_attention_heads=${n_heads} \
        inference.batch_size=64 \
        +inference.gpus=[${device}] \
        model_path="models/${logger_name}.p" \
        embed_file_name="emb_${logger_name}" \
      --config-dir conf --config-name custom_mles_params
done <<< $tuning_set

# Compare
rm results/scenario_age_pred_custom_unsupervised.txt
rm -r embeddings_validation.work/
python -m embeddings_validation \
   --config-dir conf --config-name embeddings_validation_custom_unsupervised +workers=10 +total_cpu_count=20 \
   ++report_file="results/scenario_age_pred_custom_unsupervised.txt" 
less -S results/scenario_age_pred_custom_unsupervised.txt
