ulimit -n 32000
export device=1

tuning_set=\
"768 12 12 0.0001 512 80 1 64000"
while IFS=' ' read -r n_embd n_head n_layer max_lr max_len batch_size accumulate_grad_batches max_steps
do 
    export logger_name="gpt--$n_embd-$n_head-$n_layer-$max_lr-$max_len-$batch_size-$accumulate_grad_batches-$max_steps"
    python -m ptls.pl_train_module \
      logger_name=${logger_name} \
      trainer.gpus=[${device}] \
      pl_module.max_lr=${max_lr} \
      pl_module.seq_encoder.n_embd=${n_embd} \
      pl_module.seq_encoder.n_head=${n_head} \
      pl_module.seq_encoder.n_layer=${n_layer} \
      data_module.train_data.max_len=${max_len} \
      data_module.valid_data.max_len=${max_len} \
      data_module.train_batch_size=${batch_size} \
      data_module.valid_batch_size=${batch_size} \
      trainer.accumulate_grad_batches=${accumulate_grad_batches} \
      trainer.max_steps=${max_steps} \
      model_path="models/gpt__$logger_name.p" \
    --config-dir conf --config-name gpt_params

    for inference_pooling_strategy in out trx_stat_out out_stat trx_stat
    do 
      python -m ptls.pl_inference \
        +inference.gpus=[${device}] \
        inference.batch_size=32 \
        pl_module.inference_pooling_strategy=${inference_pooling_strategy} \
        pl_module.seq_encoder.n_embd=${n_embd} \
        pl_module.seq_encoder.n_head=${n_head} \
        pl_module.seq_encoder.n_layer=${n_layer} \
        model_path="models/gpt__$logger_name.p" \
        embed_file_name="emb_gpt_${inference_pooling_strategy}__${logger_name}" \
      --config-dir conf --config-name gpt_params
    done
done <<< $tuning_set

# Compare
rm results/scenario_alpha_battle.txt
rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_short +workers=10 +total_cpu_count=30 \
    ++report_file="results/scenario_alpha_battle.txt" 
    

