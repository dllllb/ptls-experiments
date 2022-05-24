for SC_HIDDEN_SIZE in 2400 1600 1200 0800 0480 0224 0160 0096 0064 0032
do
  export SC_SUFFIX="hidden_size_bs_0064_hs_${SC_HIDDEN_SIZE}"
  python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.rnn.hidden_size=${SC_HIDDEN_SIZE} \
    +params.train.batch_size=64 \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_age_pred/age_pred_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name mles_params
done

for SC_HIDDEN_SIZE in 0800 0480 0224 0160 0096 0064 0032
do
  export SC_SUFFIX="hidden_size_bs_0064_hs_${SC_HIDDEN_SIZE}"
  python -m ptls.pl_inference \
    inference_dataloader.loader.batch_size=64 \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_age_pred/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb__$SC_SUFFIX" \
    --config-dir conf --config-name mles_params
done

for SC_HIDDEN_SIZE in 2400 1600 1200
do
  export SC_SUFFIX="hidden_size_bs_0064_hs_${SC_HIDDEN_SIZE}"
  python -m ptls.pl_inference \
    inference_dataloader.loader.batch_size=64 \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_age_pred/age_pred_mlm__$SC_SUFFIX.p" \
    +params.valid.batch_size=256 \
    output.path="${hydra:runtime.cwd}/data/emb__$SC_SUFFIX" \
    --config-dir conf --config-name mles_params
done

# Compare
rm results/scenario_age_pred__hidden_size.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
  --config-dir conf --config-name embeddings_validation_short +workers=10 +total_cpu_count=20 \
  report_file="${hydra:runtime.cwd}/results/scenario_age_pred__hidden_size.txt" \    
  auto_features=["${hydra:runtime.cwd}/data/emb__hidden_size_*.pickle"]
