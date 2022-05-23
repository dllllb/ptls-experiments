#for SC_HIDDEN_SIZE in 1024 0512 0256 0128 0064
for SC_HIDDEN_SIZE in 2048
do
  export SC_SUFFIX="hidden_size__hs_${SC_HIDDEN_SIZE}"
  python -m ptls.pl_train_module \
      logger_name=${SC_SUFFIX} \
      params.rnn.hidden_size=${SC_HIDDEN_SIZE} \
      data_module.train.batch_size=128 \
      data_module.valid.batch_size=128 \
      model_path="${hydra:runtime.cwd}/../../artifacts/scenario_alpha_battle/mlm__$SC_SUFFIX.p" \
      --config-dir conf --config-name mles_params
  python -m ptls.pl_inference \
      model_path="${hydra:runtime.cwd}/../../artifacts/scenario_alpha_battle/mlm__$SC_SUFFIX.p" \
      output.path="${hydra:runtime.cwd}/data/emb_mles__$SC_SUFFIX" \
      --config-dir conf --config-name mles_params
done

for SC_HIDDEN_SIZE in 0128 0064
do
  export SC_SUFFIX="hidden_size__hs_${SC_HIDDEN_SIZE}"
  python -m ptls.pl_train_module \
      logger_name=${SC_SUFFIX} \
      params.rnn.hidden_size=${SC_HIDDEN_SIZE} \
      model_path="${hydra:runtime.cwd}/../../artifacts/scenario_alpha_battle/mlm__$SC_SUFFIX.p" \
      --config-dir conf --config-name mles_params
  python -m ptls.pl_inference \
      model_path="${hydra:runtime.cwd}/../../artifacts/scenario_alpha_battle/mlm__$SC_SUFFIX.p" \
      output.path="${hydra:runtime.cwd}/data/emb_mles__$SC_SUFFIX" \
      --config-dir conf --config-name mles_params
done

# Compare
rm results/scenario_alpha_battle__hidden_size.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_short +workers=10 +total_cpu_count=20 \
    report_file="${hydra:runtime.cwd}/results/scenario_alpha_battle__hidden_size.txt" \    
    auto_features=["${hydra:runtime.cwd}/data/emb_mles__hidden_size_*.pickle"]

