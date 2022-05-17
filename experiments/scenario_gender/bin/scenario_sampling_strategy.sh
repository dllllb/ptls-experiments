# AllPositivePair
export SC_SUFFIX="smpl_strategy_AllPositivePair"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.sampling_strategy="AllPositivePair" \
    model_path="../../artifacts/scenario_gender/mles__$SC_SUFFIX.p" \
    --config-dir conf --config-name mles_params
python -m ptls.pl_inference \
    model_path="../../artifacts/scenario_gender/mles__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --config-dir conf --config-name mles_params


# DistanceWeightedPair
export SC_SUFFIX="smpl_strategy_DistanceWeightedPair"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.sampling_strategy="DistanceWeightedPair" \
    params.train.n_samples_from_class=5 \
    model_path="../../artifacts/scenario_gender/mles__$SC_SUFFIX.p" \
    --config-dir conf --config-name mles_params
python -m ptls.pl_inference \
    model_path="../../artifacts/scenario_gender/mles__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --config-dir conf --config-name mles_params

# HardNegativePair
for SC_NEG_COUNT in 2 5 9
do
  export SC_SUFFIX="smpl_strategy_HardNegativePair_neg_count_${SC_NEG_COUNT}"
  python -m ptls.pl_train_module \
      logger_name=${SC_SUFFIX} \
      params.train.sampling_strategy="HardNegativePair" \
      params.train.neg_count=${SC_NEG_COUNT} \
      model_path="../../artifacts/scenario_gender/mles__$SC_SUFFIX.p" \
      --config-dir conf --config-name mles_params
  python -m ptls.pl_inference \
      model_path="../../artifacts/scenario_gender/mles__$SC_SUFFIX.p" \
      output.path="data/emb__$SC_SUFFIX" \
      --config-dir conf --config-name mles_params
done

# Compare
rm results/scenario_gender__smpl_strategy.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_short --workers 10 --total_cpu_count 20 \
    --conf_extra \
      'report_file: "../results/scenario_gender__smpl_strategy.txt",
      auto_features: ["../data/emb__smpl_strategy_*.pickle"]'
