# Train a supervised model and save scores to the file
python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_target

# Fine tune the MeLES model in supervised mode and save scores to the file
python -m ptls.pl_train_module \
  pl_module.seq_encoder.hidden_size=256 \
  +pl_module.loss='{_target_: ptls.frames.coles.losses.MarginLoss, margin: 0.2, beta: 0.4}' \
  +pl_module.loss.pair_selector='{_target_: ptls.frames.coles.sampling_strategies.HardNegativePairSelector, neg_count: 5}' \
  ~pl_module.loss.sampling_strategy \
  model_path="models/mles_model_for_finetuning.p" \
  --config-dir conf --config-name mles_params

python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_finetuning_mles

# Fine tune the CPC model in supervised mode and save scores to the file
python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_finetuning_cpc

# Fine tune the RTD model in supervised mode and save scores to the file
python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_finetuning_rtd

# Fine tune the MeLES model in supervised mode and save scores to the file
python -m ptls.pl_train_module \
  pl_module.seq_encoder.hidden_size=256 \
  model_path="models/barlow_twins_model_for_finetuning.p" \
  --config-dir conf --config-name barlow_twins_params
# Fine tune the RTD model in supervised mode and save scores to the file
python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_finetuning_barlow_twins

# Compare
rm results/scenario_gender_baselines_supervised.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_baselines_supervised +workers=10 +total_cpu_count=20
