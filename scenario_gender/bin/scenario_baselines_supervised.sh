# Train a supervised model and save scores to the file
python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_target

# Fine tune the MeLES model in supervised mode and save scores to the file
python -m ptls.pl_train_module \
  params.rnn.hidden_size=256 \
  params.train.loss="MarginLoss" params.train.margin=0.2 params.train.beta=0.4 \
  model_path="${hydra:runtime.cwd}/../../artifacts/scenario_gender/mles_model_for_finetuning.p" \
  --config-dir conf --config-name mles_params

python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_finetuning_mles

# Fine tune the CPC model in supervised mode and save scores to the file
python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_finetuning_cpc

# Fine tune the RTD model in supervised mode and save scores to the file
python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_finetuning_rtd

# Fine tune the MeLES model in supervised mode and save scores to the file
python -m ptls.pl_train_module \
  params.rnn.hidden_size=256 \
  model_path="${hydra:runtime.cwd}/../../artifacts/scenario_gender/barlow_twins_model_for_finetuning.p" \
  --config-dir conf --config-name barlow_twins_params
# Fine tune the RTD model in supervised mode and save scores to the file
python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_finetuning_barlow_twins

# Compare
rm results/scenario_gender_baselines_supervised.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_baselines_supervised +workers=10 +total_cpu_count=20
