# Train a supervised model and save scores to the file
python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_target

# Fine tune the MeLES model in supervised mode and save scores to the file
python -m ptls.pl_train_module \
    params.rnn.type="gru" params.rnn.hidden_size=512 params.train.n_epoch=50 \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/mles_model_for_finetuning.p" \
    --config-dir conf --config-name mles_params

python -m ptls.pl_fit_target \
    params.rnn.type="gru" params.rnn.hidden_size=512 \
    params.pretrained_model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/mles_model_for_finetuning.p" \
    --config-dir conf --config-name pl_fit_finetuning_mles

# Fine tune the CPC model in supervised mode and save scores to the file
python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_finetuning_cpc

# Fine tune the NSP and RTD model in supervised mode and save scores to the file
python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_finetuning_nsp
python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_finetuning_rtd

#cp "../../artifacts/scenario_rosbank/barlow_twins_model.p" "../../artifacts/scenario_rosbank/barlow_twins_model_for_finetuning.p"
python -m ptls.pl_train_module \
  params.rnn.type="gru" params.rnn.hidden_size=512 \
  model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/barlow_twins_model_for_finetuning.p" \
  trainer.max_epochs=50 \
  --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_finetuning_barlow_twins


# Compare
rm results/scenario_rosbank_baselines_supervised.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_baselines_supervised +workers=10 +total_cpu_count=20
