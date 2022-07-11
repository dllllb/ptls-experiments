# Train a supervised model and save scores to the file
python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_target

# Train a special MeLES model for fine-tuning
# it is quite smaller, than one which is used for embeddings extraction, due to insufficiency labeled data to fine-tune a big model.
python -m ptls.pl_train_module --config-dir conf --config-name mles_params_for_finetuning
# Take the pretrained metric learning model and fine tune it in supervised mode; save scores to the file
python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_finetuning_mles

python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_finetuning_cpc

# Fine tune the RTD model in supervised mode and save scores to the file
python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_finetuning_rtd


# cp "../../artifacts/scenario_age_pred/barlow_twins_model.p" "../../artifacts/scenario_age_pred/barlow_twins_model_for_finetuning.p"
python -m ptls.pl_train_module \
  pl_module.seq_encoder.hidden_size=160 \
  trainer.max_epochs=100 \
  model_path="models/barlow_twins_model_for_finetuning.p" \
  --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_finetuning_barlow_twins

# # Compare
rm results/scenario_age_pred_baselines_supervised.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_baselines_supervised +workers=10 +total_cpu_count=20
