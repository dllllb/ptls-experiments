# Train a supervised model and save scores to the file
python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_target_rnn

# Train a special MeLES model for fine-tuning
# it is quite smaller, than one which is used for embeddings extraction, due to insufficiency labeled data to fine-tune a big model.
# Fine tune the MeLES model in supervised mode and save scores to the file
python -m ptls.pl_train_module --config-dir conf --config-name mles_params_for_finetuning
python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_finetuning_on_mles

# Fine tune the CPC model in supervised mode and save scores to the file
python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_finetuning_on_cpc

# Train a special MeLES model for fine-tuning
# it is quite smaller, than one which is used for embeddings extraction, due to insufficiency labeled data to fine-tune a big model.
# Fine tune the RTD model in supervised mode and save scores to the file
python -m ptls.pl_train_module --config-dir conf --config-name rtd_params_for_finetuning
python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_finetuning_on_rtd

cp "models/barlow_twins_model.p" "models/barlow_twins_model_ft.p"
# lightning_logs/bt_tuning_base/version_0/checkpoints/epoch\=20-step\=23750.ckpt
#python -m ptls.pl_train_module \
#  params.rnn.hidden_size=160 \
#  trainer.max_epochs=100 \
#  model_path="${hydra:runtime.cwd}/../../artifacts/scenario_x5/barlow_twins_model_for_finetuning.p" \
#  --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_finetuning_on_barlow_twins

# Compare
rm results/scenario_x5_baselines_supervised.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_baselines_supervised +workers=10 +total_cpu_count=20
