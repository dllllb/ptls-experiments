# Train a supervised model and save scores to the file
python -m dltranz.pl_fit_target --conf conf/pl_fit_target_rnn.hocon

# Train a special MeLES model for fine-tuning
# it is quite smaller, than one which is used for embeddings extraction, due to insufficiency labeled data to fine-tune a big model.
python -m dltranz.pl_train_module --conf conf/mles_params_for_finetuning.hocon
# Fine tune the MeLES model in supervised mode and save scores to the file
python -m dltranz.pl_fit_target --conf conf/pl_fit_finetuning_on_mles.hocon

# Fine tune the CPC model in supervised mode and save scores to the file
python -m dltranz.pl_fit_target --conf conf/pl_fit_finetuning_on_cpc.hocon

# Train a special MeLES model for fine-tuning
# it is quite smaller, than one which is used for embeddings extraction, due to insufficiency labeled data to fine-tune a big model.
python -m dltranz.pl_train_module --conf conf/rtd_params_for_finetuning.hocon
# Fine tune the RTD model in supervised mode and save scores to the file
python -m dltranz.pl_fit_target --conf conf/pl_fit_finetuning_on_rtd.hocon

cp "../../artifacts/scenario_x5/barlow_twins_model.p" "../../artifacts/scenario_x5/barlow_twins_model_ft.p"
# lightning_logs/bt_tuning_base/version_0/checkpoints/epoch\=20-step\=23750.ckpt
#python -m dltranz.pl_train_module \
#  params.rnn.hidden_size=160 \
#  trainer.max_epochs=100 \
#  model_path="../../artifacts/scenario_x5/barlow_twins_model_for_finetuning.p" \
#  --conf conf/barlow_twins_params.hocon
python -m dltranz.pl_fit_target --conf conf/pl_fit_finetuning_on_barlow_twins.hocon

# Compare
rm results/scenario_x5_baselines_supervised.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_baselines_supervised.hocon --workers 10 --total_cpu_count 20
