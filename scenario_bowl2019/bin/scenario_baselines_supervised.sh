# Train a supervised model and save scores to the file
python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_target

# Fine tune the MeLES model in supervised mode and save scores to the file
python -m ptls.pl_train_module \
    pl_module.loss.sampling_strategy.neg_count=5 \
    model_path="models/mles_model_ft.p" \
    --config-dir conf --config-name mles_params
python -m ptls.pl_fit_target \
    pretrained_encoder_path="models/mles_model_ft.p" \
    +data_module.train_drop_last=true \
    --config-dir conf --config-name pl_fit_finetuning_mles

# Train a special CPC model for fine-tuning
# it is quite smaller, than one which is used for embeddings extraction, due to insufficiency labeled data to fine-tune a big model.
python -m ptls.pl_train_module --config-dir conf --config-name cpc_params_for_finetuning
python -m ptls.pl_fit_target --config-dir conf --config-name pl_fit_finetuning_cpc

# Fine tune the RTD model in supervised mode and save scores to the file
python -m ptls.pl_fit_target +data_module.train_drop_last=true --config-dir conf --config-name pl_fit_finetuning_rtd

cp "models/barlow_twins_model.p" "models/barlow_twins_model_ft.p"
# Fine tune the RTD model in supervised mode and save scores to the file
python -m ptls.pl_fit_target +data_module.train_drop_last=true --config-dir conf --config-name pl_fit_finetuning_barlow_twins


# Compare
rm results/scenario_bowl2019_baselines_supervised.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_baselines_supervised +workers=10 +total_cpu_count=20 +local_scheduler=True
