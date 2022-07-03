# ReduceLROnPlateau
export SC_SUFFIX="lr_reduce_on_plateau"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.lr_scheduler.ReduceLROnPlateau=true \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/mles__$SC_SUFFIX.p" \
    --config-dir conf --config-name mles_params
python -m ptls.pl_inference \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/mles__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb_mles__$SC_SUFFIX" \
    --config-dir conf --config-name mles_params

# ReduceLROnPlateau x2 epochs
export SC_SUFFIX="lr_reduce_on_plateau_x2epochs"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.lr_scheduler.ReduceLROnPlateau=true \
    params.lr_scheduler.threshold=0.0001 \
    trainer.max_epochs=300 \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/mles__$SC_SUFFIX.p" \
    --config-dir conf --config-name mles_params
python -m ptls.pl_inference \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/mles__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb_mles__$SC_SUFFIX" \
    --config-dir conf --config-name mles_params

# CosineAnnealing
export SC_SUFFIX="lr_cosine_annealing"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.lr_scheduler.n_epoch=150 \
    params.lr_scheduler.CosineAnnealing=true \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/mles__$SC_SUFFIX.p" \
    --config-dir conf --config-name mles_params
python -m ptls.pl_inference \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_rosbank/mles__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb_mles__$SC_SUFFIX" \
    --config-dir conf --config-name mles_params

# Compare
rm results/rm results/scenario_rosbank_lr_schedule.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_short +workers=10 +total_cpu_count=20 \
    report_file="${hydra:runtime.cwd}/results/scenario_rosbank_lr_schedule.txt" \    
    auto_features=["${hydra:runtime.cwd}/data/emb_mles__lr_*.pickle"]
