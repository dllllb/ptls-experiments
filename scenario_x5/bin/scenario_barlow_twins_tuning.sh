export SC_SUFFIX="bt_tuning_base"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    data_module.train.batch_size=256 \
    params.train.lr=0.002 \
    params.lr_scheduler.step_size=3 \
    trainer.max_epochs=100 \
    params.train.checkpoints_every_n_val_epochs=1 trainer.checkpoint_callback=none\
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_x5/gender_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name barlow_twins_params

export SC_SUFFIX="bt_tuning_v01"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    data_module.train.batch_size=1024 \
    params.train.lr=0.002 \
    params.lr_scheduler.step_size=5 \
    trainer.max_epochs=150 \
    params.train.checkpoints_every_n_val_epochs=1 trainer.checkpoint_callback=none\
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_x5/gender_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name barlow_twins_params

export SC_SUFFIX="bt_tuning_v02"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    data_module.train.batch_size=1024 \
    params.train.lr=0.001 \
    params.lr_scheduler.step_size=5 \
    trainer.max_epochs=150 \
    params.train.checkpoints_every_n_val_epochs=1 trainer.checkpoint_callback=none\
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_x5/gender_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name barlow_twins_params

export SC_SUFFIX="bt_tuning_v03"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    data_module.train.batch_size=512 \
    params.train.lr=0.002 \
    params.lr_scheduler.step_size=3 \
    trainer.max_epochs=100 \
    params.train.checkpoints_every_n_val_epochs=1 trainer.checkpoint_callback=none\
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_x5/gender_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name barlow_twins_params


export SC_SUFFIX="bt_tuning_v01"; export SC_VERSION=4
export SC_SUFFIX="bt_tuning_v02"; export SC_VERSION=1

ls "lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/"
# ep = 0; st = 282; {i: (st + 1) // (ep + 1) * (i + 1) - 1 for i in range(ep, 600, 1)}

python -m ptls.pl_inference     inference_dataloader.loader.batch_size=1000 \
    model_path="${hydra:runtime.cwd}/lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=0-step\=282.ckpt" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}_000" \
    --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_inference     inference_dataloader.loader.batch_size=1000 \
    model_path="${hydra:runtime.cwd}/lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=9-step\=2829.ckpt" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}_009" \
    --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_inference     inference_dataloader.loader.batch_size=1000 \
    model_path="${hydra:runtime.cwd}/lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=19-step\=5659.ckpt" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}_019" \
    --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_inference     inference_dataloader.loader.batch_size=1000 \
    model_path="${hydra:runtime.cwd}/lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=29-step\=8489.ckpt" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}_029" \
    --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_inference     inference_dataloader.loader.batch_size=1000 \
    model_path="${hydra:runtime.cwd}/lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=39-step\=11319.ckpt" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}_039" \
    --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_inference     inference_dataloader.loader.batch_size=1000 \
    model_path="${hydra:runtime.cwd}/lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=49-step\=14149.ckpt" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}_049" \
    --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_inference     inference_dataloader.loader.batch_size=1000 \
    model_path="${hydra:runtime.cwd}/lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=59-step\=16979.ckpt" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}_059" \
    --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_inference     inference_dataloader.loader.batch_size=1000 \
    model_path="${hydra:runtime.cwd}/lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=69-step\=19809.ckpt" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}_069" \
    --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_inference     inference_dataloader.loader.batch_size=1000 \
    model_path="${hydra:runtime.cwd}/lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=79-step\=22639.ckpt" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}_079" \
    --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_inference     inference_dataloader.loader.batch_size=1000 \
    model_path="${hydra:runtime.cwd}/lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=89-step\=25469.ckpt" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}_089" \
    --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_inference     inference_dataloader.loader.batch_size=1000 \
    model_path="${hydra:runtime.cwd}/lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=99-step\=28299.ckpt" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}_099" \
    --config-dir conf --config-name barlow_twins_params


rm results/res_bt_tuning.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_short +workers=4 +total_cpu_count=20 \
    report_file="${hydra:runtime.cwd}/results/res_bt_tuning.txt" \    
    auto_features=["${hydra:runtime.cwd}/data/emb__bt_tuning_*.pickle", "${hydra:runtime.cwd}/data/barlow_twins_embeddings.pickle"]
less -S results/res_bt_tuning.txt

