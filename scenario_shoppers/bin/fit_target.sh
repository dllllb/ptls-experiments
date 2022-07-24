#!/usr/bin/env bash

DATA_PATH="data/sample"
CV_SPLIT=5
MAX_EPOCHS=20
EMB_IN_DIM=133
EMB_OUT_DIM=64
HIDDEN_SIZE=128
export CUDA_VISIBLE_DEVICES=0

echo
TARGET="target_logvar"
LOGGER_NAME="${TARGET}_mse_$(date +'%Y%m%d%H%M')"
echo "================== GPU(${CUDA_VISIBLE_DEVICES}) == LOSS(MSE:1) == OUTPUT(${LOGGER_NAME}) =================="
python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
    hydra/job_logging=disabled hydra/hydra_logging=disabled \
    data_path=${DATA_PATH} target.col_target=${TARGET} split.cv_split_count=${CV_SPLIT} \
    environment.work_dir="../lightning_logs/${LOGGER_NAME}"

python3 pl_trainer.py --config-dir conf --config-name pl_regressor \
    data_path=${DATA_PATH} logger_name=${LOGGER_NAME} \
    data_module.setup.col_target=${TARGET} \
    pl_module.seq_encoder.hidden_size=${HIDDEN_SIZE} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${EMB_IN_DIM} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.out=${EMB_OUT_DIM} \
    trainer.max_epochs=${MAX_EPOCHS}

echo
TARGET="target_bin"
LOGGER_NAME="${TARGET}_mse_$(date +'%Y%m%d%H%M')"
echo "================== GPU(${CUDA_VISIBLE_DEVICES}) == LOSS(MSE:1) == OUTPUT(${LOGGER_NAME}) =================="
python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
    hydra/job_logging=disabled hydra/hydra_logging=disabled \
    data_path=${DATA_PATH} target.col_target=${TARGET} split.cv_split_count=${CV_SPLIT} \
    environment.work_dir="../lightning_logs/${LOGGER_NAME}"

python3 pl_trainer.py --config-dir conf --config-name pl_regressor \
    data_path=${DATA_PATH} logger_name=${LOGGER_NAME} \
    data_module.setup.col_target=${TARGET} \
    pl_module.seq_encoder.hidden_size=${HIDDEN_SIZE} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${EMB_IN_DIM} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.out=${EMB_OUT_DIM} \
    trainer.max_epochs=${MAX_EPOCHS}

echo
TARGET="target_var"
LOGGER_NAME="${TARGET}_ziln2_$(date +'%Y%m%d%H%M')"
echo "================== GPU(${CUDA_VISIBLE_DEVICES}) == LOSS(ZILN:2) == OUTPUT(${LOGGER_NAME}) =================="
python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
    hydra/job_logging=disabled hydra/hydra_logging=disabled \
    data_path=${DATA_PATH} target.col_target=${TARGET} split.cv_split_count=${CV_SPLIT} \
    environment.work_dir="../lightning_logs/${LOGGER_NAME}"

python3 pl_trainer.py --config-dir conf --config-name pl_regressor \
    data_path=${DATA_PATH} logger_name=${LOGGER_NAME} \
    data_module.setup.col_target=${TARGET} \
    +pl_module.metric_list.acc.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    +pl_module.metric_list.err.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    pl_module.head.num_classes=2 \
    pl_module.seq_encoder.hidden_size=${HIDDEN_SIZE} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${EMB_IN_DIM} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.out=${EMB_OUT_DIM} \
    trainer.max_epochs=${MAX_EPOCHS}

echo
TARGET="target_var"
LOGGER_NAME="${TARGET}_ziln3_$(date +'%Y%m%d%H%M')"
echo "================== GPU(${CUDA_VISIBLE_DEVICES}) == LOSS(ZILN:3) == OUTPUT(${LOGGER_NAME}) =================="
python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
    hydra/job_logging=disabled hydra/hydra_logging=disabled \
    data_path=${DATA_PATH} target.col_target=${TARGET} split.cv_split_count=${CV_SPLIT} \
    environment.work_dir="../lightning_logs/${LOGGER_NAME}"

python3 pl_trainer.py --config-dir conf --config-name pl_regressor \
    data_path=${DATA_PATH} logger_name=${LOGGER_NAME} \
    data_module.setup.col_target=${TARGET} \
    +pl_module.metric_list.acc.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    +pl_module.metric_list.err.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    pl_module.head.num_classes=3 \
    pl_module.seq_encoder.hidden_size=${HIDDEN_SIZE} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${EMB_IN_DIM} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.out=${EMB_OUT_DIM} \
    trainer.max_epochs=${MAX_EPOCHS}
