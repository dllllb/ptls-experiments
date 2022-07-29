#!/usr/bin/env bash

DATA_PATH="data/sample"
CV_SPLIT=5
EPOCHS=20
INS=133
EMB=64
HID=128
export CUDA_VISIBLE_DEVICES=0

echo
TARGET="target_logvar"
LOGGER_NAME="lightning_logs/${TARGET}_MSE_in${INS}_gpu${CUDA_VISIBLE_DEVICES}_t$(date +%s)"
echo "================== ${LOGGER_NAME} =================="
python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
    hydra/job_logging=disabled hydra/hydra_logging=disabled \
    data_path=${DATA_PATH} target.col_target=${TARGET} split.cv_split_count=${CV_SPLIT} \
    environment.work_dir=${LOGGER_NAME}

python3 pl_trainer.py --config-dir conf --config-name pl_regressor \
    data_path=${DATA_PATH} logger_name=${LOGGER_NAME} \
    data_module.setup.col_target=${TARGET} \
    pl_module.seq_encoder.hidden_size=${HID} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.out=${EMB} \
    trainer.max_epochs=${EPOCHS}

echo
TARGET="target_bin"
LOGGER_NAME="lightning_logs/${TARGET}_MSE_in${INS}_gpu${CUDA_VISIBLE_DEVICES}_t$(date +%s)"
echo "================== ${LOGGER_NAME} =================="
python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
    hydra/job_logging=disabled hydra/hydra_logging=disabled \
    data_path=${DATA_PATH} target.col_target=${TARGET} split.cv_split_count=${CV_SPLIT} \
    environment.work_dir=${LOGGER_NAME}

python3 pl_trainer.py --config-dir conf --config-name pl_regressor \
    data_path=${DATA_PATH} logger_name=${LOGGER_NAME} \
    data_module.setup.col_target=${TARGET} \
    pl_module.seq_encoder.hidden_size=${HID} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.out=${EMB} \
    trainer.max_epochs=${EPOCHS}

echo
TARGET="target_var"
LOGGER_NAME="lightning_logs/${TARGET}_ZILN2_in${INS}_gpu${CUDA_VISIBLE_DEVICES}_t$(date +%s)"
echo "================== ${LOGGER_NAME} =================="
python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
    hydra/job_logging=disabled hydra/hydra_logging=disabled \
    data_path=${DATA_PATH} target.col_target=${TARGET} split.cv_split_count=${CV_SPLIT} \
    environment.work_dir=${LOGGER_NAME}

python3 pl_trainer.py --config-dir conf --config-name pl_regressor \
    data_path=${DATA_PATH} logger_name=${LOGGER_NAME} \
    data_module.setup.col_target=${TARGET} \
    +pl_module.metric_list.acc.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    +pl_module.metric_list.err.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    pl_module.head.num_classes=2 \
    pl_module.seq_encoder.hidden_size=${HID} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.out=${EMB} \
    trainer.max_epochs=${EPOCHS}

echo
TARGET="target_var"
LOGGER_NAME="lightning_logs/${TARGET}_ZILN3_in${INS}_gpu${CUDA_VISIBLE_DEVICES}_t$(date +%s)"
echo "================== ${LOGGER_NAME} =================="
python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
    hydra/job_logging=disabled hydra/hydra_logging=disabled \
    data_path=${DATA_PATH} target.col_target=${TARGET} split.cv_split_count=${CV_SPLIT} \
    environment.work_dir=${LOGGER_NAME}

python3 pl_trainer.py --config-dir conf --config-name pl_regressor \
    data_path=${DATA_PATH} logger_name=${LOGGER_NAME} \
    data_module.setup.col_target=${TARGET} \
    +pl_module.metric_list.acc.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    +pl_module.metric_list.err.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    pl_module.head.num_classes=3 \
    pl_module.seq_encoder.hidden_size=${HID} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.out=${EMB} \
    trainer.max_epochs=${EPOCHS}
