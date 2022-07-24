#!/usr/bin/env bash

DATA_PATH="data/sample"
CV_SPLIT=5
MAX_EPOCHS=20
EMB_IN_DIM=133
EMB_OUT_DIM=64
HIDDEN_SIZE=128
export CUDA_VISIBLE_DEVICES=1
TARGET="target_dist"

echo
LOGGER_NAME="${TARGET}_sum_pois_$(date +'%Y%m%d%H%M')"
echo "================== GPU(${CUDA_VISIBLE_DEVICES}) == LOSS(POIS:1) == OUTPUT(${LOGGER_NAME}) =================="
python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
    hydra/job_logging=disabled hydra/hydra_logging=disabled \
    data_path=${DATA_PATH} target.col_target=${TARGET} split.cv_split_count=${CV_SPLIT} \
    environment.work_dir="../lightning_logs/${LOGGER_NAME}"

python3 pl_trainer.py --config-dir conf --config-name pl_regressor \
    data_path=${DATA_PATH} logger_name=${LOGGER_NAME} \
    data_module.distribution_target_size=1 \
    data_module.setup.col_target=${TARGET} \
    +pl_module.metric_list.acc.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
    +pl_module.metric_list.err.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
    pl_module.loss._target_=torch.nn.PoissonNLLLoss \
    +pl_module.loss.log_input=false \
    pl_module.head.objective="softplus" \
    pl_module.seq_encoder.hidden_size=${HIDDEN_SIZE} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${EMB_IN_DIM} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.out=${EMB_OUT_DIM} \
    trainer.max_epochs=${MAX_EPOCHS}

echo
LOGGER_NAME="${TARGET}_pois_$(date +'%Y%m%d%H%M')"
echo "================== GPU(${CUDA_VISIBLE_DEVICES}) == LOSS(POIS:C) == OUTPUT(${LOGGER_NAME}) =================="
python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
    hydra/job_logging=disabled hydra/hydra_logging=disabled \
    data_path=${DATA_PATH} target.col_target=${TARGET} split.cv_split_count=${CV_SPLIT} \
    environment.work_dir="../lightning_logs/${LOGGER_NAME}"

python3 pl_trainer.py --config-dir conf --config-name pl_regressor \
    data_path=${DATA_PATH} logger_name=${LOGGER_NAME} \
    data_module.distribution_target_size=$((${EMB_IN_DIM}-1)) \
    data_module.setup.col_target=${TARGET} \
    pl_module.metric_list.acc._target_=ptls.frames.supervised.metrics.JSDiv \
    +pl_module.metric_list.acc.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
    +pl_module.metric_list.err.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
    pl_module.loss._target_=torch.nn.PoissonNLLLoss \
    +pl_module.loss.log_input=false \
    pl_module.head.num_classes=$((${EMB_IN_DIM}-1)) \
    pl_module.head.objective="softplus" \
    pl_module.seq_encoder.hidden_size=${HIDDEN_SIZE} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${EMB_IN_DIM} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.out=${EMB_OUT_DIM} \
    trainer.max_epochs=${MAX_EPOCHS}

echo
LOGGER_NAME="${TARGET}_mult_$(date +'%Y%m%d%H%M')"
echo "================== GPU(${CUDA_VISIBLE_DEVICES}) == LOSS(MULT:C+1) == OUTPUT(${LOGGER_NAME}) =================="
python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
    hydra/job_logging=disabled hydra/hydra_logging=disabled \
    data_path=${DATA_PATH} target.col_target=${TARGET} split.cv_split_count=${CV_SPLIT} \
    environment.work_dir="../lightning_logs/${LOGGER_NAME}"

python3 pl_trainer.py --config-dir conf --config-name pl_regressor \
    data_path=${DATA_PATH} logger_name=${LOGGER_NAME} \
    data_module.distribution_target_size=$((${EMB_IN_DIM}-1)) \
    data_module.setup.col_target=${TARGET} \
    pl_module.metric_list.acc._target_=ptls.frames.supervised.metrics.JSDiv \
    ~pl_module.metric_list.err \
    pl_module.head.num_classes=${EMB_IN_DIM} \
    pl_module.seq_encoder.hidden_size=${HIDDEN_SIZE} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${EMB_IN_DIM} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.out=${EMB_OUT_DIM} \
    trainer.max_epochs=${MAX_EPOCHS}

echo
LOGGER_NAME="${TARGET}_ziln_$(date +'%Y%m%d%H%M')"
echo "================== GPU(${CUDA_VISIBLE_DEVICES}) == LOSS(ZILN:C+3) == OUTPUT(${LOGGER_NAME}) =================="
python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
    hydra/job_logging=disabled hydra/hydra_logging=disabled \
    data_path=${DATA_PATH} target.col_target=${TARGET} split.cv_split_count=${CV_SPLIT} \
    environment.work_dir="../lightning_logs/${LOGGER_NAME}"

python3 pl_trainer.py --config-dir conf --config-name pl_regressor \
    data_path=${DATA_PATH} logger_name=${LOGGER_NAME} \
    data_module.distribution_target_size=$((${EMB_IN_DIM}-1)) \
    data_module.setup.col_target=${TARGET} \
    pl_module.metric_list.acc._target_=ptls.frames.supervised.metrics.JSDiv \
    +pl_module.metric_list.err.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    pl_module.head.num_classes=$((${EMB_IN_DIM}+2)) \
    pl_module.seq_encoder.hidden_size=${HIDDEN_SIZE} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${EMB_IN_DIM} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.out=${EMB_OUT_DIM} \
    trainer.max_epochs=${MAX_EPOCHS}
