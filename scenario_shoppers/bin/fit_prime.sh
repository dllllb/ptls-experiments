#!/usr/bin/env bash

DATA_PATH="data"
CV_SPLIT=5
EPOCHS=20
EMB=64
HID=128
export CUDA_VISIBLE_DEVICES=0

for INS in 133 101 71 41; do
    echo
    TARGET="target_var"
    LOGGER_NAME="lightning_logs/${TARGET}_ZILN3_in${INS}"
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

    echo
    TARGET="target_dist"
    LOGGER_NAME="lightning_logs/${TARGET}_POIS$((${INS}-1))_in${INS}"
    echo "================== ${LOGGER_NAME} =================="
    python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
        hydra/job_logging=disabled hydra/hydra_logging=disabled \
        data_path=${DATA_PATH} target.col_target=${TARGET} split.cv_split_count=${CV_SPLIT} \
        environment.work_dir=${LOGGER_NAME}

    python3 pl_trainer.py --config-dir conf --config-name pl_regressor \
        data_path=${DATA_PATH} logger_name=${LOGGER_NAME} \
        data_module.distribution_target_size=$((${INS}-1)) \
        data_module.setup.col_target=${TARGET} \
        pl_module.metric_list.acc._target_=ptls.frames.supervised.metrics.JSDiv \
        +pl_module.metric_list.acc.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
        +pl_module.metric_list.err.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
        pl_module.loss._target_=torch.nn.PoissonNLLLoss \
        +pl_module.loss.log_input=false \
        pl_module.head.num_classes=$((${INS}-1)) \
        pl_module.head.objective="softplus" \
        pl_module.seq_encoder.hidden_size=${HID} \
        pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS} \
        pl_module.seq_encoder.trx_encoder.embeddings.category.out=${EMB} \
        trainer.max_epochs=${EPOCHS}
done
