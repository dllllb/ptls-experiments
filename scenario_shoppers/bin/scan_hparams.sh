#!/usr/bin/env bash
DATA_PATH="data/sample"
CV_SPLIT=5
EPOCHS=20
INS=133
EMB=64
export CUDA_VISIBLE_DEVICES=1

for HID in 256 512; do
    TARGET="target_logvar"
    WORK_DIR="lightning_logs/hid${HID}_mse_${TARGET}_in${INS}_$(date +%s)"
    echo "================== ${WORK_DIR} =================="
    python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
        hydra/job_logging=disabled hydra/hydra_logging=disabled \
        data_path=${DATA_PATH} target.col_target=${TARGET} split.cv_split_count=${CV_SPLIT} \
        environment.work_dir=${WORK_DIR}

    python3 pl_trainer.py --config-dir conf --config-name pl_regressor \
        data_path=${DATA_PATH} work_dir=${WORK_DIR} data_module.setup.col_target=${TARGET} \
        ~pl_module.metric_list.jsd pl_module.seq_encoder.hidden_size=${HID} \
        pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS} \
        pl_module.seq_encoder.trx_encoder.embeddings.category.out=${EMB} \
        trainer.max_epochs=${EPOCHS}

    echo
    TARGET="target_dist"
    WORK_DIR="lightning_logs/hid${HID}_pois$((${INS}-1))_${TARGET}_in${INS}_$(date +%s)"
    echo "================== ${WORK_DIR} =================="
    python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
        hydra/job_logging=disabled hydra/hydra_logging=disabled \
        data_path=${DATA_PATH} target.col_target=${TARGET} split.cv_split_count=${CV_SPLIT} \
        environment.work_dir=${WORK_DIR}

    python3 pl_trainer.py --config-dir conf --config-name pl_regressor \
        data_path=${DATA_PATH} work_dir=${WORK_DIR} \
        data_module.distribution_target_size=$((${INS}-1)) data_module.setup.col_target=${TARGET} \
        +pl_module.metric_list.acc.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
        +pl_module.metric_list.auc.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
        +pl_module.metric_list.err.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
        +pl_module.metric_list.jsd.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
        pl_module.loss._target_=torch.nn.PoissonNLLLoss +pl_module.loss.log_input=false \
        pl_module.head.num_classes=$((${INS}-1)) pl_module.head.objective="softplus" \
        pl_module.seq_encoder.hidden_size=${HID} \
        pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS} \
        pl_module.seq_encoder.trx_encoder.embeddings.category.out=${EMB} \
        trainer.max_epochs=${EPOCHS}
done

HID=128
for BATCH in 256 512; do
    TARGET="target_logvar"
    WORK_DIR="lightning_logs/batch${BATCH}_mse_${TARGET}_in${INS}_$(date +%s)"
    echo "================== ${WORK_DIR} =================="
    python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
        hydra/job_logging=disabled hydra/hydra_logging=disabled \
        data_path=${DATA_PATH} target.col_target=${TARGET} split.cv_split_count=${CV_SPLIT} \
        environment.work_dir=${WORK_DIR}

    python3 pl_trainer.py --config-dir conf --config-name pl_regressor \
        data_path=${DATA_PATH} work_dir=${WORK_DIR} \
        data_module.setup.col_target=${TARGET} data_module.train.batch_size=${BATCH} \
        ~pl_module.metric_list.jsd pl_module.seq_encoder.hidden_size=${HID} \
        pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS} \
        pl_module.seq_encoder.trx_encoder.embeddings.category.out=${EMB} \
        trainer.max_epochs=${EPOCHS}

    echo
    TARGET="target_dist"
    WORK_DIR="lightning_logs/batch${BATCH}_pois$((${INS}-1))_${TARGET}_in${INS}_$(date +%s)"
    echo "================== ${WORK_DIR} =================="
    python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
        hydra/job_logging=disabled hydra/hydra_logging=disabled \
        data_path=${DATA_PATH} target.col_target=${TARGET} split.cv_split_count=${CV_SPLIT} \
        environment.work_dir=${WORK_DIR}

    python3 pl_trainer.py --config-dir conf --config-name pl_regressor \
        data_path=${DATA_PATH} work_dir=${WORK_DIR} \
        data_module.distribution_target_size=$((${INS}-1)) \
        data_module.setup.col_target=${TARGET} data_module.train.batch_size=${BATCH} \
        +pl_module.metric_list.acc.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
        +pl_module.metric_list.auc.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
        +pl_module.metric_list.err.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
        +pl_module.metric_list.jsd.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
        pl_module.loss._target_=torch.nn.PoissonNLLLoss +pl_module.loss.log_input=false \
        pl_module.head.num_classes=$((${INS}-1)) pl_module.head.objective="softplus" \
        pl_module.seq_encoder.hidden_size=${HID} \
        pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS} \
        pl_module.seq_encoder.trx_encoder.embeddings.category.out=${EMB} \
        trainer.max_epochs=${EPOCHS}
done
