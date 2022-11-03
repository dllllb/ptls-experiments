#!/usr/bin/env bash
INS=71
export CUDA_VISIBLE_DEVICES=0

for HID in 128 256 512; do
    WORK_DIR="lightning_logs/hid${HID}_mse_logvar_in${INS}"
    echo "================== ${WORK_DIR} =================="
    python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
        hydra/job_logging=disabled hydra/hydra_logging=disabled environment.work_dir=${WORK_DIR} \
        target.col_target="target_logvar"

    python3 pl_trainer.py --config-dir conf --config-name pl_regressor work_dir=${WORK_DIR} \
        data_module.setup.col_target="target_logvar" \
        ~pl_module.metric_list.jsd \
        pl_module.seq_encoder.hidden_size=${HID} \
        pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS}

    echo
    WORK_DIR="lightning_logs/hid${HID}_pois$((${INS}-1))_in${INS}"
    echo "================== ${WORK_DIR} =================="
    python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
        hydra/job_logging=disabled hydra/hydra_logging=disabled environment.work_dir=${WORK_DIR}

    python3 pl_trainer.py --config-dir conf --config-name pl_regressor work_dir=${WORK_DIR} \
        data_module.distribution_target_size=$((${INS}-1)) \
        +pl_module.metric_list.acc.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
        +pl_module.metric_list.auc.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
        +pl_module.metric_list.err.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
        +pl_module.metric_list.jsd.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
        pl_module.loss._target_=torch.nn.PoissonNLLLoss +pl_module.loss.log_input=false \
        pl_module.head.num_classes=$((${INS}-1)) pl_module.head.objective="softplus" \
        pl_module.seq_encoder.hidden_size=${HID} \
        pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS}
done

for BATCH in 128 256 512; do
    WORK_DIR="lightning_logs/batch${BATCH}_mse_logvar_in${INS}"
    echo "================== ${WORK_DIR} =================="
    python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
        hydra/job_logging=disabled hydra/hydra_logging=disabled environment.work_dir=${WORK_DIR} \
        target.col_target="target_logvar"

    python3 pl_trainer.py --config-dir conf --config-name pl_regressor work_dir=${WORK_DIR} \
        data_module.setup.col_target="target_logvar" \
        data_module.train.batch_size=${BATCH} \
        ~pl_module.metric_list.jsd \
        pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS}

    echo
    WORK_DIR="lightning_logs/batch${BATCH}_pois$((${INS}-1))_in${INS}"
    echo "================== ${WORK_DIR} =================="
    python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
        hydra/job_logging=disabled hydra/hydra_logging=disabled environment.work_dir=${WORK_DIR}

    python3 pl_trainer.py --config-dir conf --config-name pl_regressor work_dir=${WORK_DIR} \
        data_module.distribution_target_size=$((${INS}-1)) \
        data_module.train.batch_size=${BATCH} \
        +pl_module.metric_list.acc.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
        +pl_module.metric_list.auc.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
        +pl_module.metric_list.err.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
        +pl_module.metric_list.jsd.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
        pl_module.loss._target_=torch.nn.PoissonNLLLoss +pl_module.loss.log_input=false \
        pl_module.head.num_classes=$((${INS}-1)) pl_module.head.objective="softplus" \
        pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS}
done
