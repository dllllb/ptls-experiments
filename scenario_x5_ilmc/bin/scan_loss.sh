#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

for INS in 31 51 71 101 131; do
    DATA_PATH="data/$((${INS}-1))"

    WORK_DIR="lightning_logs/loss_mse_logvar_in${INS}"
    echo "================== ${WORK_DIR} =================="
    python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
        hydra/job_logging=disabled hydra/hydra_logging=disabled environment.work_dir=${WORK_DIR} \
        data_path=${DATA_PATH} target.col_target="target_logvar"

    python3 pl_trainer.py --config-dir conf --config-name pl_regressor work_dir=${WORK_DIR} \
        data_path=${DATA_PATH} data_module.setup.col_target="target_logvar" \
        ~pl_module.metric_list.jsd \
        pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS}

    echo
    WORK_DIR="lightning_logs/loss_pois$((${INS}-1))_in${INS}"
    echo "================== ${WORK_DIR} =================="
    python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
        hydra/job_logging=disabled hydra/hydra_logging=disabled environment.work_dir=${WORK_DIR} \
        data_path=${DATA_PATH}

    python3 pl_trainer.py --config-dir conf --config-name pl_regressor work_dir=${WORK_DIR} \
        data_path=${DATA_PATH} data_module.distribution_target_size=$((${INS}-1)) \
        +pl_module.metric_list.acc.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
        +pl_module.metric_list.auc.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
        +pl_module.metric_list.err.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
        +pl_module.metric_list.jsd.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
        pl_module.loss._target_=torch.nn.PoissonNLLLoss +pl_module.loss.log_input=false \
        pl_module.head.num_classes=$((${INS}-1)) pl_module.head.objective="softplus" \
        pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS}

    echo
    WORK_DIR="lightning_logs/loss_ziln$((${INS}+2))_in${INS}"
    echo "================== ${WORK_DIR} =================="
    python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
        hydra/job_logging=disabled hydra/hydra_logging=disabled environment.work_dir=${WORK_DIR} \
        data_path=${DATA_PATH}

    python3 pl_trainer.py --config-dir conf --config-name pl_regressor work_dir=${WORK_DIR} \
        data_path=${DATA_PATH} data_module.distribution_target_size=$((${INS}-1)) \
        +pl_module.metric_list.acc.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
        +pl_module.metric_list.auc.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
        +pl_module.metric_list.err.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
        +pl_module.metric_list.jsd.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
        pl_module.head.num_classes=$((${INS}+2)) \
        pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS}
done
