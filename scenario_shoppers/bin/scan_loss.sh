#!/usr/bin/env bash
INS=111
export CUDA_VISIBLE_DEVICES=0

WORK_DIR="lightning_logs/loss_mse_logvar_in${INS}_$(date +%s)"
echo "================== ${WORK_DIR} =================="
python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
    hydra/job_logging=disabled hydra/hydra_logging=disabled environment.work_dir=${WORK_DIR} \
    target.col_target="target_logvar"

python3 pl_trainer.py --config-dir conf --config-name pl_regressor work_dir=${WORK_DIR} \
    data_module.setup.col_target="target_logvar" \
    ~pl_module.metric_list.jsd \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS}

echo
WORK_DIR="lightning_logs/loss_mse_binvar_in${INS}_$(date +%s)"
echo "================== ${WORK_DIR} =================="
python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
    hydra/job_logging=disabled hydra/hydra_logging=disabled environment.work_dir=${WORK_DIR} \
    target.col_target="target_bin"

python3 pl_trainer.py --config-dir conf --config-name pl_regressor work_dir=${WORK_DIR} \
    data_module.setup.col_target="target_bin" \
    ~pl_module.metric_list.jsd \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS}

echo
WORK_DIR="lightning_logs/loss_ziln2_var_in${INS}_$(date +%s)"
echo "================== ${WORK_DIR} =================="
python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
    hydra/job_logging=disabled hydra/hydra_logging=disabled environment.work_dir=${WORK_DIR} \
    target.col_target="target_var"

python3 pl_trainer.py --config-dir conf --config-name pl_regressor work_dir=${WORK_DIR} \
    data_module.setup.col_target="target_var" \
    +pl_module.metric_list.acc.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    +pl_module.metric_list.auc.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    +pl_module.metric_list.err.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    ~pl_module.metric_list.jsd \
    pl_module.head.num_classes=2 \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS}

echo
WORK_DIR="lightning_logs/loss_ziln3_var_in${INS}_$(date +%s)"
echo "================== ${WORK_DIR} =================="
python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
    hydra/job_logging=disabled hydra/hydra_logging=disabled environment.work_dir=${WORK_DIR} \
    target.col_target="target_var"

python3 pl_trainer.py --config-dir conf --config-name pl_regressor work_dir=${WORK_DIR} \
    data_module.setup.col_target="target_var" \
    +pl_module.metric_list.acc.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    +pl_module.metric_list.auc.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    +pl_module.metric_list.err.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    ~pl_module.metric_list.jsd \
    pl_module.head.num_classes=3 \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS}

echo
WORK_DIR="lightning_logs/loss_pois$((${INS}-1))_in${INS}_$(date +%s)"
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
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS}

echo
WORK_DIR="lightning_logs/loss_ziln$((${INS}+2))_in${INS}_$(date +%s)"
echo "================== ${WORK_DIR} =================="
python3 -m embeddings_validation --config-dir conf --config-name ev_regressor \
    hydra/job_logging=disabled hydra/hydra_logging=disabled environment.work_dir=${WORK_DIR}

python3 pl_trainer.py --config-dir conf --config-name pl_regressor work_dir=${WORK_DIR} \
    data_module.distribution_target_size=$((${INS}-1)) \
    +pl_module.metric_list.acc.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    +pl_module.metric_list.auc.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    +pl_module.metric_list.err.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    +pl_module.metric_list.jsd.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    pl_module.head.num_classes=$((${INS}+2)) \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS}
