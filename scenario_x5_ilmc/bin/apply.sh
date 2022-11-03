#!/usr/bin/env bash
INS=71
BMRK_CKPT=10
POIS_CKPT=10
ZILN_CKPT=10
export CUDA_VISIBLE_DEVICES=0
DATA_PATH="data/$((${INS}-1))"

BMRK_DIR="lightning_logs/loss_mse_logvar_in${INS}"
echo "================== ${BMRK_DIR} =================="
python3 pl_trainer.py --config-dir conf --config-name pl_regressor \
    hydra/hydra_logging=disabled hydra.run.dir="." work_dir=${BMRK_DIR} data_path=${DATA_PATH} \
    data_module.setup.col_target="target_logvar" ~pl_module.metric_list.jsd \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS} ckpt=${BMRK_CKPT}

echo
WORK_DIR="lightning_logs/loss_pois$((${INS}-1))_in${INS}"
echo "================== ${WORK_DIR} =================="
python3 pl_trainer.py --config-dir conf --config-name pl_regressor \
    hydra/hydra_logging=disabled hydra.run.dir="." work_dir=${WORK_DIR} data_path=${DATA_PATH} \
    data_module.distribution_target_size=$((${INS}-1)) \
    +pl_module.metric_list.acc.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
    +pl_module.metric_list.auc.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
    +pl_module.metric_list.err.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
    +pl_module.metric_list.jsd.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
    pl_module.loss._target_=torch.nn.PoissonNLLLoss +pl_module.loss.log_input=false \
    pl_module.head.num_classes=$((${INS}-1)) pl_module.head.objective="softplus" \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS} ckpt=${POIS_CKPT}

python3 monte_carlo.py --config-dir conf --config-name pl_regressor \
    hydra/hydra_logging=disabled hydra.run.dir="." work_dir=${WORK_DIR} data_path=${DATA_PATH} \
    pl_module.head.num_classes=$((${INS}-1)) pl_module.head.objective="softplus" \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS} ckpt=${POIS_CKPT}

if [ -f ${WORK_DIR}/monte_carlo.csv ]; then
    rm -f ${WORK_DIR}/monte_carlo_*.csv
    python3 eval_metrics.py --config-dir conf --config-name pl_regressor \
        hydra/hydra_logging=disabled hydra.run.dir="." work_dir=${WORK_DIR} \
        monte_carlo.benchmark.dir=${BMRK_DIR}
fi
echo
WORK_DIR="lightning_logs/loss_ziln$((${INS}+2))_in${INS}"
echo "================== ${WORK_DIR} =================="
python3 pl_trainer.py --config-dir conf --config-name pl_regressor \
    hydra/hydra_logging=disabled hydra.run.dir="." work_dir=${WORK_DIR} data_path=${DATA_PATH} \
    data_module.distribution_target_size=$((${INS}-1)) \
    +pl_module.metric_list.acc.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    +pl_module.metric_list.auc.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    +pl_module.metric_list.err.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    +pl_module.metric_list.jsd.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    pl_module.head.num_classes=$((${INS}+2)) \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS} ckpt=${ZILN_CKPT}

python3 monte_carlo.py --config-dir conf --config-name pl_regressor \
    hydra/hydra_logging=disabled hydra.run.dir="." work_dir=${WORK_DIR} data_path=${DATA_PATH} \
    pl_module.head.num_classes=$((${INS}+2)) \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS} ckpt=${ZILN_CKPT}

if [ -f ${WORK_DIR}/monte_carlo.csv ]; then
    rm -f ${WORK_DIR}/monte_carlo_*.csv
    python3 eval_metrics.py --config-dir conf --config-name pl_regressor \
        hydra/hydra_logging=disabled hydra.run.dir="." work_dir=${WORK_DIR} \
        monte_carlo.benchmark.dir=${BMRK_DIR}
fi
