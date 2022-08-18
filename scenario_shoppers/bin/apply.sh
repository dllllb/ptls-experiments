#!/usr/bin/env bash
INS=111
BENCHMARK_CKPT=10
BENCHMARK_DIR="lightning_logs/benchmark"
export CUDA_VISIBLE_DEVICES=0

echo "================== ${BENCHMARK_DIR} =================="
python3 pl_trainer.py --config-dir conf --config-name pl_regressor \
    hydra/hydra_logging=disabled hydra.run.dir="." work_dir=${BENCHMARK_DIR} \
    data_module.setup.col_target="target_logvar" ~pl_module.metric_list.jsd \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS} \
    ckpt=${BENCHMARK_CKPT}

echo
CKPT=10
WORK_DIR="lightning_logs/"
echo "================== ${WORK_DIR} =================="
python3 pl_trainer.py --config-dir conf --config-name pl_regressor \
    hydra/hydra_logging=disabled hydra.run.dir="." work_dir=${WORK_DIR} \
    data_module.distribution_target_size=$((${INS}-1)) \
    +pl_module.metric_list.acc.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
    +pl_module.metric_list.auc.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
    +pl_module.metric_list.err.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
    +pl_module.metric_list.jsd.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
    pl_module.loss._target_=torch.nn.PoissonNLLLoss +pl_module.loss.log_input=false \
    pl_module.head.num_classes=$((${INS}-1)) pl_module.head.objective="softplus" \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS} \
    ckpt=${CKPT}

python3 monte_carlo.py --config-dir conf --config-name pl_regressor \
    hydra/hydra_logging=disabled hydra.run.dir="." work_dir=${WORK_DIR} \
    pl_module.head.num_classes=$((${INS}-1)) pl_module.head.objective="softplus" \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS} \
    ckpt=${CKPT}

if [ -f ${WORK_DIR}/monte_carlo.csv ]; then
    rm -f ${WORK_DIR}/monte_carlo_*.csv
    python3 eval_metrics.py --config-dir conf --config-name pl_regressor \
        hydra/hydra_logging=disabled hydra.run.dir="." work_dir=${WORK_DIR} \
        monte_carlo.benchmark.dir=${BENCHMARK_DIR}
fi
echo
CKPT=10
WORK_DIR="lightning_logs/"
echo "================== ${WORK_DIR} =================="
python3 pl_trainer.py --config-dir conf --config-name pl_regressor \
    hydra/hydra_logging=disabled hydra.run.dir="." work_dir=${WORK_DIR} \
    data_module.distribution_target_size=$((${INS}-1)) \
    +pl_module.metric_list.acc.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    +pl_module.metric_list.auc.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    +pl_module.metric_list.err.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    +pl_module.metric_list.jsd.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    pl_module.head.num_classes=$((${INS}+2)) \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS} \
    ckpt=${CKPT}

python3 monte_carlo.py --config-dir conf --config-name pl_regressor \
    hydra/hydra_logging=disabled hydra.run.dir="." work_dir=${WORK_DIR} \
    pl_module.head.num_classes=$((${INS}+2)) \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS} \
    ckpt=${CKPT}

if [ -f ${WORK_DIR}/monte_carlo.csv ]; then
    rm -f ${WORK_DIR}/monte_carlo_*.csv
    python3 eval_metrics.py --config-dir conf --config-name pl_regressor \
        hydra/hydra_logging=disabled hydra.run.dir="." work_dir=${WORK_DIR} \
        monte_carlo.benchmark.dir=${BENCHMARK_DIR}
fi
