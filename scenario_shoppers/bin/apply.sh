#!/usr/bin/env bash
DATA_PATH="data"
INS=133
EMB=64
HID=128
REPEATS=1000
BENCHMARK_DIR="lightning_logs/benchmark"
BENCHMARK_TARGET="target_logvar"
export CUDA_VISIBLE_DEVICES=0

CKPT=10
echo "================== ${BENCHMARK_DIR} =================="
python3 pl_trainer.py --config-dir conf --config-name pl_regressor \
    hydra/hydra_logging=disabled hydra.run.dir="." data_path=${DATA_PATH} work_dir=${BENCHMARK_DIR} \
    data_module.setup.col_target=${BENCHMARK_TARGET} \
    ~pl_module.metric_list.jsd pl_module.seq_encoder.hidden_size=${HID} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.out=${EMB} \
    monte_carlo.ckpt=${CKPT}

echo
CKPT=10
WORK_DIR="lightning_logs/"
echo "================== ${WORK_DIR} =================="
python3 pl_trainer.py --config-dir conf --config-name pl_regressor \
    hydra/hydra_logging=disabled hydra.run.dir="." data_path=${DATA_PATH} work_dir=${WORK_DIR} \
    data_module.distribution_target_size=$((${INS}-1)) data_module.setup.col_target="target_dist" \
    +pl_module.metric_list.acc.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
    +pl_module.metric_list.auc.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
    +pl_module.metric_list.err.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
    +pl_module.metric_list.jsd.scaler._target_=ptls.nn.trx_encoder.scalers.PoissonScaler \
    pl_module.loss._target_=torch.nn.PoissonNLLLoss +pl_module.loss.log_input=false \
    pl_module.head.num_classes=$((${INS}-1)) pl_module.head.objective="softplus" \
    pl_module.seq_encoder.hidden_size=${HID} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.out=${EMB} \
    monte_carlo.ckpt=${CKPT}

python3 monte_carlo.py --config-dir conf --config-name pl_regressor \
    hydra/hydra_logging=disabled hydra.run.dir="." data_path=${DATA_PATH} work_dir=${WORK_DIR} \
    pl_module.head.num_classes=$((${INS}-1)) pl_module.head.objective="softplus" \
    pl_module.seq_encoder.hidden_size=${HID} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.out=${EMB} \
    monte_carlo.ckpt=${CKPT} monte_carlo.repeats=${REPEATS}

python3 eval_metrics.py --config-dir conf --config-name pl_regressor \
    hydra/hydra_logging=disabled hydra.run.dir="." work_dir=${WORK_DIR} \
    monte_carlo.benchmark.dir=${BENCHMARK_DIR}

echo
CKPT=10
WORK_DIR="lightning_logs/"
echo "================== ${WORK_DIR} =================="
python3 pl_trainer.py --config-dir conf --config-name pl_regressor \
    hydra/hydra_logging=disabled hydra.run.dir="." data_path=${DATA_PATH} work_dir=${WORK_DIR} \
    data_module.distribution_target_size=$((${INS}-1)) data_module.setup.col_target="target_dist" \
    +pl_module.metric_list.acc.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    +pl_module.metric_list.auc.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    +pl_module.metric_list.err.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    +pl_module.metric_list.jsd.scaler._target_=ptls.nn.trx_encoder.scalers.ExpScaler \
    pl_module.head.num_classes=$((${INS}+2)) pl_module.seq_encoder.hidden_size=${HID} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.out=${EMB} \
    monte_carlo.ckpt=${CKPT}

python3 monte_carlo.py --config-dir conf --config-name pl_regressor \
    hydra/hydra_logging=disabled hydra.run.dir="." data_path=${DATA_PATH} work_dir=${WORK_DIR} \
    pl_module.head.num_classes=$((${INS}+2)) pl_module.seq_encoder.hidden_size=${HID} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.in=${INS} \
    pl_module.seq_encoder.trx_encoder.embeddings.category.out=${EMB} \
    monte_carlo.ckpt=${CKPT} monte_carlo.repeats=${REPEATS}

python3 eval_metrics.py --config-dir conf --config-name pl_regressor \
    hydra/hydra_logging=disabled hydra.run.dir="." work_dir=${WORK_DIR} \
    monte_carlo.benchmark.dir=${BENCHMARK_DIR}
