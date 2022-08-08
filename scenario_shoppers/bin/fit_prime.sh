#!/usr/bin/env bash
DATA_PATH="data/sample"
INS=133
EMB=64
HID=128
REPEATS=1000
export CUDA_VISIBLE_DEVICES=0

CKPT=6
WORK_DIR="lightning_logs/loss_pois132_target_dist_in133_1659983233"
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

echo
CKPT=11
WORK_DIR="lightning_logs/loss_ziln135_target_dist_in133_1659985822"
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
