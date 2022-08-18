#!/usr/bin/env bash

if [ ! -f data/transactions.csv.gz ]; then
    mkdir -p data
    curl -OL https://storage.yandexcloud.net/di-datasets/acquire-valued-shoppers.zip
    unzip -j acquire-valued-shoppers.zip "[ost]*.csv.gz" -d data
    rm -f acquire-valued-shoppers.zip
fi

python3 make_dataset.py --config-dir conf --config-name pl_regressor \
    hydra/job_logging=disabled hydra/hydra_logging=disabled hydra.run.dir="." \
    monte_carlo.chunk=50000 data_path="data"

mkdir -p data/sample
mv data/transactions_sample.csv.gz data/sample/transactions.csv.gz

python3 make_dataset.py --config-dir conf --config-name pl_regressor \
    hydra/hydra_logging=disabled hydra.run.dir="." \
    monte_carlo.agg_time=10 data_path="data/sample"

python3 make_dataset.py --config-dir conf --config-name pl_regressor \
    hydra/hydra_logging=disabled hydra.run.dir="." \
    monte_carlo.agg_time=10 data_path="data"
