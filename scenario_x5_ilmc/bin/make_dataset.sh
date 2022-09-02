#!/usr/bin/env bash

if [ ! -f data/purchases.csv ]; then
    mkdir -p data
    curl -OL https://storage.yandexcloud.net/datasouls-ods/materials/9c6913e5/retailhero-uplift.zip
    unzip -j retailhero-uplift.zip "data/p*.csv" -d data
    rm -f retailhero-uplift.zip
fi

for INS in 30 50 70 100 130; do
    python3 make_dataset.py --config-dir conf --config-name pl_regressor \
        hydra/hydra_logging=disabled hydra.run.dir="." ncats=${INS}
done
