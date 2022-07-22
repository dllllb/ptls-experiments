#!/usr/bin/env bash

python3 make_dataset_pd.py --data data/transactions.csv.gz --sample_id id --sample_size 30000

mkdir -p data/sample
mv data/transactions_sample.csv.gz data/sample/transactions.csv.gz

python3 make_dataset_pd.py --data data/sample/transactions.csv.gz \
    --bins 33 --qlim 0.995 --log_file results/make_sample.log

python3 make_dataset_pd.py --data data/transactions.csv.gz \
    --bins 33 --qlim 0.995 --log_file results/make_dataset.log
