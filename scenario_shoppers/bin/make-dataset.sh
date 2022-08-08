#!/usr/bin/env bash

python3 make_dataset.py --data data/transactions.csv.gz --id id --size 30000

mkdir -p data/sample
mv data/transactions_sample.csv.gz data/sample/transactions.csv.gz

python3 make_dataset.py --data data/sample/transactions.csv.gz --bins 33 --fsum 0.683 --qlim 0.995
python3 make_dataset.py --data data/transactions.csv.gz --bins 33 --fsum 0.683 --qlim 0.995
