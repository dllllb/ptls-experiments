#!/usr/bin/env bash

mkdir data
cd data

curl -o gender_train.csv -L 'https://huggingface.co/datasets/dllllb/transactions-gender/resolve/main/gender_train.csv?download=true'
curl -o transactions.csv -L 'https://huggingface.co/datasets/dllllb/transactions-gender/resolve/main/transactions.csv.gz?download=true'

gunzip -f *.csv.gz
