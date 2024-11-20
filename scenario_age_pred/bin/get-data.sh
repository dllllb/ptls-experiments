#!/usr/bin/env bash

mkdir data
cd data

curl -o transactions_train.csv -L 'https://huggingface.co/datasets/dllllb/age-group-prediction/resolve/main/transactions_train.csv.gz?download=true'
curl -o transactions_test.csv -L 'https://huggingface.co/datasets/dllllb/age-group-prediction/resolve/main/transactions_test.csv.gz?download=true'
curl -o train_target.csv -L 'https://huggingface.co/datasets/dllllb/age-group-prediction/resolve/main/train_target.csv?download=true'

gunzip -f *.csv.gz
