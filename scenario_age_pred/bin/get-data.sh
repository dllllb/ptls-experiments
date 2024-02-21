#!/usr/bin/env bash

mkdir data
cd data

curl -OL 'https://huggingface.co/datasets/dllllb/age-group-prediction/resolve/main/transactions_train.csv.gz?download=true'
curl -OL 'https://huggingface.co/datasets/dllllb/age-group-prediction/resolve/main/transactions_test.csv.gz?download=true'
curl -OL 'https://huggingface.co/datasets/dllllb/age-group-prediction/resolve/main/train_target.csv?download=true'

gunzip -f *.csv.gz
