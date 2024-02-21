#!/usr/bin/env bash

mkdir data
cd data

curl -OL 'https://huggingface.co/datasets/dllllb/rosbank-churn/resolve/main/train.csv.gz?download=true'
curl -OL 'https://huggingface.co/datasets/dllllb/rosbank-churn/resolve/main/test.csv.gz?download=true'

gunzip -f *.csv.gz
