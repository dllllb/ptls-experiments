#!/usr/bin/env bash

mkdir data
cd data

curl -o train.csv 'https://huggingface.co/datasets/dllllb/rosbank-churn/resolve/main/train.csv.gz?download=true'
curl -o test.csv 'https://huggingface.co/datasets/dllllb/rosbank-churn/resolve/main/test.csv.gz?download=true'

gunzip -f *.csv.gz
