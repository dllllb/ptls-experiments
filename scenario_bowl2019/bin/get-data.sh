#!/usr/bin/env bash

mkdir data
cd data

curl -o train.csv -L 'https://huggingface.co/datasets/dllllb/datascience-bowl2019/resolve/main/train.csv.gz?download=true'
curl -o test.csv -L 'https://huggingface.co/datasets/dllllb/datascience-bowl2019/resolve/main/test.csv.gz?download=true'
curl -o train_labels.csv -L 'https://huggingface.co/datasets/dllllb/datascience-bowl2019/resolve/main/train_labels.csv.gz?download=true'

gunzip -f *.csv.gz
