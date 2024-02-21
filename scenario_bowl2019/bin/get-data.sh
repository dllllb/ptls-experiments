#!/usr/bin/env bash

mkdir data
cd data

curl -OL 'https://huggingface.co/datasets/dllllb/datascience-bowl2019/resolve/main/train_labels.csv.gz?download=true'
curl -OL 'https://huggingface.co/datasets/dllllb/datascience-bowl2019/resolve/main/train.csv.gz?download=true'
curl -OL 'https://huggingface.co/datasets/dllllb/datascience-bowl2019/resolve/main/test.csv.gz?download=true'

gunzip -f *.csv.gz
