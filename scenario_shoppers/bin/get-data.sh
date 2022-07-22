#!/usr/bin/env bash

# https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data

mkdir -p data
mkdir -p results

curl -OL https://storage.yandexcloud.net/di-datasets/acquire-valued-shoppers.zip
unzip -j acquire-valued-shoppers.zip "[ost]*.csv.gz" -d data
rm -f acquire-valued-shoppers.zip
