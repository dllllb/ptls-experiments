#!/usr/bin/env bash

# https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data

mkdir data

curl -OL https://storage.yandexcloud.net/di-datasets/acquire-valued-shoppers.zip

unzip acquire-valued-shoppers.zip -d data/
mv acquire-valued-shoppers.zip data/
