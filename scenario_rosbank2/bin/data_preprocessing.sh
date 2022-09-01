#!/usr/bin/env bash

export PYTHONPATH="../../"
SPARK_LOCAL_IP="127.0.0.1" spark-submit \
    --master local[8] \
    --name "Rosbank Make Dataset" \
    --driver-memory 16G \
    --conf spark.sql.shuffle.partitions=60 \
    --conf spark.sql.parquet.compression.codec="snappy" \
    --conf spark.ui.port=4041 \
    --conf spark.local.dir="data/.spark_local_dir" \
    data_preprocessing.py
