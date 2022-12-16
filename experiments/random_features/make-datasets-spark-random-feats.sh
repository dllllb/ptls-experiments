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
    make_dataset.py \
    --data_path data/ \
    --trx_files "#predefined" \
    --col_client_id "cl_id" \
    --cols_event_time "TRDATETIME" \
    --cols_category "feature_1" "feature_2" "feature_3" "feature_4" "feature_5" "arange" "mcc" "channel_type" "currency" "trx_category" \
    --cols_log_norm "amount" \
    --target_files "#predefined" \
    --col_target "target_flag" "target_sum" \
    --test_size 0.1 \
    --output_train_path "data/train_trx.parquet" \
    --output_test_path "data/test_trx.parquet" \
    --output_test_ids_path "data/test_ids.csv" \
    --log_file "results/dataset_rosbank.txt" \
    --print_dataset_info

# 41 sec with    --print_dataset_info
# 21 sec without --print_dataset_info
