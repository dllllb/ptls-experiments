echo "============================="
echo "RANDOM PART"
echo "============================="
echo "==== Device cuda:${CUDA_VISIBLE_DEVICES} will be used"

rm data/train.csv
rm data/test.csv

sh bin/get-data.sh

echo "============================="
echo "adding random features"
python add_random_features.py
echo "============================="
echo "done"


echo "============================="
echo "making datasets"
sh bin/make-datasets-spark-random-feats.sh #--category_features "mcc" "channel_type" "currency" "trx_category" "feature_1" "feature_2" # +chtototam
echo "============================="
echo "done"

# echo "==== Folds split"
# rm -r lightning_logs/
# rm -r conf/embeddings_validation.work/
# python -m embeddings_validation \
#     --config-dir conf --config-name embeddings_validation_baselines_supervised +workers=10 +total_cpu_count=20 \
#     +split_only=True



# Train the MeLES encoder and take embeddings; inference
echo "============================="
echo "training random"
# python -m ptls.pl_train_module --config-dir conf --config-name mles_params_random
# python -m ptls.pl_inference --config-dir conf --config-name mles_params_random
echo "============================="
echo "done"



echo "============================="
echo "NO RANDOM PART"
echo "============================="

# Train the MeLES encoder and take embeddings; inference
echo "============================="
echo "training without random"
# python -m ptls.pl_train_module --config-dir conf --config-name mles_params
# python -m ptls.pl_inference --config-dir conf --config-name mles_params
echo "============================="
echo "done"


echo "============================="
echo "training cpc with arange"
# Train the Contrastive Predictive Coding (CPC) model; inference
python -m ptls.pl_train_module --config-dir conf --config-name cpc_params_arange
python -m ptls.pl_inference --config-dir conf --config-name cpc_params_arange
echo "============================="
echo "done"



# echo "============================="
# echo "training cpc without arange"
# # Train the Contrastive Predictive Coding (CPC) model; inference
# python -m ptls.pl_train_module --config-dir conf --config-name cpc_params
# python -m ptls.pl_inference --config-dir conf --config-name cpc_params
# echo "============================="
# echo "done"

# agg feature encoder
# python -m ptls.pl_inference --config-dir conf --config-name agg_features_params

# Random encoder
# python -m ptls.pl_inference --config-dir conf --config-name random_params

Compare
rm -r embeddings_validation.work/
rm results/scenario_random_features.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation --config-dir conf --config-name random_features_validation +workers=10 +total_cpu_count=20