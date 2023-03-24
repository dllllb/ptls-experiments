# agg feature encoder
python -m ptls.pl_inference --config-dir conf --config-name agg_features_params

# # Train the MeLES encoder and take embeddings; inference
python -m ptls.pl_train_module --config-dir conf --config-name coles_agg_params
python -m ptls.pl_inference --config-dir conf --config-name coles_agg_params

# # Train the MeLES encoder and take embeddings; inference
python -m ptls.pl_train_module --config-dir conf --config-name coles_params
python -m ptls.pl_inference --config-dir conf --config-name coles_params

## concat agg embeddings with coles embeddings 
python bin/concat.py 


# Compare
rm results/scenario_rosbank_baselines_unsupervised.txt
rm -r embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_baselines_unsupervised +workers=10 +total_cpu_count=20
