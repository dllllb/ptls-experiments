# Train the MeLES encoder and take embeddings; inference
# python -m ptls.pl_train_module --config-dir conf --config-name mles_params
# python -m ptls.pl_inference --config-dir conf --config-name mles_params

# Train the Transformers embedder; inference
python -m ptls.pl_train_module --config-dir conf --config-name transformer_params
python -m ptls.pl_inference --config-dir conf --config-name transformer_params


# Compare
rm -f results/scenario_age_pred_transformer.txt

rm -rf conf/embeddings_validation.work/

python -m embeddings_validation \
   --config-dir conf --config-name embeddings_validation_transformer +workers=10 +total_cpu_count=20
