python -m ptls.pl_train_module --config-dir conf --config-name coles_xtransformer_best_params
python -m ptls.pl_inference --config-dir conf --config-name coles_xtransformer_best_params

rm -r conf/embeddings_validation.work
rm results/scenario_age_pred_coles_xtransformer.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
   --config-dir conf --config-name embeddings_validation_coles_xtransformer +workers=12 +total_cpu_count=16