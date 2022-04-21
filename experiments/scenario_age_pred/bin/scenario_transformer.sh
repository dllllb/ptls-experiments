# Train the MeLES encoder and take embedidngs; inference
# python -m dltranz.pl_train_module --conf conf/mles_params.hocon
# python -m dltranz.pl_inference --conf conf/mles_params.hocon

# Train the Transformers embedder; inference
python -m dltranz.pl_train_module --conf conf/transformer_params.hocon
python -m dltranz.pl_inference --conf conf/transformer_params.hocon


# Compare
rm -f results/scenario_age_pred_transformer.txt

rm -rf conf/embeddings_validation.work/

python -m embeddings_validation \
   --conf conf/embeddings_validation_transformer.hocon --workers 10 --total_cpu_count 20
