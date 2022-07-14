
# train special model for fine-tunnig in semi-supervised setup 
# it is quite smaller, than one which is used in supervised setup, due to insufficiency labeled data to train a big model. 
python -m ptls.pl_train_module \
    params.rnn.hidden_size=160 \
	model_path="${hydra:runtime.cwd}/../../artifacts/scenario_age_pred/age_pred_ml_model_ss_ft.p" \
    --config-dir conf --config-name mles_params

for SC_AMOUNT in 00337 00675 01350 02700 05400 10800 21600
do
    python -m ptls.pl_fit_target \
        logger_name="fit_target_${SC_AMOUNT}" \
        trainer.max_epochs=20 \
        +data_module.train_drop_last=true \
        +data_module.train.labeled_amount=$SC_AMOUNT \
        embedding_validation_results.feature_name="target_scores_${SC_AMOUNT}" \
        embedding_validation_results.output_path="${hydra:runtime.cwd}/results/fit_target_${SC_AMOUNT}_results.json" \
        --config-dir conf --config-name pl_fit_target

    python -m ptls.pl_fit_target \
        logger_name="mles_finetuning_${SC_AMOUNT}" \
        +data_module.train.labeled_amount=$SC_AMOUNT \
        +params.rnn.hidden_size=160 \
        +params.pretrained_model_path="${hydra:runtime.cwd}/../../artifacts/scenario_age_pred/age_pred_ml_model_ss_ft.p" \
        embedding_validation_results.feature_name="mles_finetuning_${SC_AMOUNT}" \
        embedding_validation_results.output_path="${hydra:runtime.cwd}/results/mles_finetuning_${SC_AMOUNT}_results.json" \
        --config-dir conf --config-name pl_fit_finetuning_mles

    python -m ptls.pl_fit_target \
        logger_name="cpc_finetuning_${SC_AMOUNT}" \
        +data_module.train.labeled_amount=$SC_AMOUNT \
        +params.pretrained_model_path="${hydra:runtime.cwd}/../../artifacts/scenario_age_pred/cpc_model.p" \
        embedding_validation_results.feature_name="cpc_finetuning_${SC_AMOUNT}" \
        embedding_validation_results.output_path="${hydra:runtime.cwd}/results/cpc_finetuning_${SC_AMOUNT}_results.json" \
        --config-dir conf --config-name pl_fit_finetuning_cpc
done

rm results/scenario_age_pred__semi_supervised.txt
python -m embeddings_validation \
  --config-dir conf --config-name embeddings_validation_semi_supervised \
  +workers=10 --total_cpu_count 10
