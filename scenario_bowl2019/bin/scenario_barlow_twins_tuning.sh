python -m ptls.pl_fit_target \
  data_module.train_drop_last=true \
  logger_name="bf_ftning_v01" \
  params.pretrained.lr=0.001 \
  --config-dir conf --config-name pl_fit_finetuning_barlow_twins


