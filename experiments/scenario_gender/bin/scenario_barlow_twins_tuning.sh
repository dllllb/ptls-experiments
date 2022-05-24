

# Lambda in loss
for SC_PARAMETER in 0.06  # 0.001 0.01 0.1 0.02 0.04 0.005 0.002
do
  export SC_SUFFIX="bt_lambd_${SC_PARAMETER}"
  python -m ptls.pl_train_module \
      logger_name=${SC_SUFFIX} \
      params.train.lambd=${SC_PARAMETER} \
      model_path="${hydra:runtime.cwd}/../../artifacts/scenario_gender/gender_mlm__$SC_SUFFIX.p" \
      --config-dir conf --config-name barlow_twins_params
  python -m ptls.pl_inference \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_gender/gender_mlm__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}" \
      --config-dir conf --config-name barlow_twins_params
done
# Compare
rm results/res_bt_lambd.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_short +workers=10 +total_cpu_count=20 \
    report_file="${hydra:runtime.cwd}/results/res_bt_lambd.txt" \    
    auto_features=["${hydra:runtime.cwd}/data/emb__bt_lambd_*.pickle", "${hydra:runtime.cwd}/data/barlow_twins_embeddings.pickle"]
less -S results/res_bt_lambd.txt


# hidden_size
for SC_PARAMETER in 1536 # 128 256 512 768
do
  export SC_SUFFIX="bt_hs_${SC_PARAMETER}"
  python -m ptls.pl_train_module \
      logger_name=${SC_SUFFIX} \
      params.rnn.hidden_size=${SC_PARAMETER} \
      model_path="${hydra:runtime.cwd}/../../artifacts/scenario_gender/gender_mlm__$SC_SUFFIX.p" \
      --config-dir conf --config-name barlow_twins_params
  python -m ptls.pl_inference \
    inference_dataloader.loader.batch_size=500 \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_gender/gender_mlm__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}" \
      --config-dir conf --config-name barlow_twins_params
done
# Compare
rm results/res_bt_hs.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_short +workers=10 +total_cpu_count=20 \
    report_file="${hydra:runtime.cwd}/results/res_bt_hs.txt" \    
    auto_features=["${hydra:runtime.cwd}/data/emb__bt_hs_*.pickle", "${hydra:runtime.cwd}/data/barlow_twins_embeddings.pickle"]
less -S results/res_bt_hs.txt

# prj
for SC_PARAMETER in 256  # 64 128 256 512 768
do
  export RNN_SIZE=2048
  export SC_SUFFIX="bt_prj_${RNN_SIZE}_${SC_PARAMETER}"
  export PRJ_SIZE=${SC_PARAMETER}
  python -m ptls.pl_train_module \
      logger_name=${SC_SUFFIX} \
      params.rnn.hidden_size="${RNN_SIZE}" \
      "params.head_layers=[[Linear, {in_features: ${RNN_SIZE}, out_features: ${PRJ_SIZE}, bias: false}], [BatchNorm1d, {num_features: ${PRJ_SIZE}}], [ReLU, {}], [Linear, {in_features: ${PRJ_SIZE}, out_features: ${PRJ_SIZE}, bias: false}], [BatchNorm1d, {num_features: ${PRJ_SIZE}, affine: False}]]" \
      model_path="${hydra:runtime.cwd}/../../artifacts/scenario_gender/gender_mlm__$SC_SUFFIX.p" \
      --config-dir conf --config-name barlow_twins_params
  python -m ptls.pl_inference \
    inference_dataloader.loader.batch_size=500 \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_gender/gender_mlm__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}" \
      --config-dir conf --config-name barlow_twins_params
done
# Compare
rm results/res_bt_prj.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_short +workers=10 +total_cpu_count=20 \
    report_file="${hydra:runtime.cwd}/results/res_bt_prj.txt" \    
    auto_features=["${hydra:runtime.cwd}/data/emb__bt_prj_*.pickle", "${hydra:runtime.cwd}/data/barlow_twins_embeddings.pickle"]
less -S results/res_bt_prj.txt


# batch_size
for SC_PARAMETER in 64 # 256
do
  export SC_SUFFIX="bt_bs_${SC_PARAMETER}"
  python -m ptls.pl_train_module \
      logger_name=${SC_SUFFIX} \
      data_module.train.batch_size=${SC_PARAMETER} \
      model_path="${hydra:runtime.cwd}/../../artifacts/scenario_gender/gender_mlm__$SC_SUFFIX.p" \
      --config-dir conf --config-name barlow_twins_params
  python -m ptls.pl_inference \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_gender/gender_mlm__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}" \
      --config-dir conf --config-name barlow_twins_params
done
# Compare
rm results/res_bt_bs.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_short +workers=10 +total_cpu_count=20 \
    report_file="${hydra:runtime.cwd}/results/res_bt_bs.txt" \    
    auto_features=["${hydra:runtime.cwd}/data/emb__bt_bs_*.pickle", "${hydra:runtime.cwd}/data/barlow_twins_embeddings.pickle"]
less -S results/res_bt_bs.txt



export SC_SUFFIX="bt_tuning_new"
python -m ptls.pl_train_module \
    logger_name=${SC_SUFFIX} \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_gender/gender_mlm__$SC_SUFFIX.p" \
    --config-dir conf --config-name barlow_twins_params
python -m ptls.pl_inference         inference_dataloader.loader.batch_size=500 \
    model_path="${hydra:runtime.cwd}/../../artifacts/scenario_gender/gender_mlm__$SC_SUFFIX.p" \
    output.path="${hydra:runtime.cwd}/data/emb__${SC_SUFFIX}" \
    --config-dir conf --config-name barlow_twins_params
