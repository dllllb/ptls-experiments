# nsp
checkpoints_folder="lightning_logs/nsp_model/version_0/checkpoints/*.ckpt"
output_file_prefix="nsp__"
conf_file="nsp_params"
batch_size=800

for model_file in $(ls -vr $checkpoints_folder)
do
  echo "--------: $model_file"
  model_name=$(basename "$model_file")
#  model_file=${model_file//"="/"\="}

  # echo $model_name  # epoch=9-step=44889.ckpt
  epoch_num=$(echo $model_name | cut -f 1 -d "-")
  epoch_num=$(echo $epoch_num | cut -f 2 -d "=")
  epoch_num=$(printf %03d $epoch_num)
  # echo $epoch_num

  output_file=$(echo $output_file_prefix$epoch_num)

  if [ -f "$output_file.pickle" ]; then
    echo "--------: $output_file exists"
  else
    echo "--------: Run inference for $output_file"
    python -m ptls.pl_inference model_path=\""${model_file}"\" embed_file_name="${output_file}" inference.batch_size=${batch_size} --config-dir conf --config-name "${conf_file}"
  fi
done

rm results/epochs_nsp.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_short +workers=10 +total_cpu_count=20 \
    +report_file="../results/epochs_nsp.txt" \
    +auto_features=["../data/nsp__???.pickle"]
