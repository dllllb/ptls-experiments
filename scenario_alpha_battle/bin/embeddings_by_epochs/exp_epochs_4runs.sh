
checkpoints_folder="lightning_logs/$1/version_${2:-0}/checkpoints/*.ckpt"
output_file_prefix="${1}__"
conf_file="$1"
batch_size=64

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

 if [ $epoch_num = "024" ] || [ $epoch_num = "020" ] || [ $epoch_num = "015" ] || [ $epoch_num = "029" ]  || [ $epoch_num = "034" ] || [ $epoch_num = "039" ]; then
  if [ -f "data/$output_file.pickle" ]; then
    echo "--------: $output_file exists"
  else
      echo "--------: Run inference for $output_file"
      python -m ptls.pl_inference model_path=\""${model_file}"\" embed_file_name="${output_file}" inference.batch_size=${batch_size} --config-dir conf --config-name "${conf_file}"
  fi
 fi
done

rm results/$1.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_short +workers=10 +total_cpu_count=20 \
    +report_file="results/$1.txt" \
    +auto_features=["data/${1}__???.pickle"]
