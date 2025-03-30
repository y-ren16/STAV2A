
steps=200
guidance=3
num_samples=1
models=(
  "STA-V2A"
)
caption_keys=(
  "captions"
)
save_dir=(
  "data_vgg"
)
json_file_name=(
  "data/test.json"
)
gpu_id=(
  0
)
epoch_num=(
  best
)
model_save_type="pytorch_model_2.bin"
mkdir -p ./outputs

for i in "${!models[@]}"; do
  model="${models[i]}"
  unique_identifier=$(date +"%Y%m%d_%H%M%S")

  mkdir -p ./logs_inf
  mkdir -p ./logs_inf/${save_dir[i]}

  CUDA_VISIBLE_DEVICES="${gpu_id[i]}" python inf_Paper_CN_onset.py \
    --original_args="saved/$model/summary.jsonl" \
    --model="saved/$model/${epoch_num[i]}/$model_save_type" \
    --save_dir ./outputs/${save_dir[i]} \
    --num_steps $steps \
    --guidance $guidance \
    --num_samples $num_samples \
    --text_key ${caption_keys[i]} \
    --video_key video_location \
    --video_fps 40 \
    --has_global_video_feature \
    --video_feature_key CAVP_feature \
    --use_feature_window \
    --test_file ./${json_file_name[i]} # > ./logs_inf/${save_dir[i]}/${unique_identifier}_${model}_${epoch_num[i]}.log 2>&1
done
