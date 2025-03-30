#!/bin/bash
unique_identifier=$(date +"%Y%m%d_%H%M%S")

base_path="./outputs"
folders=(
    data_vgg/1743265671_STA-V2A_best_steps_200_guidance_3.0_vgg_dataset_v2a_test_av_align_filter_filter
)
logs_names=(
    captions
)
cuda_devices=(
    "0"
)
gt_audio_path=(
    ./data/VGGSound_CVAP_AValign_Filter_wav
    # /xxxxxxxx/STA-V2A/data/VGGSound_CVAP_AValign_Filter_wav
)
for i in "${!folders[@]}"; do
    export CUDA_VISIBLE_DEVICES=${cuda_devices[$i]}
    folder="${base_path}/${folders[i]}"
    exp_dir=${folder##*/}
    python3 ./EvalTools/audioldm_eval/test.py \
    --generation_result_path $folder \
    --gt_audio_path ${gt_audio_path[$i]}
done