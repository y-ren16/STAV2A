#!/bin/bash
# pip3 install laion_clap
unique_identifier=$(date +"%Y%m%d_%H%M%S")

base_path="./outputs"
folders=(
    # "data_vgg/1743262910_STA-V2A_best_steps_200_guidance_3.0_test"
    "data_vgg/1743265671_STA-V2A_best_steps_200_guidance_3.0_vgg_dataset_v2a_test_av_align_filter_filter"
)
logs_names=(
    "data_vgg"
)
cuda_devices=(
    "0"
)

mkdir -p ./Eval_logs
mkdir -p ./Eval_logs/AV_Align
mkdir -p ./Eval_logs/AA_Align
mkdir -p ./Eval_logs/PAM
mkdir -p ./Eval_logs/CLAP


for i in "${!folders[@]}"; do
    export CUDA_VISIBLE_DEVICES=${cuda_devices[$i]}
    folder="${base_path}/${folders[i]}"
    exp_dir=${folder##*/}
    json_folder=${folder%"$exp_dir"}"JSON_$exp_dir"
    if [ ! -d "$json_folder" ]; then
        mkdir -p "$json_folder"
    fi
    # generate json file
    python3 EvalTools/folder2json.py --folder $folder --json_folder $json_folder

    # av_align
    nohup python3 EvalTools/av_align.py --wavscp $json_folder/base_json.json --max_workers 10 > ./Eval_logs/AV_Align/${unique_identifier}_${logs_names[$i]}.log 2>&1 &
    # python3 EvalTools/av_align.py --wavscp $json_folder/base_json.json --max_workers 10

    # # aa_align
    nohup python3 EvalTools/aa_align.py --input_dir_GT ./data/VGGSound_CVAP_AValign_Filter_wav --wavscp $json_folder/base_json.json --tim_win 0.1 > ./Eval_logs/AA_Align/${unique_identifier}_${logs_names[$i]}.log 2>&1 &
    # python3 EvalTools/aa_align.py --input_dir_GT ./data/VGGSound_CVAP_AValign_Filter_wav --wavscp $json_folder/base_json.json --tim_win 0.1 

    # PAM
    nohup python3 EvalTools/PAM/eval_pam.py --wavscp $json_folder/base_json.json --batch_size 20 --num_workers 0 > ./Eval_logs/PAM/${unique_identifier}_${logs_names[$i]}.log 2>&1 &
    # python3 EvalTools/PAM/eval_pam.py --wavscp $json_folder/base_json.json --batch_size 20 --num_workers 0 

    # CLAP
    nohup python3 EvalTools/CLAP_score.py --generation_result_path $folder --ref_json ./data/vgg_dataset_v2a_test_av_align_filter_filter.json > ./Eval_logs/CLAP/${unique_identifier}_${logs_names[$i]}.log 2>&1 &
    # python3 EvalTools/CLAP_score.py --generation_result_path $folder --ref_json ./data/vgg_dataset_v2a_test_av_align_filter_filter.json
done