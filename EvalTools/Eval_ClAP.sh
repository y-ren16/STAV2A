json_paths_base='/xxxxxxxx/VideoLLaMA2/data'
json_files=(
    "vgg_dataset_v2a_test_av_align_filter_filter.json"
    "vgg_dataset_v2a_test_av_align_filter_filter_audio_caption_for_video_videollama2.1_7b_16f_base.json"
    "vgg_dataset_v2a_test_av_align_filter_filter_audio_caption_for_video_lora_finetune_siglip_tcv35_7b_16f.json"
    "vgg_dataset_v2a_test_av_align_filter_filter_audio_caption_for_video_lora_finetune_siglip_tcv35_7b_16f_ep2.json"
    "vgg_dataset_v2a_test_av_align_filter_filter_videollava_llama3ft.json"
    "vgg_dataset_v2a_test_av_align_filter_filter_videollava_llama3ft.json"
)

caption_keys=(
    "captions"
    "audio_caption_for_video_7b_16f"
    "audio_caption_for_video_sft_model1"
    "audio_caption_for_video_sft_model2"
    "llama3_ft_caption"
    "no_caption"
)

# 使用索引遍历数组
for i in "${!json_files[@]}"
do
    python CLAP_score.py \
    --generation_result_path "/xxxxxxxx/STA-V2A/data/VGGSound_CVAP_AValign_Filter_wav" \
    --ref_json "$json_paths_base/${json_files[$i]}" \
    --caption_keys "${caption_keys[$i]}"
done
