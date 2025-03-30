import os
import json
from tqdm import tqdm

json_file = "/xxxxxxxx/STA-V2A/data/VGGSound_CVAP_AValign_Filter/vgg_dataset_v2a_test_av_align_filter_filter.json"
# target_dir = "/xxxxxxxx/STA-V2A/data/VGGSound_CVAP_AValign_Filter_wav" 
target_dir = "/xxxxxxxx/STA-V2A/data/VGGSound_CVAP_AValign_Filter_mp4" 
os.makedirs(target_dir, exist_ok=True)

all_data = []
with open(json_file, 'r') as f:
    all_lines = f.readlines()
    for line in tqdm(all_lines):
        data = json.loads(line)
        all_data.append(data)
        # wavlocation = data['location']
        # basename = os.path.basename(wavlocation).split('.')[0]
        # # cp command
        # cmd = f"cp {wavlocation} {target_dir}/{basename}.wav"
        mp4location = data['video_location']
        basename = os.path.basename(mp4location).split('.')[0]
        # cp command
        cmd = f"cp {mp4location} {target_dir}/{basename}.mp4"
        os.system(cmd)