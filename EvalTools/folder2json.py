import os
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, required=True, help='Insert the audio folder path')
parser.add_argument("--json_folder", type=str, required=True, help='Insert the json folder path')
args = parser.parse_args()

audio_path = args.folder
json_folder = args.json_folder
video_path = audio_path

with open(os.path.join(json_folder, "base_json.json"), "w") as f_out:
    for file in tqdm(os.listdir(audio_path)):
        if file.endswith(".wav"):
            # import pdb; pdb.set_trace()
            basename = file.split(".")[0]
            audio_file_path = os.path.join(audio_path, file)
            video_file_path = os.path.join(video_path, basename + ".mp4")
            assert os.path.exists(audio_file_path)
            assert os.path.exists(video_file_path)
            assert os.path.exists(os.path.join(audio_path, basename + "_origin.mp4"))
            f_out.write(json.dumps({"location": audio_file_path, "video_location": video_file_path}) + "\n")
    # for video_file in tqdm(os.listdir(video_path)):
    #     if video_file.endswith("_origin.mp4"):
    #         audio_name = video_file.split(".")[0][:-7]
    #         audio_file_path = os.path.join(audio_path, audio_name + ".mp4.wav")
    #         # audio_file_path = os.path.join(audio_path, audio_name + ".wav")
    #         video_file_path = os.path.join(video_path, audio_name + ".mp4")
    #         f_out.write(json.dumps({"location": audio_file_path, "video_location": video_file_path}) + "\n")

        