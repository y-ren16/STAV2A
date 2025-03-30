import os
from moviepy.editor import VideoFileClip
from moviepy.audio.AudioClip import AudioArrayClip
import json
import librosa
import numpy as np
from tqdm import tqdm
import logging
import warnings
import argparse
logging.getLogger('moviepy').setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, message="In file")

import sys
import contextlib
from io import StringIO

class MuteOutput:
    def __enter__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        return self

    def __exit__(self, *args):
        sys.stdout = self.stdout
        sys.stderr = self.stderr

def run(args):
    generation_result_path = args.generation_result_path
    # *.mp4
    # *.wav
    # *_origin.mp4
    all_file = os.listdir(generation_result_path)
    wav_list = []
    mp4_list = []
    # original_mp4_list = []
    for file in all_file:
        if file.endswith(".wav"):
            basename = file.split(".")[0]
            wav_list.append(basename)
        elif file.endswith(".mp4"):
            if file.endswith("_origin.mp4"):
                continue
                # basename = file.split(".")[0]
                # original_mp4_list.append(basename[:-7])
            else:
                basename = file.split(".")[0]
                mp4_list.append(basename)

    wav_list_set = set(wav_list)
    mp4_list_set = set(mp4_list)
    # original_mp4_list_set = set(original_mp4_list)

    # fild wav that not combined to mp4
    # wav_list_set - mp4_list_set
    wav_to_combine = wav_list_set - mp4_list_set
    print(f"wav to combine: {len(wav_to_combine)}")

    output_dir = generation_result_path 

    video_paths = [os.path.join(args.wav_path, f"{wav}.mp4") for wav in wav_to_combine]


    for video_path in tqdm(video_paths):
        basename = os.path.basename(video_path).split(".")[0]
        if not os.path.exists(f"{generation_result_path}/{basename}.wav"):
            import pdb; pdb.set_trace()
            continue
        audio_target = librosa.load(f"{generation_result_path}/{basename}.wav", sr=16000)[0] 

        video_target = VideoFileClip(video_path)
        duration = video_target.duration if video_target.duration < 10 else 10
        if duration > 10:
            video_target = video_target.subclip(0, 10)
        resampled_audio = librosa.resample(audio_target[:16000 * int(duration)], orig_sr=16000, target_sr=44100)
        audio_stereo = np.vstack((resampled_audio, resampled_audio)).T
        audio_clip = AudioArrayClip(audio_stereo, fps=44100)
        with MuteOutput():
            video_target.write_videofile("{}/{}_origin.mp4".format(output_dir, basename.split(".")[0]), codec="libx264", audio_codec="aac")
            video_target = video_target.set_audio(audio_clip)
            video_target.write_videofile("{}/{}.mp4".format(output_dir, basename.split(".")[0]), codec="libx264", audio_codec="aac")
        video_target.close()
        audio_clip.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command to do generate noisy speech for speech enhancement.",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--generation_result_path", type=str, default=None, help="Path to the generated audio")
    parser.add_argument("--wav_path", type=str, default="/xxxxxxxx/STA-V2A/data/VGGSound_CVAP_AValign_Filter_mp4", help="Path to the generated audio")
    args = parser.parse_args()
    run(args)