"""
AA-Align Metric: Audio-Audeo Onset Alignment Evaluation

AA-Align is a metric for evaluating the alignment between audios.
It assesses synchronization by detecting peaks of two audios and calculating their Intersection over Union (IoU).
A higher IoU score indicates better alignment.

Usage:
- Provide a folder of audio files as input.
- The script calculates the AA-Align score for the set of audios.
"""


import argparse
import glob
import librosa
import librosa.display
import os
from tqdm import tqdm
import json


# Function to detect audio peaks using the Onset Detection algorithm
def detect_audio_peaks(audio_file, length=None):
    """
    Detect audio peaks using the Onset Detection algorithm.

    Args:
        audio_file (str): Path to the audio file.

    Returns:
        onset_times (list): List of times (in seconds) where audio peaks occur.
    """
    y, sr = librosa.load(audio_file)
    if length:
        y = y[:int(length*sr)]
    # Calculate the onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # Get the onset events
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return onset_times, sr


# Function to calculate Intersection over Union (IoU) for audio and audio peaks
def calc_intersection_over_union(audio_peaks_gaudio_peaks_gtt, audio_peaks_gen, sr, tim_win):
    """
    Calculate Intersection over Union (IoU) between peaks of two audios.

    Args:
        audio_peaks (list): List of audio peak times (in seconds).
        fps (float): Frame rate of the video.

    Returns:
        iou (float): Intersection over Union score.
    """
    # import pdb; pdb.set_trace()
    intersection_length = 0
    for audio_peak_gen in audio_peaks_gen:
        for audio_peak_gt in audio_peaks_gt:
            if audio_peak_gt - tim_win < audio_peak_gen < audio_peak_gt + tim_win:
                intersection_length += 1
                break
    return intersection_length / (len(audio_peaks_gen) + len(audio_peaks_gt) - intersection_length)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir_GT", type=str, default='/xxxxxxx/intern/yongren/v2a/v2a_pretrain/data/New_Audio_GT_Test_VGGSound_Filter', help='Insert the videos folder path') 
    parser.add_argument("--wavscp", type=str, required=True, help='Insert the videos folder path')
    parser.add_argument("--tim_win", type=float, default=0.1, help='Insert the time window for calculating IoU')
    parser.add_argument("--length", type=float, default=None, help='Insert the length of audio for calculating IoU')
    args = parser.parse_args()
    print(args.input_dir_GT)
    print(args.wavscp)
    f_out = open(args.wavscp + ".aa_align_score.json",'w')
    all_scores = []
    # import pdb; pdb.set_trace()
    with open(args.wavscp) as f:
        data = f.readlines()
        for line in tqdm(data):
            utt = line.strip()
            input_file = json.loads(utt)
            audio_path_gen = input_file["location"]
            file_basename_gen = os.path.basename(audio_path_gen)
            audio_path_gt = os.path.join(args.input_dir_GT, file_basename_gen)
            if not os.path.exists(audio_path_gt):
                print(f"{audio_path_gt} not exist")
                continue
            audio_peaks_gt, sr_gt = detect_audio_peaks(audio_path_gt, length=args.length)
            audio_peaks_gen, sr_gen = detect_audio_peaks(audio_path_gen, length=args.length)
            assert sr_gt == sr_gen

            score = calc_intersection_over_union(audio_peaks_gt, audio_peaks_gen, sr_gt, args.tim_win)
            input_file["aa_align_score"] = score
            f_out.write(json.dumps(input_file) + "\n") 
            f_out.flush()
            # print(f"AA-Align: {score}")
            all_scores.append(score)
    f_out.close()
    mean_scores = sum(all_scores)/len(all_scores)
    print('AA-Align: ', mean_scores)
    mean_score_out = open(args.wavscp + ".aa_align_score_mean.txt",'w')
    mean_score_out.write(str(mean_scores))
    mean_score_out.close()
