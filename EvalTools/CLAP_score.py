from datasets import load_dataset
from transformers import AutoProcessor, ClapModel
import numpy as np
import librosa
import argparse
import os
import soundfile as sf
import torch
import laion_clap
import json
from tqdm.auto import tqdm

device = torch.device("cuda:0") 
model = ClapModel.from_pretrained("saved/pretrained_model/clap/clap-htsat-unfused").to(device)
model.eval()
processor = AutoProcessor.from_pretrained("saved/pretrained_model/clap/clap-htsat-unfused")

model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
model.load_ckpt("saved/pretrained_model/clap/laion_clap/630k-best.pt")


def get_clap_score(audio, texts):
    # audio: (batch_size, num_samples)
    # texts: (batch_size, num_texts)
    audio = audio.reshape(1, -1)
    audio = audio

    text_embed = model.get_text_embedding(texts)
    audio_embed = model.get_audio_embedding_from_data(x=audio)
    probs = torch.tensor(audio_embed) @ torch.tensor(text_embed).t()

    #inputs = processor(text=texts, audios=audio, return_tensors="pt", padding=True, sampling_rate=48000)
    #inputs = {k: v.to(device) for k, v in inputs.items()}
    #outputs = model(**inputs)
    #logits_per_audio = outputs.logits_per_audio #.detach().cpu().numpy()  # this is the audio-text similarity score
    #probs = logits_per_audio.softmax(dim=-1).detach().cpu().numpy()
    #probs = logits_per_audio.detach().cpu().numpy()
    return probs

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def run(args):
    
    config_dict = {}
    with open(args.ref_json) as f:
        data = f.readlines()
    for line in data:
        input_file = json.loads(line)
        audio_base_name = input_file["location"].split("/")[-1].split(".")[0]
        # config_dict[audio_base_name] = input_file["captions"]
        config_dict[audio_base_name] = input_file[args.caption_keys]
        
    probs_all = 0.0
    count = 0
    
    for file_name in tqdm(os.listdir(args.generation_result_path)):
        if not file_name.endswith(".wav"):
            continue
        gen_wav = args.generation_result_path + "/" + file_name
        file_base_name = file_name.split(".")[0] 
        # import pdb; pdb.set_trace()
        #gen_wav = args.generation_result_path + "/" + ref_wav.split("/")[-1].split(".wav")[0] + "_0.wav"
         
        #gen_wav = "/xxxxxx/dataset/audiocaps/test/" + ref_wav.split("/")[-1]
        if file_base_name not in config_dict:
            print(file_base_name, "not in config_dict")
            continue
        
        ref_text = config_dict[file_base_name]

        audio_sample, sr = sf.read(gen_wav)
        if sr != 48000:
            audio_sample = librosa.resample(audio_sample, orig_sr=sr, target_sr=48000)

        input_text = [ref_text, ref_text]
        #print("input_text", input_text.size())
        if audio_sample.shape[0]==0:
            continue
        probs = get_clap_score(audio_sample, input_text)

        probs_all = probs_all + probs[0][0]
        count = count + 1
        #print("probs", probs, probs.softmax(dim=-1))

    print("clap score: ", args.generation_result_path, probs_all/count)
    

if __name__ == "__main__":

    # generation_result_path = "outputs/tango_video2audio_clip4clip_vggsound_filtered_2/1720270226_tango_video2audio_clip4clip_vggsound_filtered_2_best_steps_300_guidance_3.0_sampleRate_16000"
    
    parser = argparse.ArgumentParser(description="Command to do generate noisy speech for speech enhancement.",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--generation_result_path", type=str, default=None, help="Path to the generated audio")
    parser.add_argument("--ref_json", type=str, default="data/video_processed/valid_selected.json")
    parser.add_argument("--caption_keys", type=str, default="captions") 

    args = parser.parse_args()
    run(args)
