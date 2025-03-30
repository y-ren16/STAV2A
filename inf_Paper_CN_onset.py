import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import copy
import json
import time
import torch
import argparse
import soundfile as sf
#import wandb
from tqdm import tqdm
from diffusers import DDPMScheduler
# from audioldm_eval import EvaluationHelper
from models_paper_CN_onset import build_pretrained_models, AudioDiffusion
from transformers import AutoProcessor, ClapModel
import torchaudio
import tools.torch_tools as torch_tools

from moviepy.editor import VideoFileClip
from moviepy.audio.AudioClip import AudioArrayClip
import librosa
import sys
import numpy as np
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

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for text to audio generation task.")
    parser.add_argument(
        "--original_args", type=str, default=None,
        help="Path for summary jsonl file saved during training."
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path for saved model bin file."
    )
    parser.add_argument(
        "--vae_model", type=str, default="audioldm-s-full",
        help="Path for saved model bin file."
    )
    parser.add_argument(
        "--test_file", type=str, default="data/test_audiocaps_subset.json",
        help="json file containing the test prompts for generation."
    )
    parser.add_argument(
        "--text_key", type=str, default="captions",
        help="Key containing the text in the json file."
    )
    parser.add_argument(
        "--audio_key", type=str, default="location",
        help="Key containing the audio in the json file."
    )
    parser.add_argument(
        "--video_key", type=str, default="video_location",
        help="Key containing the video in the json file."
    )
    parser.add_argument(
        "--video_feature_key", type=str, default=None,
        help="Key containing the video feature in the json file."
    )
    parser.add_argument(
        "--test_references", type=str, default="data/audiocaps_test_references/subset",
        help="Folder containing the test reference wav files."
    )
    parser.add_argument(
        "--num_steps", type=int, default=200,
        help="How many denoising steps for generation.",
    )
    parser.add_argument(
        "--guidance", type=float, default=3,
        help="Guidance scale for classifier free guidance."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1,
        help="How many samples per prompt.",
    )
    parser.add_argument(
        "--num_test_instances", type=int, default=-1,
        help="How many test instances to evaluate.",
    )
    parser.add_argument(
        "--clap_model", type=str, default="/xxxxxx/experiments/pretrain_models/clap/clap-htsat-unfused",
        help="How many test instances to evaluate.",
    )
    parser.add_argument(
        "--save_dir", type=str, default="./outputs/tmp",
        help="output save dir"
    )
    parser.add_argument(
        "--video_fps", type=int, default=8,
        help="video fps",
    ) 
    parser.add_argument(
        "--has_global_video_feature", action="store_true", default=False,
        help="global video feature",
    )   
    parser.add_argument(
        "--use_feature_window", action="store_true", default=False,
        help="use feature window",
    ) 

    
    args = parser.parse_args()

    return args

def main():
    # import pdb; pdb.set_trace()
    args = parse_args()
    
    train_args = dotdict(json.loads(open(args.original_args).readlines()[0]))
    if "hf_model" not in train_args:
        train_args["hf_model"] = None
    
    # Load Models #
    #name = "audioldm-s-full"
    name = args.vae_model
    vae, stft = build_pretrained_models(name)
    vae, stft = vae.cuda(), stft.cuda()
    print("222", train_args.unet_model_name)
    model = AudioDiffusion(
        train_args.text_encoder_name, train_args.scheduler_name, train_args.unet_model_name, train_args.unet_model_config, train_args.snr_gamma, video_fps=args.video_fps, has_global_video_feature=args.has_global_video_feature, use_feature_window=args.use_feature_window
    ).cuda()
    model.eval()
    # import pdb; pdb.set_trace()
    
    # Load Trained Weight #
    device = vae.device()
    if args.model.endswith(".pt") or args.model.endswith(".bin"):
        model.load_state_dict(torch.load(args.model), strict=False)
    else:
        from safetensors.torch import load_model
        load_model(model, args.model, strict=False)

    # model.load_state_dict(torch.load(args.model))
    
    scheduler = DDPMScheduler.from_pretrained(train_args.scheduler_name, subfolder="scheduler")
    #evaluator = EvaluationHelper(16000, "cuda:0")
    
    if args.num_samples > 1:
        clap = ClapModel.from_pretrained(args.clap_model).to(device)
        clap.eval()
        clap_processor = AutoProcessor.from_pretrained(args.clap_model)
    
    #wandb.init(project="Text to Audio Diffusion Evaluation")

    def audio_text_matching(waveforms, text, sample_freq=16000, max_len_in_seconds=10):
        new_freq = 48000
        resampled = []
        
        for wav in waveforms:
            x = torchaudio.functional.resample(torch.tensor(wav, dtype=torch.float).reshape(1, -1), orig_freq=sample_freq, new_freq=new_freq)[0].numpy()
            resampled.append(x[:new_freq*max_len_in_seconds])

        inputs = clap_processor(text=text, audios=resampled, return_tensors="pt", padding=True, sampling_rate=48000)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = clap(**inputs)

        logits_per_audio = outputs.logits_per_audio
        ranks = torch.argsort(logits_per_audio.flatten(), descending=True).cpu().numpy()
        return ranks
    
    # Load Data #
    if train_args.prefix:
        prefix = train_args.prefix
    else:
        prefix = ""
    
    text_prompts = []
    video = []
    wavname = []
    video_feature_paths = []
    with open(args.test_file) as f:
       for line in f:
           utt = line.strip()
           input_file = json.loads(utt)
           if input_file[args.text_key] is None:
               print("None text key: ", input_file["location"])
               continue
           text_prompts.append(input_file[args.text_key])
           video.append(input_file[args.video_key])
           wavname.append(os.path.basename(input_file[args.video_key].replace(".mp4", ".wav")))
           if args.video_feature_key is not None:
                video_feature_paths.append(audio_feature_path)

    # with open(args.test_file) as f:
    #     data = f.read()
    # input_file = json.loads(data)
    # text_prompts = [line[args.text_key] for line in input_file["data"]]
    # wavname = [os.path.basename(line["wav"]) for line in input_file["data"]]

    try:
        text_prompts = [prefix + inp for inp in text_prompts]
        # test_prompts = []
        # for inp in text_prompts:
        #     if inp is None:
        #         import pdb; pdb.set_trace()
        #     test_prompts.append(prefix + inp)

    except:
        import pdb; pdb.set_trace()
    
    if args.num_test_instances != - 1:
        text_prompts = text_prompts[:args.num_test_instances]
    
    # Generate #
    num_steps, guidance, batch_size, num_samples = args.num_steps, args.guidance, args.batch_size, args.num_samples
    all_outputs = []
        
    for k in tqdm(range(0, len(text_prompts), batch_size)):
        # import pdb; pdb.set_trace()
        text = text_prompts[k: k+batch_size]
        video_paths = video[k: k+batch_size]
        if args.video_feature_key is not None:
            # video_feature = torch.cat([torch_tools.read_video_from_file(v_path, args.video_fps).unsqueeze(0) for v_path in video_paths], 0)
            video_feature_path = video_feature_paths[k: k+batch_size]
        else:
            # video_feature = None
            video_feature_path = None
        
        with torch.no_grad():
            latents = model.inference(text, video_paths, video_feature_path, scheduler, num_steps, guidance, num_samples, disable_progress=True)
            # latents = model.inference(text, video_paths, None, scheduler, num_steps, guidance, num_samples, disable_progress=True)
            mel = vae.decode_first_stage(latents)
            wave = vae.decode_to_waveform(mel)
            all_outputs += [item for item in wave]
            
    # Save #
    exp_id = str(int(time.time()))
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    test_name = os.path.basename(args.test_file).split(".")[0]
    
    if num_samples == 1:
        output_dir = "{}/{}_{}_steps_{}_guidance_{}_{}".format(args.save_dir, exp_id, "_".join(args.model.split("/")[1:-1]), num_steps, guidance, test_name)
        os.makedirs(output_dir, exist_ok=True)
        for j, wav in enumerate(all_outputs):
            basename = os.path.basename(wavname[j])
            sf.write("{}/{}".format(output_dir, wavname[j]), wav, samplerate=16000)

            try:
                video_target = VideoFileClip(video[j])
                duration = video_target.duration
                audio_target = librosa.load(f"{output_dir}/{wavname[j]}", sr=16000)[0] 
                if len(audio_target) < 16000 * int(duration):
                    audio_target = np.pad(audio_target, (0, 16000 * int(duration) - len(audio_target)))
                resampled_audio = librosa.resample(audio_target[:16000 * int(duration)], orig_sr=16000, target_sr=44100)
                audio_stereo = np.vstack((resampled_audio, resampled_audio)).T
                audio_clip = AudioArrayClip(audio_stereo, fps=44100)
                with MuteOutput():
                    video_target.write_videofile("{}/{}_origin.mp4".format(output_dir, basename.split(".")[0]), codec="libx264", audio_codec="aac")
                    video_target = video_target.set_audio(audio_clip)
                    video_target.write_videofile("{}/{}.mp4".format(output_dir, basename.split(".")[0]), codec="libx264", audio_codec="aac")
                video_target.close()
                audio_clip.close()
            except Exception as e:
                print(e)
    else:
        for i in range(num_samples):
            output_dir = "{}/{}_{}_steps_{}_guidance_{}_{}/rank_{}".format(args.save_dir, exp_id, "_".join(args.model.split("/")[1:-1]), num_steps, guidance, test_name, i+1)
            os.makedirs(output_dir, exist_ok=True)
        
        groups = list(chunks(all_outputs, num_samples))
        for k in tqdm(range(len(groups))):
            wavs_for_text = groups[k]
            rank = audio_text_matching(wavs_for_text, text_prompts[k])
            ranked_wavs_for_text = [wavs_for_text[r] for r in rank]
            
            for i, wav in enumerate(ranked_wavs_for_text):
                output_dir = "{}/{}_{}_steps_{}_guidance_{}_{}/rank_{}".format(args.save_dir, exp_id, "_".join(args.model.split("/")[1:-1]), num_steps, guidance, test_name, i+1)
                sf.write("{}/output_{}.wav".format(output_dir, k), wav, samplerate=16000)

                try:
                    video_target = VideoFileClip(video[k])
                    duration = video_target.duration
                    audio_target = librosa.load(f"{output_dir}/output_{k}.wav", sr=16000)[0]
                    if len(audio_target) < 16000 * int(duration):
                        audio_target = np.pad(audio_target, (0, 16000 * int(duration) - len(audio_target)))
                    resampled_audio = librosa.resample(audio_target[:16000 * int(duration)], orig_sr=16000, target_sr=44100)
                    audio_stereo = np.vstack((resampled_audio, resampled_audio)).T
                    audio_clip = AudioArrayClip(audio_stereo, fps=44100)
                    with MuteOutput():
                        video_target.write_videofile("{}/output_{}_origin.mp4".format(output_dir, k), codec="libx264", audio_codec="aac")
                        video_target = video_target.set_audio(audio_clip)
                        video_target.write_videofile("{}/output_{}.mp4".format(output_dir, k), codec="libx264", audio_codec="aac")
                    video_target.close()
                    audio_clip.close()
                except Exception as e:
                    print(e)
        
if __name__ == "__main__":
    main()
