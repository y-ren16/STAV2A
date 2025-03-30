import yaml
import random
import inspect
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat
import time
from tools.torch_tools import wav_to_fbank
# from tools.feature_trans import ConvBlockRes

from audioldm.audio.stft import TacotronSTFT
from audioldm.variational_autoencoder import AutoencoderKL
from audioldm.utils import default_audioldm_config, get_metadata
# from languagebind import LanguageBindVideo, LanguageBindVideoTokenizer, LanguageBindVideoProcessor

from transformers import CLIPTokenizer, AutoTokenizer, T5Tokenizer
from transformers import CLIPTextModel, T5EncoderModel, AutoModel

import diffusers
# from diffusers.utils import randn_tensor
from diffusers.utils.torch_utils import randn_tensor
# from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers import (
    # AutoencoderKL,
    # ControlNetFDNModel,
    ControlNetMultiModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
import os
# from diffusers import AutoencoderKL as DiffuserAutoencoderKL
from cvap.demo_util import Extract_CAVP_Features 
from tools.feature_trans import ConditioningEmbeddingForVideo_40_CVAP, PreOnSet, PreOnSet_by_MultiScaleFeature
# from tools.embedder import TempEmbedder
from tools.video_feature_trans import TransVideoFeature, GlobalVideoFeature


def build_pretrained_models(name):
    print("111", get_metadata()[name]["path"])
    checkpoint = torch.load(get_metadata()[name]["path"], map_location="cpu")
    scale_factor = checkpoint["state_dict"]["scale_factor"].item()

    vae_state_dict = {k[18:]: v for k, v in checkpoint["state_dict"].items() if "first_stage_model." in k}

    config = default_audioldm_config(name)
    vae_config = config["model"]["params"]["first_stage_config"]["params"]
    vae_config["scale_factor"] = scale_factor

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(vae_state_dict)

    fn_STFT = TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )

    vae.eval()
    fn_STFT.eval()
    return vae, fn_STFT


class AudioDiffusion(nn.Module):
    def __init__(
        self,
        text_encoder_name,
        scheduler_name,
        unet_model_name=None,
        unet_model_config_path=None,
        snr_gamma=None,
        freeze_text_encoder=True,
        uncondition=False,
        video_fps=8,
        fraze_unet=True,
        predict_onset_model=None,
        has_global_video_feature=False,
        use_feature_window=False,
        guidance_free_rate=0.1,

    ):
        super().__init__()

        assert unet_model_name is not None or unet_model_config_path is not None, "Either UNet pretrain model name or a config file path is required"

        self.text_encoder_name = text_encoder_name
        self.scheduler_name = scheduler_name
        self.unet_model_name = unet_model_name
        self.unet_model_config_path = unet_model_config_path
        self.snr_gamma = snr_gamma
        self.freeze_text_encoder = freeze_text_encoder
        self.uncondition = uncondition
        self.guidance_free_rate = guidance_free_rate

        # https://huggingface.co/docs/diffusers/v0.14.0/en/api/schedulers/overview
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.scheduler_name, subfolder="scheduler")
        self.inference_scheduler = DDPMScheduler.from_pretrained(self.scheduler_name, subfolder="scheduler")

        print("333 unet_model_name", unet_model_name)
        self.unet = UNet2DConditionModel.from_pretrained(unet_model_name, subfolder="unet", use_safetensors=False)
        print("UNet initialized from pre-trained tta checkpoint.")

        # import pdb; pdb.set_trace()
        self.controlnet = ControlNetMultiModel.from_unet(self.unet, video_fps=video_fps)
        print("initialized controlnet weights from unet")

        if "stable-diffusion" in self.text_encoder_name:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.text_encoder_name, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(self.text_encoder_name, subfolder="text_encoder")
        elif "t5" in self.text_encoder_name and "Chinese" not in self.text_encoder_name:
            # import pdb; pdb.set_trace()
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)
            self.text_encoder = T5EncoderModel.from_pretrained(self.text_encoder_name)
        elif "Chinese" in self.text_encoder_name:
            self.tokenizer = T5Tokenizer.from_pretrained(self.text_encoder_name)
            self.text_encoder = T5EncoderModel.from_pretrained(self.text_encoder_name)
        elif "clap" in self.text_encoder_name:
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            self.CLAP_model = laion_clap.CLAP_Module(enable_fusion=False)
            self.CLAP_model.load_ckpt(self.text_encoder_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)
            self.text_encoder = AutoModel.from_pretrained(self.text_encoder_name)
        
        if fraze_unet:
            self.unet.requires_grad_(False)
        else:
            self.unet.train()
        self.text_encoder.requires_grad_(False)
        self.controlnet.train()

        # pretrained_ckpt = '/xxxxxxx/experiments/pretrain_models/languagebind/LanguageBind_Video_FT'  # also 'LanguageBind/LanguageBind_Video'
        # self.v_model = LanguageBindVideo.from_pretrained(pretrained_ckpt) 
        # self.v_tokenizer = LanguageBindVideoTokenizer.from_pretrained(pretrained_ckpt)
        # self.v_video_process = LanguageBindVideoProcessor(self.v_model.config, self.v_tokenizer)
        # # self.video_reshape = ConvBlockRes(in_channels=1024)
        # self.v_model.eval()
        # self.v_model.requires_grad_(False)
        # # self.video_reshape.requires_grad_(True)
        print(self.text_encoder.device)
        # import pdb; pdb.set_trace()
        self.extract_cavp = Extract_CAVP_Features(fps=4, batch_size=40, device=self.text_encoder.device, config_path="./cvap/Stage1_CAVP.yaml", ckpt_path="./cvap/cavp_epoch66.ckpt")
        self.extract_cavp.eval()
        self.extract_cavp.requires_grad_(False) 
        # self.condition_embedding = ConditioningEmbeddingForVideo_40_CVAP(conditioning_embedding_channels=32)
        # self.pre_onset = PreOnSet()
        # self.at_embedder = TempEmbedder()
        self.trans_cvap_feature = TransVideoFeature()
        self.use_feature_window = use_feature_window
        self.has_global_video_feature = has_global_video_feature
        if self.has_global_video_feature:
            self.global_video_feature = GlobalVideoFeature()
        if use_feature_window:
            self.pre_onset = PreOnSet_by_MultiScaleFeature(input_dim=128*5)
            print("Use Feature Window")
        else: 
            self.pre_onset = PreOnSet_by_MultiScaleFeature(input_dim=128)
            print("Not Use Feature Window")
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.predict_onset_model = predict_onset_model
        # import pdb; pdb.set_trace()
        if predict_onset_model is not None:
            onset_predict_model = torch.load(predict_onset_model, map_location="cpu")
            # at_embedder_model = {}
            trans_cvap_feature_model = {}
            pre_onset_model = {} 
            for key in onset_predict_model.keys():
                # if "at_embedder" in key:
                #     at_embedder_model[key.replace("at_embedder.", "")] = onset_predict_model[key]
                if "trans_cvap_feature" in key:
                    trans_cvap_feature_model[key.replace("trans_cvap_feature.", "")] = onset_predict_model[key]
                elif "pre_onset" in key:
                    pre_onset_model[key.replace("pre_onset.", "")] = onset_predict_model[key]
            
            # missing, unexpected = self.at_embedder.load_state_dict(at_embedder_model, strict=False)
            # print(f"Restored from {predict_onset_model} with {len(missing)} missing and {len(unexpected)} unexpected keys")
            # if len(missing) > 0:
            #     print(f"Missing Keys: {missing}")
            # if len(unexpected) > 0:
            #     print(f"Unexpected Keys: {unexpected}")

            missing, unexpected = self.trans_cvap_feature.load_state_dict(trans_cvap_feature_model, strict=False)
            print(f"Restored from {predict_onset_model} with {len(missing)} missing and {len(unexpected)} unexpected keys")
            if len(missing) > 0:
                print(f"Missing Keys: {missing}")
            if len(unexpected) > 0:
                print(f"Unexpected Keys: {unexpected}")
            
            missing, unexpected = self.pre_onset.load_state_dict(pre_onset_model, strict=False)
            print(f"Restored from {predict_onset_model} with {len(missing)} missing and {len(unexpected)} unexpected keys")
            if len(missing) > 0:
                print(f"Missing Keys: {missing}")
            if len(unexpected) > 0:
                print(f"Unexpected Keys: {unexpected}")
            
            # self.at_embedder.requires_grad_(False)
            self.trans_cvap_feature.requires_grad_(False)
            self.pre_onset.requires_grad_(False)

    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    def encode_text(self, prompt):
        device = self.text_encoder.device
        #batch = self.tokenizer(
        #    prompt, max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt"
        #)

        batch = self.tokenizer(
            prompt, max_length=200, padding=True, truncation=True, return_tensors="pt"
        )

        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

        if self.freeze_text_encoder:
            with torch.no_grad():
                encoder_hidden_states = self.text_encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                )[0]
        else:
            encoder_hidden_states = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]

        boolean_encoder_mask = (attention_mask == 1).to(device)
        return encoder_hidden_states, boolean_encoder_mask

    def encode_text_CLAP(self, prompt):
        device = self.text_encoder.device
        batch = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length, return_tensors="pt")
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

        if self.freeze_text_encoder:
            with torch.no_grad():
                encoder_hidden_states = self.CLAP_model.model.get_text_embedding(prompt)
        else:
            encoder_hidden_states = self.CLAP_model.model.get_text_embedding(prompt)

        boolean_encoder_mask = (attention_mask == 1).to(device)
        return encoder_hidden_states, boolean_encoder_mask

    def encode_video(self, video_paths, feature_paths=None):
        # import pdb; pdb.set_trace()
        if feature_paths is not None:
            feature_exist = True
            # Note: Only First Time RUN or TESTING need
            for feature_path in feature_paths:
                if not os.path.exists(feature_path):
                    feature_exist = False
                    break
        else:
            feature_exist = False
        if feature_exist:
            # import pdb; pdb.set_trace()
            try:
                encode_video_hidden = torch.cat([torch.load(feature_path).unsqueeze(0) for feature_path in feature_paths], 0)
            except:
                import pdb; pdb.set_trace()
            encode_video_hidden = encode_video_hidden.to(self.text_encoder.device)
        else:
            device = self.text_encoder.device
            # video_feature = video_feature.to(device)
            # self.v_model = self.v_model.to(device)
            # self.v_model.eval()
            # with torch.no_grad():
            #     encode_video_hidden = self.v_model(pixel_values=video_feature)
            with torch.no_grad():
                encode_video_hidden_list = []
                for video_path in video_paths:
                    os.makedirs("./cvap_temp_cache", exist_ok=True)
                    # encode_video_hidden_temp, _ = self.extract_cavp(video_path, device, start_second=0, truncate_second=10, tmp_path="./cvap_temp_cache")
                    if video_path.endswith(".mp4"):
                        encode_video_hidden_temp, _ = self.extract_cavp.forward_from_mp4(video_path, device, start_second=0, truncate_second=10)
                    elif video_path.endswith(".pt"):
                        encode_video_hidden_temp = self.extract_cavp(video_path, device, start_second=0, truncate_second=10)
                    else:
                        raise ValueError("Video format not supported")
                    encode_video_hidden_list.append(torch.from_numpy(encode_video_hidden_temp).to(device))
                # encode_video_hidden = torch.cat(encode_video_hidden_list.unsqueeze(0), 0)
                encode_video_hidden = torch.stack(encode_video_hidden_list, 0) 
            if feature_paths is not None:
                for i, feature_path in enumerate(feature_paths):
                    torch.save(encode_video_hidden[i].cpu(), feature_path)
            # import pdb; pdb.set_trace()
        return encode_video_hidden

    def forward(self, latents, prompt, video_paths, feature_paths, audio_sound_starts, validation_mode=False):
        device = self.text_encoder.device
        audio_sound_starts = audio_sound_starts.to(dtype=torch.int64)
        num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_train_timesteps, device=device)

        encoder_hidden_states, boolean_encoder_mask = self.encode_text(prompt)
        
        if self.uncondition:
            mask_indices = [k for k in range(len(prompt)) if random.random() < self.guidance_free_rate]
            #mask_indices = [k for k in range(len(prompt))]
            if len(mask_indices) > 0:
                encoder_hidden_states[mask_indices] = 0

        bsz = latents.shape[0]

        if validation_mode:
            timesteps = (self.noise_scheduler.config.num_train_timesteps//2) * torch.ones((bsz,), dtype=torch.int64, device=device)
        else:
            # Sample a random timestep for each instance
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
        timesteps = timesteps.long()

        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # import pdb; pdb.set_trace()

        controlnet_video = self.encode_video(video_paths, feature_paths).to(dtype=latents.dtype)
        video_length = controlnet_video.shape[1]
        # temporal_token, _, _, _, _, video_token = self.at_embedder(controlnet_video) 
        if self.use_feature_window:
            temporal_token, local_window_1, local_window_2, local_window_3, local_window_4 = self.trans_cvap_feature(controlnet_video)
            video_feature_token = torch.cat([temporal_token, local_window_1, local_window_2, local_window_3, local_window_4], dim=-1)
        else:
            temporal_token, _, _, _, _= self.trans_cvap_feature(controlnet_video)
            video_feature_token = temporal_token
        if self.has_global_video_feature:
            video_token = self.global_video_feature(controlnet_video)
            encoder_hidden_states = torch.cat([encoder_hidden_states, video_token], dim=1) 
            boolean_encoder_mask = torch.cat([boolean_encoder_mask, torch.ones(bsz, 4).to(device)], dim=1)
        # [B, 40, 128], [B, 40, 128], [B, 40, 128], [B, 40, 128], [B, 40, 128], [B, 4, 1024] 
        onset_logits, video_feature_new  = self.pre_onset(video_feature_token) 
        # video_feature_new [B, 256, 128]
        loss_onset = self.criterion(onset_logits.view(-1, 2), audio_sound_starts.view(-1))

        # video_latents_reshape = self.video_reshape(controlnet_video)
        # encoder_hidden_states.shape [B, 38, 1024]
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=video_feature_new,
            return_dict=False,
        )

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        #print("size", noisy_latents.size(), target.size())
        # model_pred = self.unet(
        #     noisy_latents, timesteps, encoder_hidden_states, 
        #     encoder_attention_mask=boolean_encoder_mask
        # ).sample
        model_output = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=boolean_encoder_mask,
            down_block_additional_residuals=[
                sample for sample in down_block_res_samples
            ],
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )
        model_pred = model_output[0]

        if self.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            loss_onset = torch.mean(loss_onset)
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Adaptef from huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()
            loss_onset = torch.mean(loss_onset)

        return loss, loss_onset

    @torch.no_grad()
    def inference(self, prompt, video_paths, feature_paths, inference_scheduler, num_steps=20, guidance_scale=3, num_samples_per_prompt=1, duration=10.24, 
                  disable_progress=True):
        start = time.time()
        device = self.text_encoder.device
        classifier_free_guidance = guidance_scale > 1.0
        batch_size = len(prompt) * num_samples_per_prompt

        #print("ldm time 0", time.time()-start, prompt)
        if classifier_free_guidance:
            prompt_embeds, boolean_prompt_mask = self.encode_text_classifier_free(prompt, num_samples_per_prompt)
        else:
            prompt_embeds, boolean_prompt_mask = self.encode_text(prompt)
            prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
            boolean_prompt_mask = boolean_prompt_mask.repeat_interleave(num_samples_per_prompt, 0)

        #print("ldm time 1", time.time()-start)
        inference_scheduler.set_timesteps(num_steps, device=device)
        timesteps = inference_scheduler.timesteps

        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(batch_size, inference_scheduler, num_channels_latents, duration, prompt_embeds.dtype, device)

        controlnet_video = self.encode_video(video_paths, feature_paths).to(dtype=latents.dtype)
        # temporal_token, _, _, _, _, video_token = self.at_embedder(controlnet_video) 
        if self.use_feature_window:
            temporal_token, local_window_1, local_window_2, local_window_3, local_window_4 = self.trans_cvap_feature(controlnet_video)
            video_feature_token = torch.cat([temporal_token, local_window_1, local_window_2, local_window_3, local_window_4], dim=-1)
        else:
            temporal_token, _, _, _, _= self.trans_cvap_feature(controlnet_video)
            video_feature_token = temporal_token
        if self.has_global_video_feature:
            video_token = self.global_video_feature(controlnet_video)
            video_token = torch.cat([video_token] * 2) if classifier_free_guidance else video_token
            prompt_embeds = torch.cat([prompt_embeds, video_token], dim=1) 
            boolean_prompt_mask = torch.cat([boolean_prompt_mask, torch.ones(boolean_prompt_mask.shape[0], 4).to(device)], dim=1)
        # import pdb; pdb.set_trace()
        onset_logits, video_feature_new  = self.pre_onset(video_feature_token)
        video_feature_new = torch.cat([video_feature_new] * 2) if classifier_free_guidance else video_feature_new
        video_feature_new = video_feature_new.repeat_interleave(num_samples_per_prompt, 0)

        num_warmup_steps = len(timesteps) - num_steps * inference_scheduler.order
        progress_bar = tqdm(range(num_steps), disable=disable_progress)

        #print("ldm time 2", time.time()-start, timesteps)
        for i, t in enumerate(timesteps):
            # import pdb; pdb.set_trace()
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if classifier_free_guidance else latents
            latent_model_input = inference_scheduler.scale_model_input(latent_model_input, t)
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=video_feature_new,
                return_dict=False,
            ) 

            #print("ldm emu", i, time.time()-start)
            # noise_pred = self.unet(
            #     latent_model_input, t, encoder_hidden_states=prompt_embeds,
            #     encoder_attention_mask=boolean_prompt_mask
            # ).sample
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=boolean_prompt_mask,
                down_block_additional_residuals=[
                    sample for sample in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]

            # perform guidance
            if classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = inference_scheduler.step(noise_pred, t, latents).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % inference_scheduler.order == 0):
                progress_bar.update(1)

        #print("ldm time 3", time.time()-start)
        return latents

    def prepare_latents(self, batch_size, inference_scheduler, num_channels_latents, duration, dtype, device):

        length = int(duration * 16000/(160*4))
        #print("length", length)
        shape = (batch_size, num_channels_latents, length, 16)

        #shape = (batch_size, num_channels_latents, 256, 16)
        latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * inference_scheduler.init_noise_sigma
        return latents

    def encode_text_classifier_free(self, prompt, num_samples_per_prompt):
        device = self.text_encoder.device
        #batch = self.tokenizer(
        #    prompt, max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt"
        #)

        batch = self.tokenizer(
            prompt, max_length=200, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]
                
        prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        attention_mask = attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        # get unconditional embeddings for classifier free guidance
        uncond_tokens = [""] * len(prompt)

        max_length = prompt_embeds.shape[1]
        uncond_batch = self.tokenizer(
            uncond_tokens, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt",
        )
        uncond_input_ids = uncond_batch.input_ids.to(device)
        uncond_attention_mask = uncond_batch.attention_mask.to(device)

        with torch.no_grad():
            negative_prompt_embeds = self.text_encoder(
                input_ids=uncond_input_ids, attention_mask=uncond_attention_mask
            )[0]
                
        negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        uncond_attention_mask = uncond_attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        # For classifier free guidance, we need to do two forward passes.
        # We concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_mask = torch.cat([uncond_attention_mask, attention_mask])
        boolean_prompt_mask = (prompt_mask == 1).to(device)

        return prompt_embeds, boolean_prompt_mask
