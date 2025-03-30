import torch
from audioldm_eval import EvaluationHelper

import sys
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler
import os
 
default_collate_func = dataloader.default_collate
 
def default_collate_override(batch):
  dataloader._use_shared_memory = False
  return default_collate_func(batch)
 
setattr(dataloader, 'default_collate', default_collate_override)
 
for t in torch._storage_classes:
  if sys.version_info[0] == 2:
    if t in ForkingPickler.dispatch:
        del ForkingPickler.dispatch[t]
  else:
    if t in ForkingPickler._extra_reducers:
        del ForkingPickler._extra_reducers[t]

device = torch.device(f"cuda:{0}")

# base_path = "/xxxxxxx/experiments/audio_encoder/gaudio_16k_multilingual_vq_beats_pt/outputs"
# base_path = "/xxxxxxx/experiments/audio_encoder/gaudio_16k_multilingual_vq_clap/outputs"
base_path = "/xxxxxxx/experiments/audio_encoder/gaudio_16k_multilingual_vq_beats_pt/outputs"
# base_path = "/xxxxxxx/experiments/audio_encoder/gaudio_16k_multilingual_vq_clap/outputs"
generation_result_paths = []
# find folder in base_path that starts with "1727"
for generation_result_path in os.listdir(base_path):
    if generation_result_path.startswith("1727"):
        generation_result_paths.append(generation_result_path)
print(generation_result_paths)
print(f'len(generation_result_paths): {len(generation_result_paths)}')

for generation_result_path in generation_result_paths:
    generation_result_path = f"{base_path}/{generation_result_path}"
    # target_audio_path = "/xxxxxxx/intern/yongren/v2a/v2a_pretrain/data/New_Audio_GT_Test_VGGSound_Filter_wav"
    target_audio_path = "/xxxxxx/dataset/audiocaps/test16k"

    evaluator = EvaluationHelper(16000, device)

    # Perform evaluation, result will be print out and saved as json
    metrics = evaluator.main(
        generation_result_path,
        target_audio_path,
    )
