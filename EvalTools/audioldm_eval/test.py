import torch
from audioldm_eval import EvaluationHelper
import sys
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler
import argparse
 
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

def run(args):
    generation_result_path = args.generation_result_path 
    target_audio_path = args.gt_audio_path 

    evaluator = EvaluationHelper(16000, device)

    metrics = evaluator.main(
        generation_result_path,
        target_audio_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command to do generate noisy speech for speech enhancement.",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--generation_result_path", type=str, default=None, help="Path to the generated audio")
    parser.add_argument("--gt_audio_path", type=str, default=None, help="Path to the generated audio")
    args = parser.parse_args()
    run(args)