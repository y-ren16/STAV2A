import torch.nn as nn
from fga.atten import Atten
import torch
import math


class TransVideoFeature(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, output_size=128):
        super(TransVideoFeature, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.fc4 = nn.Linear(output_size, output_size)

        self.gelu = nn.GELU()


    def forward(self, video_embs):
        # [B, 40, 512] -> [B, 40, 128]
        video_embs = self.fc1(video_embs)
        video_embs = self.gelu(video_embs)
        video_embs = self.fc2(video_embs)
        video_embs = self.gelu(video_embs)
        video_embs = self.fc3(video_embs)
        video_embs = self.gelu(video_embs)
        video_embs = self.fc4(video_embs)

        local_window_1 = torch.cat([torch.unsqueeze(video_embs[:, max(i-1, 0):min(i+1, video_embs.shape[1]), :].mean(
            dim=1), dim=1) for i in range(video_embs.shape[1])], dim=1)
        local_window_2 = torch.cat([torch.unsqueeze(video_embs[:, max(i-2, 0):min(i+2, video_embs.shape[1]), :].mean(
            dim=1), dim=1) for i in range(video_embs.shape[1])], dim=1)
        local_window_3 = torch.cat([torch.unsqueeze(video_embs[:, max(i-4, 0):min(i+4, video_embs.shape[1]), :].mean(
            dim=1), dim=1) for i in range(video_embs.shape[1])], dim=1)
        local_window_4 = torch.cat([torch.unsqueeze(video_embs[:, max(i-8, 0):min(i+8, video_embs.shape[1]), :].mean(
            dim=1), dim=1) for i in range(video_embs.shape[1])], dim=1)

        return video_embs, local_window_1, local_window_2, local_window_3, local_window_4

class GlobalVideoFeature(nn.Module):
    def __init__(self, input_size=512):
        super(GlobalVideoFeature, self).__init__()
        self.fc = nn.Linear(input_size, 1024)
        self.gelu = nn.GELU()
        self.fga = Atten(util_e=[1024], pairwise_flag=False)
        self.conv = nn.Conv1d(1, 4, 3, padding=1)
    def forward(self, video_embs):
        embs_for_fga = self.fc(video_embs) 
        video_embs = self.gelu(video_embs)
        attend = self.fga([embs_for_fga])[0]
        # [B, 1024]
        attend = self.conv(attend.unsqueeze(1))
        # [B, 4, 1024]
        return attend
