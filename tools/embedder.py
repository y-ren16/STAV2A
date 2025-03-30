import torch.nn as nn
from fga.atten import Atten
import torch
import math


class TempEmbedder(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, output_size=128, class_token=True, seq_target_length=24):
        super(TempEmbedder, self).__init__()
        self.output_size = output_size
        self.class_token = class_token
        self.seq_target_length = seq_target_length

        self.fc = nn.Linear(input_size, 1024)
        self.conv = nn.Conv1d(1, 4, 3, padding=1)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.fc4 = nn.Linear(output_size, output_size)

        self.gelu = nn.GELU()
        if class_token:
            # self.fga = Atten(util_e=[output_size], pairwise_flag=False)
            self.fga = Atten(util_e=[1024], pairwise_flag=False)

    def forward(self, audio_embs):
        # import pdb; pdb.set_trace()

        bs, seq_len = audio_embs.shape[0], audio_embs.shape[1]
        embs_for_fga = self.fc(audio_embs)
        audio_embs = self.fc1(audio_embs)
        audio_embs = self.gelu(audio_embs)
        audio_embs = self.fc2(audio_embs)
        audio_embs = self.gelu(audio_embs)
        audio_embs = self.fc3(audio_embs)
        # audio_embs = audio_embs.view(bs, self.seq_target_length, -1, self.output_size).mean(dim=2)
        audio_embs = self.gelu(audio_embs)
        audio_embs = self.fc4(audio_embs)

        attend = None
        if self.class_token:
            attend = self.fga([embs_for_fga])[0] # [B, 1024]
            attend = self.conv(attend.unsqueeze(1))

        local_window_1 = torch.cat([torch.unsqueeze(audio_embs[:, max(i-1, 0):min(i+1, audio_embs.shape[1]), :].mean(
            dim=1), dim=1) for i in range(audio_embs.shape[1])], dim=1)
        local_window_2 = torch.cat([torch.unsqueeze(audio_embs[:, max(i-2, 0):min(i+2, audio_embs.shape[1]), :].mean(
            dim=1), dim=1) for i in range(audio_embs.shape[1])], dim=1)
        local_window_3 = torch.cat([torch.unsqueeze(audio_embs[:, max(i-4, 0):min(i+4, audio_embs.shape[1]), :].mean(
            dim=1), dim=1) for i in range(audio_embs.shape[1])], dim=1)
        local_window_4 = torch.cat([torch.unsqueeze(audio_embs[:, max(i-8, 0):min(i+8, audio_embs.shape[1]), :].mean(
            dim=1), dim=1) for i in range(audio_embs.shape[1])], dim=1)

        return audio_embs, local_window_1, local_window_2, local_window_3, local_window_4, attend
