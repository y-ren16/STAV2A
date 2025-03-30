import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ConditioningEmbeddingForVideo_40_CVAP(nn.Module):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        block_out_channels: Tuple[int, ...] = (4, 8, 16, 32),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(1, 4, kernel_size=3, padding=1, stride=(1, 4))

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=(1, 4)))

        self.conv_out = nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1, stride=(1, 2))

    def forward(self, conditioning):
        # import pdb; pdb.set_trace()
        # B, 40, 512
        # B, 128, 256, 16
        conditioning = conditioning.float() 
        conditioning = conditioning.transpose(1, 2)
        scale_factor = 250 / 40
        embedding = F.interpolate(conditioning, scale_factor=scale_factor, mode='nearest')
        embedding = torch.cat((embedding, embedding[:,:,-1].unsqueeze(2).repeat(1, 1, 6)), dim=2)
        embedding = embedding.transpose(1, 2).unsqueeze(1) 
        embedding = self.conv_in(embedding)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding

class ConditioningEmbeddingForVideo_40_Multi(nn.Module):
    def __init__(
        self,
        conditioning_embedding_channels: int = 128,
    ):
        super().__init__()

        self.conv_in = nn.Conv1d(768, 256, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.conv_out = zero_module(nn.Conv2d(128, conditioning_embedding_channels, kernel_size=3, padding=1))


    def forward(self, conditioning):
        # import pdb; pdb.set_trace()
        # [B, 40, 768] -> [B, 128, 256, 16]
        # torch.Size([B, 128, 128, 8])
        # torch.Size([B, 64, 64, 4])
        # torch.Size([B, 32, 32, 2])
        # torch.Size([B, 16, 16, 1])
        conditioning = conditioning.float() 
        conditioning = conditioning.transpose(1, 2)
        scale_factor = 250 / 40
        embedding = F.interpolate(conditioning, scale_factor=scale_factor, mode='nearest')
        embedding = torch.cat((embedding, embedding[:,:,-1].unsqueeze(2).repeat(1, 1, 6)), dim=2)
        embedding = embedding
        # [B, 40, 768] -> [B, 768, 256]
        embedding = self.conv_in(embedding)
        embedding = F.silu(embedding)
        # [B, 768, 256] -> [B, 256, 256]
        embedding = self.conv_1(embedding)
        embedding = F.silu(embedding)
        # [B, 256, 256] -> [B, 128, 256]
        embedding = embedding.unsqueeze(-1).repeat(1, 1, 1, 16)
        # [B, 128, 256] -> [B, 128, 256, 16] repeat 16 times
        embedding = self.conv_out(embedding)

        return embedding

class BLSTMLayer(nn.Module):
    """ Wrapper over dilated conv1D
    Input tensor:  (batchsize=1, length, dim_in)
    Output tensor: (batchsize=1, length, dim_out)
    We want to keep the length the same
    """

    def __init__(self, input_dim, output_dim):
        super(BLSTMLayer, self).__init__()
        # bi-directional LSTM
        self.l_blstm = nn.LSTM(input_dim, output_dim // 2, bidirectional=True)

    def forward(self, x):
        # permute to (length, batchsize=1, dim)
        blstm_data, _ = self.l_blstm(x.permute(1, 0, 2))
        # permute it backt to (batchsize=1, length, dim)
        return blstm_data.permute(1, 0, 2)

class PreOnSet(nn.Module):
    def __init__(self, input_dim=32, output_dim=16):
        super(PreOnSet, self).__init__()
        self.before_pooling = nn.Sequential(
            BLSTMLayer(input_dim, output_dim),
            BLSTMLayer(output_dim, output_dim),
        )
        self.l1 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.before_pooling(x)
        x = self.l1(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_seq_len=16):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
        pe = torch.zeros(max_seq_len, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(CrossAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, video_token):
        batch_size = x.size(0)

        q = self.q_linear(x)
        k = self.k_linear(video_token)
        v = self.v_linear(video_token)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)

        output = self.out_linear(attn_output)
        return output

class PreOnSet_by_MultiScaleFeature(nn.Module):
    def __init__(self, input_dim):
        super(PreOnSet_by_MultiScaleFeature, self).__init__()
        self.l1 = nn.Linear(input_dim, 128)
        self.l2 = nn.Linear(128, 2)

    def forward(self, x):
        # input: [B, 40, input_dim] 
        # output: [B, 256, 2]
        x = x.transpose(1, 2)
        scale_factor = 250 / 40
        embedding = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
        embedding = torch.cat((embedding, embedding[:,:,-1].unsqueeze(2).repeat(1, 1, 6)), dim=2)
        # [B, 40, input_dim] -> [B, input_dim, 256]
        embedding = embedding.transpose(1, 2) 
        # [B, 256, input_dim]
        x1 = self.l1(embedding)
        x1 = F.silu(x1)
        # [B, 256, 128]
        logits = self.l2(x1) 
        return logits, x1

class PreOnSet_by_MultiScaleFeature_None(nn.Module):
    def __init__(self, input_dim):
        super(PreOnSet_by_MultiScaleFeature_None, self).__init__()
        self.l1 = nn.Linear(input_dim, 128)

    def forward(self, x):
        # input: [B, 40, input_dim] 
        # output: [B, 256, 2]
        x = x.transpose(1, 2)
        scale_factor = 250 / 40
        embedding = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
        embedding = torch.cat((embedding, embedding[:,:,-1].unsqueeze(2).repeat(1, 1, 6)), dim=2)
        # [B, 40, input_dim] -> [B, input_dim, 256]
        embedding = embedding.transpose(1, 2) 
        # [B, 256, input_dim]
        x1 = self.l1(embedding)
        x1 = F.silu(x1)
        # [B, 256, 128]

        return x1

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module