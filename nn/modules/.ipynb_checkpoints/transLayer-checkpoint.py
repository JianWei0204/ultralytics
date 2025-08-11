import torch.nn as nn
from .trans import Transformer
import torch

__all__ = "Adjust_Transformer"

class Adjust_Transformer(nn.Module):
    def __init__(self, in_channels=256,out_channels=None):
        super(Adjust_Transformer, self).__init__()
        channel = in_channels if out_channels is None else out_channels
        self.row_embed = nn.Embedding(50, channel//2)
        self.col_embed = nn.Embedding(50, channel//2)
        self.reset_parameters()

        self.transformer = Transformer(channel, nhead = 8, num_encoder_layers = 1, num_decoder_layers = 0)

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x_f):
        # adjust search features
        device = x_f.device
        h, w = x_f.shape[-2:]
        # i = torch.arange(w).cuda()
        # j = torch.arange(h).cuda()
        i = torch.arange(w, device=device)  # 使用输入张量的设备
        j = torch.arange(h, device=device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
            ], dim= -1).permute(2, 0, 1).unsqueeze(0).repeat(x_f.shape[0], 1, 1, 1)
        b, c, h, w = x_f.size()
        x_seq=(pos+x_f).view(b, c, -1).permute(2, 0, 1)
        x_f = self.transformer(x_seq,x_seq,x_seq,h=h,w=w)
        x_f = x_f.permute(1, 2, 0).view(b, c, h, w)
        return x_f