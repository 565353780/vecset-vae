import torch
from torch import nn


class FP32LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_dtype = x.dtype
        return super().forward(x.float()).to(orig_dtype)
