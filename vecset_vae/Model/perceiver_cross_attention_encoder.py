import torch
import numpy as np
from torch import nn
from einops import repeat
from typing import Optional
from torch_cluster import fps

from vecset_vae.Model.Layer.fp32_layer_norm import FP32LayerNorm
from vecset_vae.Model.Layer.fourier_embedder import FourierEmbedder
from vecset_vae.Model.Transformer.attention import ResidualCrossAttentionBlock
from vecset_vae.Model.Transformer.perceiver_1d import Perceiver


class PerceiverCrossAttentionEncoder(nn.Module):
    def __init__(
        self,
        use_downsample: bool,
        num_latents: int,
        embedder: FourierEmbedder,
        point_feats: int,
        input_sharp_pc: bool,
        embed_point_feats: bool,
        width: int,
        heads: int,
        layers: int,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
        use_ln_post: bool = False,
        use_flash: bool = False,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents
        self.use_downsample = use_downsample
        self.input_sharp_pc = input_sharp_pc
        self.embed_point_feats = embed_point_feats

        if not self.use_downsample:
            self.query = nn.Parameter(torch.randn((num_latents, width)) * 0.02)

        self.embedder = embedder
        if self.embed_point_feats:
            self.input_proj = nn.Linear(self.embedder.out_dim * 2, width)
        else:
            self.input_proj = nn.Linear(self.embedder.out_dim + point_feats, width)
            if self.input_sharp_pc:
                self.input_proj1 = nn.Linear(self.embedder.out_dim + point_feats, width)

        self.cross_attn = ResidualCrossAttentionBlock(
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_checkpoint=False,
        )

        if self.input_sharp_pc:
            self.cross_attn1 = ResidualCrossAttentionBlock(
                width=width,
                heads=heads,
                init_scale=init_scale,
                qkv_bias=qkv_bias,
                use_flash=use_flash,
                use_checkpoint=False,
            )

        self.self_attn = Perceiver(
            n_ctx=num_latents,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_checkpoint=use_checkpoint,
        )

        if use_ln_post:
            self.ln_post = FP32LayerNorm(width)
        else:
            self.ln_post = None

    def _forward(self, coarse_pc, coarse_feats, sharp_pc, sharp_feats, split):
        bs, N_coarse, D_coarse = coarse_pc.shape

        coarse_data = self.embedder(coarse_pc)
        if coarse_feats is not None:
            if self.embed_point_feats:
                coarse_feats = self.embedder(coarse_feats)
            coarse_data = torch.cat([coarse_data, coarse_feats], dim=-1)

        coarse_data = self.input_proj(coarse_data)

        if self.input_sharp_pc:
            bs, N_sharp, D_sharp = sharp_pc.shape
            sharp_data = self.embedder(sharp_pc)
            if sharp_feats is not None:
                if self.embed_point_feats:
                    sharp_feats = self.embedder(sharp_feats)
                sharp_data = torch.cat([sharp_data, sharp_feats], dim=-1)
            sharp_data = self.input_proj1(sharp_data)

        if self.use_downsample:
            ###### fps
            tokens = np.array([128.0, 256.0, 384.0, 512.0, 640.0, 1024.0, 2048.0])

            coarse_ratios = tokens / N_coarse
            if split == "val":
                probabilities = np.array([0, 0, 0, 0, 0, 1, 0])
            elif split == "train":
                probabilities = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.2])
            ratio_coarse = np.random.choice(coarse_ratios, size=1, p=probabilities)[0]

            flattened = coarse_pc.view(bs * N_coarse, D_coarse)
            batch = torch.arange(bs).to(coarse_pc.device)
            batch = torch.repeat_interleave(batch, N_coarse)
            pos = flattened
            idx = fps(pos, batch, ratio=ratio_coarse)
            query_coarse = coarse_data.view(bs * N_coarse, -1)[idx].view(
                bs, -1, coarse_data.shape[-1]
            )

            if self.input_sharp_pc:
                index = np.where(coarse_ratios == ratio_coarse)[0]
                sharp_ratios = tokens / N_sharp
                ratio_sharp = sharp_ratios[index].item()

                flattened = sharp_pc.view(bs * N_sharp, D_sharp)
                batch = torch.arange(bs).to(sharp_pc.device)
                batch = torch.repeat_interleave(batch, N_sharp)
                pos = flattened
                idx = fps(pos, batch, ratio=ratio_sharp)
                query_sharp = sharp_data.view(bs * N_sharp, -1)[idx].view(
                    bs, -1, sharp_data.shape[-1]
                )

                query = torch.cat([query_coarse, query_sharp], dim=1)
            else:
                query = query_coarse
        else:
            query = self.query
            query = repeat(query, "m c -> b m c", b=bs)

        latents_coarse = self.cross_attn(query, coarse_data)
        if self.input_sharp_pc:
            latents_sharp = self.cross_attn1(query, sharp_data)
            latents = latents_coarse + latents_sharp
        else:
            latents = latents_coarse

        latents = self.self_attn(latents)
        if self.ln_post is not None:
            latents = self.ln_post(latents)

        return latents

    def forward(
        self,
        coarse_pc: torch.FloatTensor,
        coarse_feats: Optional[torch.FloatTensor] = None,
        sharp_pc: Optional[torch.FloatTensor] = None,
        sharp_feats: Optional[torch.FloatTensor] = None,
        split: str = "val",
    ):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]

        Returns:
            dict
        """

        return self._forward(coarse_pc, coarse_feats, sharp_pc, sharp_feats, split)
