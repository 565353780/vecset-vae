import math
import torch
import torch.nn as nn
from typing import Optional

from vecset_vae.Model.diagonal_gaussian_distribution import (
    DiagonalGaussianDistribution,
)
from vecset_vae.Model.Layer.fourier_embedder import FourierEmbedder
from vecset_vae.Model.Transformer.perceiver_1d import Perceiver
from vecset_vae.Model.perceiver_cross_attention_encoder import (
    PerceiverCrossAttentionEncoder,
)
from vecset_vae.Model.perceiver_cross_attention_decoder import (
    PerceiverCrossAttentionDecoder,
)


class VecSetVAE(nn.Module):
    def __init__(
        self,
        use_downsample: bool = True,
        num_latents: int = 64,
        embed_dim: int = 64,
        width: int = 768,
        point_feats: int = 3,
        input_sharp_pc: bool = True,
        embed_point_feats: bool = False,
        query_dim: int = 3,
        out_dim: int = 1,
        num_freqs: int = 8,
        include_pi: bool = False,
        heads: int = 12,
        num_encoder_layers: int = 8,
        num_decoder_layers: int = 16,
        init_scale: float = 0.25,
        qkv_bias: bool = False,
        use_ln_post: bool = True,
        use_flash: bool = True,
        use_checkpoint: bool = True,
        split: str = "val",
    ) -> None:
        super().__init__()

        self.use_downsample = use_downsample
        self.num_latents = num_latents
        self.embed_dim = embed_dim
        self.width = width
        self.point_feats = point_feats
        self.input_sharp_pc = input_sharp_pc
        self.embed_point_feats = embed_point_feats
        self.query_dim = query_dim
        self.out_dim = out_dim
        self.num_freqs = num_freqs
        self.include_pi = include_pi
        self.heads = heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.init_scale = init_scale
        self.qkv_bias = qkv_bias
        self.use_ln_post = use_ln_post
        self.use_flash = use_flash
        self.use_checkpoint = use_checkpoint
        self.split = split

        self.embedder = FourierEmbedder(
            num_freqs=self.num_freqs,
            include_pi=self.include_pi,
        )

        self.init_scale = self.init_scale * math.sqrt(1.0 / self.width)
        self.encoder = PerceiverCrossAttentionEncoder(
            use_downsample=self.use_downsample,
            embedder=self.embedder,
            num_latents=self.num_latents,
            point_feats=self.point_feats,
            input_sharp_pc=self.input_sharp_pc,
            embed_point_feats=self.embed_point_feats,
            width=self.width,
            heads=self.heads,
            layers=self.num_encoder_layers,
            init_scale=self.init_scale,
            qkv_bias=self.qkv_bias,
            use_ln_post=self.use_ln_post,
            use_flash=self.use_flash,
            use_checkpoint=self.use_checkpoint,
        )

        self.pre_kl = nn.Linear(self.width, self.embed_dim * 2)
        self.post_kl = nn.Linear(self.embed_dim, self.width)

        self.transformer = Perceiver(
            n_ctx=self.num_latents,
            width=self.width,
            layers=self.num_decoder_layers,
            heads=self.heads,
            init_scale=self.init_scale,
            qkv_bias=self.qkv_bias,
            use_flash=self.use_flash,
            use_checkpoint=self.use_checkpoint,
        )

        # decoder
        if self.query_dim == 3:
            self.decoder = PerceiverCrossAttentionDecoder(
                embedder=self.embedder,
                out_dim=self.out_dim,
                num_latents=self.num_latents,
                width=self.width,
                heads=self.heads,
                init_scale=self.init_scale,
                qkv_bias=self.qkv_bias,
                use_flash=self.use_flash,
                use_checkpoint=self.use_checkpoint,
            )
        else:
            self.decode_embedder = FourierEmbedder(
                num_freqs=self.num_freqs,
                input_dim=self.query_dim,
                include_pi=self.include_pi,
            )
            self.decoder = PerceiverCrossAttentionDecoder(
                embedder=self.decode_embedder,
                out_dim=self.out_dim,
                num_latents=self.num_latents,
                width=self.width,
                heads=self.heads,
                init_scale=self.init_scale,
                qkv_bias=self.qkv_bias,
                use_flash=self.use_flash,
                use_checkpoint=self.use_checkpoint,
            )
        return

    def encode(
        self,
        coarse_surface: torch.FloatTensor,
        sharp_surface: Optional[torch.FloatTensor] = None,
    ):
        """
        Args:
            surface (torch.FloatTensor): [B, N, 3+C]
        Returns:
            shape_latents (torch.FloatTensor): [B, num_latents, width]
            kl_embed (torch.FloatTensor): [B, num_latents, embed_dim]
            posterior (DiagonalGaussianDistribution or None):
        """

        coarse_pc, coarse_feats = coarse_surface[..., :3], coarse_surface[..., 3:]
        if sharp_surface is not None:
            sharp_pc, sharp_feats = sharp_surface[..., :3], sharp_surface[..., 3:]
        else:
            sharp_pc = None
            sharp_feats = None
        shape_latents = self.encoder(
            coarse_pc, coarse_feats, sharp_pc, sharp_feats, split=self.split
        )
        return shape_latents

    def encode_kl_embed(
        self, latents: torch.FloatTensor, sample_posterior: bool = True
    ):
        posterior = None
        moments = self.pre_kl(latents)  # 103，256，768 -》 103，256，128
        posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
        if sample_posterior:
            kl_embed = posterior.sample()  # 1，768，64
        else:
            kl_embed = posterior.mode()

        kl = posterior.kl()
        return kl_embed, kl

    def decode(self, latents: torch.FloatTensor) -> torch.Tensor:
        """
        Args:
            latents (torch.FloatTensor): [B, embed_dim]

        Returns:
            latents (torch.FloatTensor): [B, embed_dim]
        """
        latents = self.post_kl(
            latents
        )  # [B, num_latents, embed_dim] -> [B, num_latents, width]

        latents = self.transformer(latents)
        return latents

    def query(self, queries: torch.FloatTensor, latents: torch.Tensor):
        """
        Args:
            queries (torch.FloatTensor): [B, N, 3]
            latents (torch.FloatTensor): [B, embed_dim]

        Returns:
            logits (torch.FloatTensor): [B, N], tsdf logits
        """
        logits = self.decoder(queries, latents).squeeze(-1)
        return logits

    def forward(
        self,
        data_dict: dict,
    ) -> dict:
        coarse_surface = data_dict["coarse_surface"]
        sharp_surface = data_dict["sharp_surface"]
        queries = data_dict["rand_points"]
        self.split = data_dict["split"]
        sample_posterior = data_dict["sample_posterior"]

        if "coarse_tsdf" in data_dict.keys():
            coarse_tsdf = data_dict["coarse_tsdf"]
            queries = torch.cat([queries, coarse_tsdf.unsqueeze(-1)], dim=-1)

        shape_latents = self.encode(coarse_surface, sharp_surface)

        kl_embed, kl = self.encode_kl_embed(
            shape_latents, sample_posterior=sample_posterior
        )

        latents = self.decode(kl_embed)

        tsdf = self.query(queries, latents)

        result_dict = {
            "tsdf": tsdf,
            "kl": kl,
        }

        return result_dict


def build_dora_vae(num_latents: int = 64) -> nn.Module:
    return VecSetVAE(
        use_downsample=True,
        num_latents=num_latents,
        embed_dim=64,
        width=768,
        point_feats=3,
        input_sharp_pc=True,
        embed_point_feats=False,
        query_dim=3,
        out_dim=1,
        num_freqs=8,
        include_pi=False,
        heads=12,
        num_encoder_layers=8,
        num_decoder_layers=16,
        init_scale=0.25,
        qkv_bias=False,
        use_ln_post=True,
        use_flash=True,
        use_checkpoint=True,
        split="val",
    )


def build_refine_vae(
    num_latents: int = 64,
    embed_dim: int = 64,
) -> nn.Module:
    return VecSetVAE(
        use_downsample=True,
        num_latents=num_latents,
        embed_dim=embed_dim,
        width=768,
        point_feats=3,
        input_sharp_pc=True,
        embed_point_feats=False,
        query_dim=4,
        out_dim=1,
        num_freqs=8,
        include_pi=False,
        heads=12,
        num_encoder_layers=8,
        num_decoder_layers=16,
        init_scale=0.25,
        qkv_bias=False,
        use_ln_post=True,
        use_flash=True,
        use_checkpoint=True,
        split="val",
    )
