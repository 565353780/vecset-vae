import torch
import torch.nn as nn

from vecset_vae.Model.Layer.fp32_layer_norm import FP32LayerNorm
from vecset_vae.Model.checkpoint import checkpoint
from vecset_vae.Model.Layer.fourier_embedder import FourierEmbedder
from vecset_vae.Model.Transformer.attention import ResidualCrossAttentionBlock


class PerceiverCrossAttentionDecoder(nn.Module):
    def __init__(
        self,
        num_latents: int,
        out_dim: int,
        embedder: FourierEmbedder,
        width: int,
        heads: int,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
        use_flash: bool = False,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.embedder = embedder

        self.query_proj = nn.Linear(self.embedder.out_dim, width)

        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            n_data=num_latents,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_checkpoint=use_checkpoint,
        )

        self.ln_post = FP32LayerNorm(width)
        self.output_proj = nn.Linear(width, out_dim)

    def _forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        queries = self.query_proj(self.embedder(queries))
        x = self.cross_attn_decoder(queries, latents)
        x = self.ln_post(x)
        x = self.output_proj(x)
        return x

    def forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        logits = checkpoint(
            self._forward, (queries, latents), self.parameters(), self.use_checkpoint
        )
        return logits
