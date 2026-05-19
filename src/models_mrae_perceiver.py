import argparse

import torch
import torch.nn as nn

from models_mrae_variant_common import BaseMRAEVariantModel, CrossAttentionBlock, SelfAttentionBlock


class MRAEPerceiverModel(BaseMRAEVariantModel):
    def __init__(self, args):
        super().__init__(
            mrae_ckpt_path=args.mrae_ckpt_path,
            clip_seq_dim=args.clip_seq_dim,
            clip_emb_dim=args.clip_emb_dim,
            projector_tune_mode=args.projector_tune_mode,
            projector_lr_scale=args.projector_lr_scale,
        )
        hidden_dim = args.model_hidden_dim
        self.context_proj = nn.Linear(self.projector.manifold_dim, args.num_context_tokens * hidden_dim)
        self.latents = nn.Parameter(torch.randn(1, args.num_latents, hidden_dim) * 0.02)
        self.latent_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        CrossAttentionBlock(hidden_dim, hidden_dim, args.num_heads, 4.0, args.dropout),
                        SelfAttentionBlock(hidden_dim, args.num_heads, 4.0, args.dropout),
                    ]
                )
                for _ in range(args.num_layers)
            ]
        )
        self.query_tokens = nn.Parameter(torch.randn(1, args.clip_seq_dim, hidden_dim) * 0.02)
        self.decoder = CrossAttentionBlock(hidden_dim, hidden_dim, args.num_heads, 4.0, args.dropout)
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, args.clip_emb_dim))

    def forward_from_manifold(self, manifold: torch.Tensor):
        bsz = manifold.shape[0]
        context = self.context_proj(manifold).view(bsz, -1, self.latents.shape[-1])
        latents = self.latents.expand(bsz, -1, -1)
        for cross_block, self_block in self.latent_blocks:
            latents = cross_block(latents, context)
            latents = self_block(latents)
        queries = self.query_tokens.expand(bsz, -1, -1)
        queries = self.decoder(queries, latents)
        clip_tokens = self.out(queries)
        return {"clip_tokens": clip_tokens}


def add_model_specific_args(parser: argparse.ArgumentParser):
    parser.add_argument("--model_hidden_dim", type=int, default=512)
    parser.add_argument("--num_context_tokens", type=int, default=16)
    parser.add_argument("--num_latents", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.set_defaults(projector_tune_mode="full", projector_lr_scale=0.03)


def build_model(args):
    return MRAEPerceiverModel(args)
