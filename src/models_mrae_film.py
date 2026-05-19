import argparse

import torch
import torch.nn as nn

from models_mrae_variant_common import BaseMRAEVariantModel, ResidualMLPBlock


class MRAEFiLMModel(BaseMRAEVariantModel):
    def __init__(self, args):
        super().__init__(
            mrae_ckpt_path=args.mrae_ckpt_path,
            clip_seq_dim=args.clip_seq_dim,
            clip_emb_dim=args.clip_emb_dim,
            projector_tune_mode=args.projector_tune_mode,
            projector_lr_scale=args.projector_lr_scale,
        )
        hidden_dim = args.model_hidden_dim
        self.token_bank = nn.Parameter(torch.randn(1, args.clip_seq_dim, hidden_dim) * 0.02)
        self.latent_proj = nn.Sequential(
            nn.LayerNorm(self.projector.manifold_dim),
            nn.Linear(self.projector.manifold_dim, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, hidden_dim * 4, args.dropout) for _ in range(args.num_layers)]
        )
        self.film = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim * 2) for _ in range(args.num_layers)]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, args.clip_emb_dim),
        )

    def forward_from_manifold(self, manifold: torch.Tensor):
        latent = self.latent_proj(manifold)
        tokens = self.token_bank.expand(manifold.shape[0], -1, -1)
        for block, film_layer in zip(self.blocks, self.film):
            gamma, beta = film_layer(latent).chunk(2, dim=-1)
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
            tokens = tokens * (1.0 + 0.1 * gamma) + 0.1 * beta
            tokens = block(tokens)
        clip_tokens = self.out(tokens)
        return {"clip_tokens": clip_tokens}


def add_model_specific_args(parser: argparse.ArgumentParser):
    parser.add_argument("--model_hidden_dim", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.set_defaults(projector_tune_mode="last", projector_lr_scale=0.05)


def build_model(args):
    return MRAEFiLMModel(args)
