import argparse

import torch
import torch.nn as nn

from models_mrae_variant_common import BaseMRAEVariantModel, ResidualMLPBlock


class MRAEResMLPModel(BaseMRAEVariantModel):
    def __init__(self, args):
        super().__init__(
            mrae_ckpt_path=args.mrae_ckpt_path,
            clip_seq_dim=args.clip_seq_dim,
            clip_emb_dim=args.clip_emb_dim,
            projector_tune_mode=args.projector_tune_mode,
            projector_lr_scale=args.projector_lr_scale,
        )
        hidden_dim = args.model_hidden_dim
        self.stem = nn.Sequential(
            nn.LayerNorm(self.projector.manifold_dim),
            nn.Linear(self.projector.manifold_dim, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, hidden_dim * 4, args.dropout) for _ in range(args.num_res_blocks)]
        )
        self.to_tokens = nn.Linear(hidden_dim, args.clip_seq_dim * args.clip_emb_dim)
        self.token_proj = nn.Sequential(
            nn.LayerNorm(args.clip_emb_dim),
            nn.Linear(args.clip_emb_dim, args.clip_emb_dim),
            nn.GELU(),
            nn.Linear(args.clip_emb_dim, args.clip_emb_dim),
        )

    def forward_from_manifold(self, manifold: torch.Tensor):
        x = self.stem(manifold)
        for block in self.blocks:
            x = block(x)
        clip_tokens = self.to_tokens(x).view(-1, self.clip_seq_dim, self.clip_emb_dim)
        clip_tokens = clip_tokens + self.token_proj(clip_tokens)
        return {"clip_tokens": clip_tokens}


def add_model_specific_args(parser: argparse.ArgumentParser):
    parser.add_argument("--model_hidden_dim", type=int, default=768)
    parser.add_argument("--num_res_blocks", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.set_defaults(projector_tune_mode="last", projector_lr_scale=0.05)


def build_model(args):
    return MRAEResMLPModel(args)
