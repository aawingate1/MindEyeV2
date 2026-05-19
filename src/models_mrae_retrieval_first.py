import argparse

import torch
import torch.nn as nn

from models_mrae_variant_common import BaseMRAEVariantModel, ResidualMLPBlock


class MRAERetrievalFirstModel(BaseMRAEVariantModel):
    def __init__(self, args):
        super().__init__(
            mrae_ckpt_path=args.mrae_ckpt_path,
            clip_seq_dim=args.clip_seq_dim,
            clip_emb_dim=args.clip_emb_dim,
            projector_tune_mode=args.projector_tune_mode,
            projector_lr_scale=args.projector_lr_scale,
            token_aux_scale=args.token_aux_scale,
            global_scale=args.global_scale,
        )
        hidden_dim = args.model_hidden_dim
        self.global_head = nn.Sequential(
            nn.LayerNorm(self.projector.manifold_dim),
            nn.Linear(self.projector.manifold_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, args.clip_emb_dim),
        )
        self.token_seed = nn.Parameter(torch.randn(1, args.clip_seq_dim, hidden_dim) * 0.02)
        self.condition_proj = nn.Linear(self.projector.manifold_dim + args.clip_emb_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, hidden_dim * 4, args.dropout) for _ in range(args.num_layers)]
        )
        self.token_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, args.clip_emb_dim),
        )
        self.aux_token_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, args.clip_emb_dim),
        )

    def forward_from_manifold(self, manifold: torch.Tensor):
        global_pred = self.global_head(manifold)
        cond = torch.cat([manifold, global_pred], dim=-1)
        cond = self.condition_proj(cond).unsqueeze(1)
        tokens = self.token_seed.expand(manifold.shape[0], -1, -1) + cond
        for block in self.blocks:
            tokens = block(tokens)
        clip_tokens = self.token_out(tokens)
        aux_clip_tokens = self.aux_token_out(tokens)
        return {
            "clip_tokens": clip_tokens,
            "aux_clip_tokens": aux_clip_tokens,
            "global_pred": global_pred,
        }


def add_model_specific_args(parser: argparse.ArgumentParser):
    parser.add_argument("--model_hidden_dim", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--token_aux_scale", type=float, default=0.25)
    parser.add_argument("--global_scale", type=float, default=0.5)
    parser.set_defaults(projector_tune_mode="full", projector_lr_scale=0.05)


def build_model(args):
    return MRAERetrievalFirstModel(args)
