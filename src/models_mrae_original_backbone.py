import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from models_mrae_variant_common import BaseMRAEVariantModel


class MRAEOriginalBackboneModel(BaseMRAEVariantModel):
    def __init__(self, args):
        super().__init__(
            mrae_ckpt_path=args.mrae_ckpt_path,
            clip_seq_dim=args.clip_seq_dim,
            clip_emb_dim=args.clip_emb_dim,
            projector_tune_mode=args.projector_tune_mode,
            projector_lr_scale=args.projector_lr_scale,
        )
        self.hidden_dim = args.hidden_dim
        self.seq_len = 1
        self.n_blocks = args.n_blocks
        self.drop = args.dropout
        self.use_ridge_after_mrae = args.use_ridge_after_mrae
        self.use_mrae_encoder_only = args.use_mrae_encoder_only
        if self.use_mrae_encoder_only and args.projector_tune_mode == "last":
            raise ValueError(
                "projector_tune_mode='last' is incompatible with --use_mrae_encoder_only "
                "because the linear_m head is skipped. Use 'freeze' or 'full' instead."
            )
        self.mrae_latent_dim = (
            self.projector.linear3.out_features
            if self.use_mrae_encoder_only
            else self.projector.manifold_dim
        )
        if not self.use_ridge_after_mrae and self.mrae_latent_dim != self.hidden_dim:
            raise ValueError(
                "Direct MRAE-to-backbone mode requires the MRAE output width to match "
                f"hidden_dim. Got mrae_latent_dim={self.mrae_latent_dim} and hidden_dim={self.hidden_dim}. "
                "Enable --use_ridge_after_mrae or set --hidden_dim to the MRAE output width."
            )

        # Mirror the original MindEye path as closely as possible:
        # When ridge is enabled, treat the MRAE output like the original voxel
        # input to MindEye: ridge -> residual mixer backbone -> CLIP projector.
        self.ridge = nn.Linear(self.mrae_latent_dim, self.hidden_dim)
        self.mixer_blocks1 = nn.ModuleList(
            [self._mixer_block1(self.hidden_dim, self.drop) for _ in range(self.n_blocks)]
        )
        self.mixer_blocks2 = nn.ModuleList(
            [self._mixer_block2(self.seq_len, self.drop) for _ in range(self.n_blocks)]
        )
        self.backbone_linear = nn.Linear(self.hidden_dim * self.seq_len, args.clip_emb_dim * args.clip_seq_dim, bias=True)
        self.clip_proj = self._projector(args.clip_emb_dim, args.clip_emb_dim, h=args.clip_emb_dim)

    def _projector(self, in_dim, out_dim, h=2048):
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, out_dim),
        )

    def _mlp(self, in_dim, out_dim, drop):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(out_dim, out_dim),
        )

    def _mixer_block1(self, h, drop):
        return nn.Sequential(
            nn.LayerNorm(h),
            self._mlp(h, h, drop),
        )

    def _mixer_block2(self, seq_len, drop):
        return nn.Sequential(
            nn.LayerNorm(seq_len),
            self._mlp(seq_len, seq_len, drop),
        )

    def encode_manifold(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_mrae_encoder_only:
            return super().encode_manifold(x)

        if x.ndim == 3:
            x = x[:, 0, :]
        x = F.leaky_relu(self.projector.linear(x), negative_slope=0.1)
        x = F.leaky_relu(self.projector.linear2(x), negative_slope=0.1)
        x = F.leaky_relu(self.projector.linear3(x), negative_slope=0.1)
        return x

    def forward_from_manifold(self, manifold: torch.Tensor):
        if self.use_ridge_after_mrae:
            x = self.ridge(manifold)
        else:
            x = manifold
        x = x.unsqueeze(1)

        residual1 = x
        residual2 = x.permute(0, 2, 1)
        for block1, block2 in zip(self.mixer_blocks1, self.mixer_blocks2):
            x = block1(x) + residual1
            residual1 = x
            x = x.permute(0, 2, 1)

            x = block2(x) + residual2
            residual2 = x
            x = x.permute(0, 2, 1)

        x = x.reshape(x.size(0), -1)
        backbone = self.backbone_linear(x).reshape(len(x), -1, self.clip_emb_dim)
        clip_tokens = self.clip_proj(backbone)
        return {
            "backbone_tokens": backbone,
            "clip_tokens": clip_tokens,
        }


def add_model_specific_args(parser: argparse.ArgumentParser):
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--use_ridge_after_mrae", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_mrae_encoder_only", action=argparse.BooleanOptionalAction, default=False)
    parser.set_defaults(projector_tune_mode="full", projector_lr_scale=0.03)


def build_model(args):
    return MRAEOriginalBackboneModel(args)
