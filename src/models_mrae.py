import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenMRAEManifoldProjector(nn.Module):
    """Frozen projector that maps NSD voxels to the pretrained 100-D MRAE manifold."""

    def __init__(self, ckpt_path: str):
        super().__init__()
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        if "encoder_state_dict" not in checkpoint or "decoder_state_dict" not in checkpoint:
            raise KeyError(
                f"Expected checkpoint keys 'encoder_state_dict' and 'decoder_state_dict', got {list(checkpoint.keys())}"
            )

        enc_sd = checkpoint["encoder_state_dict"]
        dec_sd = checkpoint["decoder_state_dict"]

        required_enc = [
            "linear.weight",
            "linear.bias",
            "linear2.weight",
            "linear2.bias",
            "linear3.weight",
            "linear3.bias",
        ]
        for key in required_enc:
            if key not in enc_sd:
                raise KeyError(f"Missing encoder weight '{key}' in checkpoint")

        if "linear_m.weight" not in dec_sd or "linear_m.bias" not in dec_sd:
            raise KeyError("Missing decoder manifold head ('linear_m.*') in checkpoint")

        in_dim = int(enc_sd["linear.weight"].shape[1])
        h1_dim = int(enc_sd["linear.weight"].shape[0])
        h2_dim = int(enc_sd["linear2.weight"].shape[0])
        h3_dim = int(enc_sd["linear3.weight"].shape[0])
        manifold_dim = int(dec_sd["linear_m.weight"].shape[0])

        self.input_dim = in_dim
        self.manifold_dim = manifold_dim

        self.linear = nn.Linear(in_dim, h1_dim)
        self.linear2 = nn.Linear(h1_dim, h2_dim)
        self.linear3 = nn.Linear(h2_dim, h3_dim)
        self.linear_m = nn.Linear(h3_dim, manifold_dim)

        self._load_weights(enc_sd, dec_sd)

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def _load_weights(self, enc_sd, dec_sd):
        with torch.no_grad():
            self.linear.weight.copy_(enc_sd["linear.weight"])
            self.linear.bias.copy_(enc_sd["linear.bias"])
            self.linear2.weight.copy_(enc_sd["linear2.weight"])
            self.linear2.bias.copy_(enc_sd["linear2.bias"])
            self.linear3.weight.copy_(enc_sd["linear3.weight"])
            self.linear3.bias.copy_(enc_sd["linear3.bias"])
            self.linear_m.weight.copy_(dec_sd["linear_m.weight"])
            self.linear_m.bias.copy_(dec_sd["linear_m.bias"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            if x.shape[1] != 1:
                raise ValueError(f"Expected x shape [B,1,N] for 3D input, got {tuple(x.shape)}")
            x = x[:, 0, :]
        elif x.ndim != 2:
            raise ValueError(f"Expected x shape [B,N] or [B,1,N], got {tuple(x.shape)}")

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"MRAE projector expects {self.input_dim} voxels, got {x.shape[-1]}"
            )

        with torch.no_grad():
            h = F.leaky_relu(self.linear(x), negative_slope=0.1)
            h = F.leaky_relu(self.linear2(h), negative_slope=0.1)
            h = F.leaky_relu(self.linear3(h), negative_slope=0.1)
            z = self.linear_m(h)
        return z


class MRAEClipMLPHead(nn.Module):
    """Learned three-hidden-layer MLP from manifold features to CLIP token space."""

    def __init__(
        self,
        in_dim: int = 100,
        mlp_dim1: int = 512,
        mlp_dim2: int = 512,
        mlp_dim3: int = 512,
        clip_seq_dim: int = 256,
        clip_emb_dim: int = 1664,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.clip_seq_dim = clip_seq_dim
        self.clip_emb_dim = clip_emb_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, mlp_dim1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim1, mlp_dim2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim2, mlp_dim3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim3, clip_seq_dim * clip_emb_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        return out.view(z.shape[0], self.clip_seq_dim, self.clip_emb_dim)


class MRAE2CLIPModel(nn.Module):
    """Frozen MRAE manifold projector + learned MLP head to CLIP token embeddings."""

    def __init__(
        self,
        mrae_ckpt_path: str,
        mlp_dim1: int,
        mlp_dim2: int,
        mlp_dim3: int,
        clip_seq_dim: int = 256,
        clip_emb_dim: int = 1664,
        mlp_dropout: float = 0.1,
    ):
        super().__init__()
        self.projector = FrozenMRAEManifoldProjector(mrae_ckpt_path)
        self.head = MRAEClipMLPHead(
            in_dim=self.projector.manifold_dim,
            mlp_dim1=mlp_dim1,
            mlp_dim2=mlp_dim2,
            mlp_dim3=mlp_dim3,
            clip_seq_dim=clip_seq_dim,
            clip_emb_dim=clip_emb_dim,
            dropout=mlp_dropout,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen projector in eval mode regardless of global train/eval state.
        self.projector.eval()
        return self

    def forward(self, x: torch.Tensor):
        manifold = self.projector(x)
        clip_tokens = self.head(manifold)
        backbone = clip_tokens
        return backbone, clip_tokens
