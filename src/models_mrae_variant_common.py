import torch
import torch.nn as nn
import torch.nn.functional as F


def _layernorm_linear_gelu(in_dim, out_dim):
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, out_dim),
        nn.GELU(),
    )


class FineTunableMRAEProjector(nn.Module):
    def __init__(self, ckpt_path: str, tune_mode: str = "freeze"):
        super().__init__()
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        enc_sd = checkpoint["encoder_state_dict"]
        dec_sd = checkpoint["decoder_state_dict"]

        in_dim = int(enc_sd["linear.weight"].shape[1])
        h1_dim = int(enc_sd["linear.weight"].shape[0])
        h2_dim = int(enc_sd["linear2.weight"].shape[0])
        h3_dim = int(enc_sd["linear3.weight"].shape[0])
        manifold_dim = int(dec_sd["linear_m.weight"].shape[0])

        self.input_dim = in_dim
        self.manifold_dim = manifold_dim
        self.tune_mode = tune_mode

        self.linear = nn.Linear(in_dim, h1_dim)
        self.linear2 = nn.Linear(h1_dim, h2_dim)
        self.linear3 = nn.Linear(h2_dim, h3_dim)
        self.linear_m = nn.Linear(h3_dim, manifold_dim)

        with torch.no_grad():
            self.linear.weight.copy_(enc_sd["linear.weight"])
            self.linear.bias.copy_(enc_sd["linear.bias"])
            self.linear2.weight.copy_(enc_sd["linear2.weight"])
            self.linear2.bias.copy_(enc_sd["linear2.bias"])
            self.linear3.weight.copy_(enc_sd["linear3.weight"])
            self.linear3.bias.copy_(enc_sd["linear3.bias"])
            self.linear_m.weight.copy_(dec_sd["linear_m.weight"])
            self.linear_m.bias.copy_(dec_sd["linear_m.bias"])

        self.set_tune_mode(tune_mode)

    def set_tune_mode(self, tune_mode: str):
        self.tune_mode = tune_mode
        for p in self.parameters():
            p.requires_grad = False
        if tune_mode == "full":
            for p in self.parameters():
                p.requires_grad = True
        elif tune_mode == "last":
            for p in self.linear_m.parameters():
                p.requires_grad = True
        elif tune_mode != "freeze":
            raise ValueError(f"Unknown projector tune_mode '{tune_mode}'")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.tune_mode == "freeze":
            super().train(False)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x[:, 0, :]
        h = F.leaky_relu(self.linear(x), negative_slope=0.1)
        h = F.leaky_relu(self.linear2(h), negative_slope=0.1)
        h = F.leaky_relu(self.linear3(h), negative_slope=0.1)
        return self.linear_m(h)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = F.gelu(self.fc1(y))
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.dropout(y)
        return x + y


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(query_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.context_proj = nn.Linear(context_dim, query_dim) if context_dim != query_dim else nn.Identity()
        self.attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff_norm = nn.LayerNorm(query_dim)
        mlp_hidden = int(query_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(query_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, query_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.query_norm(query)
        kv = self.context_proj(self.context_norm(context))
        y, _ = self.attn(q, kv, kv, need_weights=False)
        query = query + y
        query = query + self.mlp(self.ff_norm(query))
        return query


class BaseMRAEVariantModel(nn.Module):
    def __init__(
        self,
        mrae_ckpt_path: str,
        clip_seq_dim: int,
        clip_emb_dim: int,
        projector_tune_mode: str = "freeze",
        projector_lr_scale: float = 0.1,
        token_aux_scale: float = 0.0,
        global_scale: float = 0.0,
    ):
        super().__init__()
        self.clip_seq_dim = clip_seq_dim
        self.clip_emb_dim = clip_emb_dim
        self.projector = FineTunableMRAEProjector(mrae_ckpt_path, tune_mode=projector_tune_mode)
        self.projector_lr_scale = projector_lr_scale
        self.token_aux_scale = token_aux_scale
        self.global_scale = global_scale

    def train(self, mode: bool = True):
        super().train(mode)
        self.projector.train(mode)
        return self

    def param_groups(self, base_lr: float):
        projector_params = []
        main_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("projector."):
                projector_params.append(param)
            else:
                main_params.append(param)

        groups = []
        if main_params:
            groups.append({"params": main_params, "lr": base_lr})
        if projector_params:
            groups.append({"params": projector_params, "lr": base_lr * self.projector_lr_scale})
        return groups

    def encode_manifold(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)

    def _normalize_global(self, clip_tokens: torch.Tensor) -> torch.Tensor:
        return clip_tokens.mean(dim=1)

    def forward_from_manifold(self, manifold: torch.Tensor):
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        manifold = self.encode_manifold(x)
        outputs = self.forward_from_manifold(manifold)
        outputs["manifold"] = manifold
        if "clip_tokens" not in outputs:
            raise KeyError("Variant model must return 'clip_tokens'")
        return outputs
