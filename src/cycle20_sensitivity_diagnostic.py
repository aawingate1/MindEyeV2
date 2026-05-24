import csv
import os

import h5py
import torch


SRC_DIR = os.environ.get("SRC_DIR", "/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src")
RUN_ROOT = os.path.dirname(SRC_DIR)

ROWS = [
    ("subj05", "relgate0", "cycle16_subj05_relgate0_1sess_150ep"),
    ("subj05", "relgate_low", "cycle16_subj05_relgate_low_1sess_150ep"),
    ("subj05", "relgate_mid", "cycle16_subj05_relgate_mid_1sess_150ep"),
    ("subj07", "relgate0", "cycle16_subj07_relgate0_1sess_150ep"),
    ("subj07", "relgate_low", "cycle16_subj07_relgate_low_1sess_150ep"),
    ("subj07", "relgate_mid", "cycle16_subj07_relgate_mid_1sess_150ep"),
]


def average_rank(x):
    order = torch.argsort(x)
    ranks = torch.empty_like(x, dtype=torch.float64)
    ranks[order] = torch.arange(len(x), dtype=torch.float64)
    return ranks


def spearman(x, y):
    x = x.detach().flatten().to(torch.float64)
    y = y.detach().flatten().to(torch.float64)
    valid = torch.isfinite(x) & torch.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3:
        return float("nan")
    rx = average_rank(x)
    ry = average_rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = torch.linalg.vector_norm(rx) * torch.linalg.vector_norm(ry)
    if denom.item() == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def get_state_dict(path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def main():
    out_path = os.path.join(SRC_DIR, "tables", "cycle20_relgate_sensitivity.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mask_path = os.path.join(SRC_DIR, "brain_region_masks.hdf5")

    rows = []
    with h5py.File(mask_path, "r") as masks:
        for subj, tag, model_name in ROWS:
            subj_num = subj[-2:]
            rel_path = os.path.join(SRC_DIR, "reliability", f"{subj}_trainrepeat_reliability.pt")
            ckpt_path = os.path.join(RUN_ROOT, "train_logs", model_name, "last.pth")
            rel_obj = torch.load(rel_path, map_location="cpu")
            reliability = rel_obj["reliability"] if isinstance(rel_obj, dict) else rel_obj
            state = get_state_dict(ckpt_path)
            weight = state["ridge.linears.0.weight"].detach().float()
            sensitivity = weight.abs().mean(dim=0)
            if sensitivity.numel() != reliability.numel():
                raise ValueError(
                    f"{model_name}: sensitivity length {sensitivity.numel()} "
                    f"does not match reliability length {reliability.numel()}"
                )

            subj_masks = masks[f"subj{subj_num}"]
            for region in ("all", "early_vis", "higher_vis"):
                if region == "all":
                    mask = torch.ones_like(reliability, dtype=torch.bool)
                else:
                    mask = torch.from_numpy(subj_masks[region][:]).bool()
                rows.append(
                    {
                        "subject": subj,
                        "tag": tag,
                        "model_name": model_name,
                        "region": region,
                        "n_voxels": int(mask.sum()),
                        "spearman_reliability_sensitivity": spearman(
                            reliability[mask], sensitivity[mask]
                        ),
                        "mean_sensitivity": float(sensitivity[mask].mean()),
                        "mean_reliability": float(reliability[mask].mean()),
                        "checkpoint_path": ckpt_path,
                        "reliability_path": rel_path,
                    }
                )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out_path)


if __name__ == "__main__":
    main()
