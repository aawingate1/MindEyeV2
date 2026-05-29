#!/usr/bin/env python
import argparse
import csv
import json
import os
import random
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
import webdataset as wds

sys.path.append("generative_models/")
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder

import utils
from models import BrainNetwork


def my_split_by_node(urls):
    return urls


class MindEyeModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class RidgeRegression(nn.Module):
    def __init__(self, input_sizes, out_features):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(input_size, out_features) for input_size in input_sizes])

    def forward(self, x, subj_idx):
        return self.linears[subj_idx](x[:, 0]).unsqueeze(1)


def rankdata_ordinal(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def pearson_np(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 3:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def spearman_np(x, y):
    return pearson_np(rankdata_ordinal(x), rankdata_ordinal(y))


def geometry_metrics(pred_emb, image_emb, ks=(1, 5, 10)):
    pred = torch.nn.functional.normalize(pred_emb.float(), dim=-1)
    image = torch.nn.functional.normalize(image_emb.float(), dim=-1)
    pred_sim = (pred @ pred.T).cpu().numpy()
    image_sim = (image @ image.T).cpu().numpy()
    n = pred_sim.shape[0]
    mask = ~np.eye(n, dtype=bool)
    pred_vec = pred_sim[mask]
    image_vec = image_sim[mask]
    pred_dist = 1.0 - pred_vec
    image_dist = 1.0 - image_vec

    results = {
        "n": int(n),
        "spearman_rsa_cosine_distance": spearman_np(pred_dist, image_dist),
        "pearson_offdiag_cosine_similarity": pearson_np(pred_vec, image_vec),
    }

    pred_order = np.argsort(-pred_sim, axis=1)
    image_order = np.argsort(-image_sim, axis=1)
    pred_order = np.asarray([row[row != i] for i, row in enumerate(pred_order)])
    image_order = np.asarray([row[row != i] for i, row in enumerate(image_order)])

    for k in ks:
        kk = min(k, n - 1)
        overlaps = []
        for i in range(n):
            overlaps.append(len(set(pred_order[i, :kk]).intersection(image_order[i, :kk])) / float(kk))
        results[f"nn_overlap_at_{k}"] = float(np.mean(overlaps))

    ranks = []
    reciprocal_ranks = []
    for i in range(n):
        teacher_nn = int(image_order[i, 0])
        rank = int(np.where(pred_order[i] == teacher_nn)[0][0]) + 1
        ranks.append(rank)
        reciprocal_ranks.append(1.0 / rank)
    results["teacher_top1_median_rank_under_pred"] = float(np.median(ranks))
    results["teacher_top1_mrr_under_pred"] = float(np.mean(reciprocal_ranks))
    return results


def build_model(num_voxels, hidden_dim, n_blocks, blurry_recon, clip_scale, device):
    clip_seq_dim = 256
    clip_emb_dim = 1664
    model = MindEyeModule()
    model.ridge = RidgeRegression([num_voxels], out_features=hidden_dim)
    model.backbone = BrainNetwork(
        h=hidden_dim,
        in_dim=hidden_dim,
        seq_len=1,
        n_blocks=n_blocks,
        clip_size=clip_emb_dim,
        out_dim=clip_emb_dim * clip_seq_dim,
        blurry_recon=blurry_recon,
        clip_scale=clip_scale,
    )
    model.to(device)
    return model


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    incompatible = model.load_state_dict(state_dict, strict=False)
    return {
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def collect_embeddings(args, row, device):
    data_type = torch.float16
    train_url = f"{args.data_path}/wds/subj0{row['subj']}/train/" + "{0.." + f"{args.num_sessions - 1}" + "}.tar"
    dataset = (
        wds.WebDataset(train_url, resampled=False, nodesplitter=my_split_by_node)
        .decode("torch")
        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")
        .to_tuple("behav", "past_behav", "future_behav", "olds_behav")
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True)

    betas_h5 = h5py.File(f"{args.data_path}/betas_all_subj0{row['subj']}_fp32_renorm.hdf5", "r")
    betas = torch.tensor(betas_h5["betas"][:], dtype=data_type, device="cpu")
    num_voxels = int(betas.shape[-1])
    images_h5 = h5py.File(f"{args.data_path}/coco_images_224_float16.hdf5", "r")
    images = images_h5["images"]

    model = build_model(
        num_voxels=num_voxels,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        blurry_recon=args.blurry_recon,
        clip_scale=1.0,
        device=device,
    )
    ckpt_info = load_checkpoint(model, row["checkpoint_path"])
    model.eval().requires_grad_(False)

    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
    )
    clip_img_embedder.eval().requires_grad_(False).to(device)

    seen_images = set()
    image_indices = []
    voxel_indices = []
    pred_chunks = []
    image_chunks = []
    duplicate_image_samples_skipped = 0

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=data_type):
        for behav0, _past_behav0, _future_behav0, _old_behav0 in loader:
            image_idx = behav0[:, 0, 0].cpu().long().numpy()
            voxel_idx = behav0[:, 0, 5].cpu().long().numpy()
            keep = []
            for i, img_id in enumerate(image_idx):
                img_id = int(img_id)
                if img_id in seen_images:
                    duplicate_image_samples_skipped += 1
                    continue
                seen_images.add(img_id)
                keep.append(i)
                image_indices.append(img_id)
                voxel_indices.append(int(voxel_idx[i]))
                if args.max_images is not None and len(image_indices) >= args.max_images:
                    break
            if not keep:
                continue
            keep = np.asarray(keep, dtype=np.int64)
            batch_image_idx = image_idx[keep]
            batch_voxel_idx = voxel_idx[keep]
            voxel = betas[batch_voxel_idx].unsqueeze(1).to(device)
            image_order = np.argsort(batch_image_idx)
            image_restore = np.argsort(image_order)
            image_np = images[batch_image_idx[image_order]][image_restore]
            image = torch.tensor(image_np, dtype=data_type, device=device)

            voxel_ridge = model.ridge(voxel, 0)
            _backbone, clip_voxels, _blur = model.backbone(voxel_ridge)
            image_clip = clip_img_embedder(image)
            pred_chunks.append(clip_voxels.detach().float().cpu())
            image_chunks.append(image_clip.detach().float().cpu())

            if args.max_images is not None and len(image_indices) >= args.max_images:
                break

    pred_tokens = torch.cat(pred_chunks, dim=0)
    image_tokens = torch.cat(image_chunks, dim=0)
    return {
        "model_name": row["model_name"],
        "subject": int(row["subj"]),
        "tag": row["tag"],
        "checkpoint_path": row["checkpoint_path"],
        "checkpoint_info": ckpt_info,
        "train_url": train_url,
        "training_only": True,
        "shared1000_or_new_test_used": False,
        "test_sources_used": [],
        "image_count": int(pred_tokens.shape[0]),
        "unique_image_count": int(len(set(image_indices))),
        "duplicate_image_samples_skipped": int(duplicate_image_samples_skipped),
        "image_indices_min": int(min(image_indices)),
        "image_indices_max": int(max(image_indices)),
        "voxel_indices_min": int(min(voxel_indices)),
        "voxel_indices_max": int(max(voxel_indices)),
        "pred_tokens_shape": list(pred_tokens.shape),
        "image_tokens_shape": list(image_tokens.shape),
        "metrics_flat_tokens": geometry_metrics(pred_tokens.flatten(1), image_tokens.flatten(1)),
        "metrics_mean_pooled_tokens": geometry_metrics(pred_tokens.mean(dim=1), image_tokens.mean(dim=1)),
    }


def main():
    parser = argparse.ArgumentParser(description="Cycle 36 training-only predicted/image CLIP geometry diagnostic.")
    parser.add_argument("--data_path", type=str, default="/src")
    parser.add_argument("--output_dir", type=str, default="/src/tables/cycle36_geometry")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_sessions", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=4096)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--blurry_recon", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--checkpoint_root", type=str, default="/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for subj in (5, 7):
        for tag in ("align0", "align_low"):
            model_name = f"cycle30_subj0{subj}_{tag}_1sess_150ep"
            rows.append({
                "subj": subj,
                "tag": tag,
                "model_name": model_name,
                "checkpoint_path": os.path.join(args.checkpoint_root, model_name, "last.pth"),
            })

    summaries = []
    for row in rows:
        print(f"Running geometry diagnostic for {row['model_name']}", flush=True)
        summary = collect_embeddings(args, row, device)
        summaries.append(summary)
        out_json = os.path.join(args.output_dir, f"{row['model_name']}_geometry.json")
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(json.dumps({
            "model_name": row["model_name"],
            "image_count": summary["image_count"],
            "flat_spearman": summary["metrics_flat_tokens"]["spearman_rsa_cosine_distance"],
            "flat_overlap5": summary["metrics_flat_tokens"]["nn_overlap_at_5"],
            "pooled_spearman": summary["metrics_mean_pooled_tokens"]["spearman_rsa_cosine_distance"],
            "pooled_overlap5": summary["metrics_mean_pooled_tokens"]["nn_overlap_at_5"],
            "json": out_json,
        }, sort_keys=True), flush=True)

    csv_path = os.path.join(args.output_dir, "cycle36_geometry_summary.csv")
    metric_sets = ("metrics_flat_tokens", "metrics_mean_pooled_tokens")
    metric_names = [
        "spearman_rsa_cosine_distance",
        "pearson_offdiag_cosine_similarity",
        "nn_overlap_at_1",
        "nn_overlap_at_5",
        "nn_overlap_at_10",
        "teacher_top1_median_rank_under_pred",
        "teacher_top1_mrr_under_pred",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject",
                "tag",
                "model_name",
                "image_count",
                "unique_image_count",
                "train_url",
                "checkpoint_path",
                "training_only",
                "shared1000_or_new_test_used",
                "metric_space",
                *metric_names,
            ],
        )
        writer.writeheader()
        for summary in summaries:
            for metric_set in metric_sets:
                row = {
                    "subject": summary["subject"],
                    "tag": summary["tag"],
                    "model_name": summary["model_name"],
                    "image_count": summary["image_count"],
                    "unique_image_count": summary["unique_image_count"],
                    "train_url": summary["train_url"],
                    "checkpoint_path": summary["checkpoint_path"],
                    "training_only": summary["training_only"],
                    "shared1000_or_new_test_used": summary["shared1000_or_new_test_used"],
                    "metric_space": metric_set,
                }
                row.update({name: summary[metric_set][name] for name in metric_names})
                writer.writerow(row)

    delta_path = os.path.join(args.output_dir, "cycle36_geometry_deltas.json")
    by_key = {(s["subject"], s["tag"]): s for s in summaries}
    deltas = {}
    for subj in (5, 7):
        deltas[f"subj0{subj}"] = {}
        for metric_set in metric_sets:
            zero = by_key[(subj, "align0")][metric_set]
            low = by_key[(subj, "align_low")][metric_set]
            deltas[f"subj0{subj}"][metric_set] = {
                name: float(low[name] - zero[name]) for name in metric_names
            }
    with open(delta_path, "w") as f:
        json.dump(deltas, f, indent=2, sort_keys=True)
    print(json.dumps({"summary_csv": csv_path, "delta_json": delta_path, "deltas": deltas}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
