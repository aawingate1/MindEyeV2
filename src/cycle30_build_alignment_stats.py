#!/usr/bin/env python
import argparse
import json
import os
import random
import sys

import h5py
import numpy as np
import torch
import webdataset as wds

sys.path.append("generative_models/")
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder


def my_split_by_node(urls):
    return urls


def update_moments(count, sum_x, sum_xx, features):
    features = features.float().cpu()
    if sum_x is None:
        sum_x = torch.zeros(features.shape[1], dtype=torch.float64)
        sum_xx = torch.zeros(features.shape[1], features.shape[1], dtype=torch.float64)
    count += int(features.shape[0])
    sum_x += features.double().sum(dim=0)
    sum_xx += features.double().T @ features.double()
    return count, sum_x, sum_xx


def finalize_moments(count, sum_x, sum_xx):
    if count < 2:
        raise ValueError(f"Need at least 2 samples for covariance, got {count}")
    mean = sum_x / count
    cov = (sum_xx - count * torch.outer(mean, mean)) / (count - 1)
    return mean.float(), cov.float()


def main():
    parser = argparse.ArgumentParser(description="Build Cycle 30 training-only functional alignment stats.")
    parser.add_argument("--data_path", type=str, default="/src")
    parser.add_argument("--cache_dir", type=str, default="/src")
    parser.add_argument("--subj", type=int, required=True, choices=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--num_sessions", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_type = torch.float16
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    train_url = f"{args.data_path}/wds/subj0{args.subj}/train/" + "{0.." + f"{args.num_sessions - 1}" + "}.tar"
    dataset = (
        wds.WebDataset(train_url, resampled=False, nodesplitter=my_split_by_node)
        .shuffle(750, initial=1500, rng=random.Random(42))
        .decode("torch")
        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")
        .to_tuple("behav", "past_behav", "future_behav", "olds_behav")
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True)

    images_h5 = h5py.File(f"{args.data_path}/coco_images_224_float16.hdf5", "r")
    images = images_h5["images"]
    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
    )
    clip_img_embedder.eval().requires_grad_(False).to(device)

    count = 0
    sum_x = None
    sum_xx = None
    skipped_duplicate_batches = 0
    used_image_indices = []

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=data_type):
        for batch_i, (behav0, _past_behav0, _future_behav0, _old_behav0) in enumerate(loader):
            image_idx = behav0[:, 0, 0].cpu().long().numpy()
            image0, image_sorted_idx = np.unique(image_idx, return_index=True)
            if len(image0) != len(image_idx):
                skipped_duplicate_batches += 1
                continue
            image = torch.tensor(images[image0], dtype=data_type, device=device)
            clip_tokens = clip_img_embedder(image)
            features = clip_tokens.float().mean(dim=1)
            count, sum_x, sum_xx = update_moments(count, sum_x, sum_xx, features)
            used_image_indices.extend([int(x) for x in image0.tolist()])
            if args.max_batches is not None and (batch_i + 1) >= args.max_batches:
                break

    mean, cov = finalize_moments(count, sum_x, sum_xx)
    cov_diag = torch.diagonal(cov)
    provenance = {
        "subject": int(args.subj),
        "num_sessions": int(args.num_sessions),
        "batch_size": int(args.batch_size),
        "train_url": train_url,
        "count": int(count),
        "skipped_duplicate_batches": int(skipped_duplicate_batches),
        "unique_image_count": int(len(set(used_image_indices))),
        "min_image_index": int(min(used_image_indices)) if used_image_indices else None,
        "max_image_index": int(max(used_image_indices)) if used_image_indices else None,
        "site": "clip",
        "feature": "FrozenOpenCLIPImageEmbedder tokens mean-pooled over sequence",
        "training_only": True,
        "test_sources_used": [],
        "shared1000_or_new_test_used": False,
    }
    payload = {
        "site": "clip",
        "mean": mean,
        "cov": cov,
        "count": int(count),
        "training_only": True,
        "source": train_url,
        "provenance": provenance,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(payload, args.output)

    mean_norm = float(torch.linalg.vector_norm(mean))
    cov_norm = float(torch.linalg.matrix_norm(cov))
    summary = {
        **provenance,
        "output": args.output,
        "mean_shape": list(mean.shape),
        "cov_shape": list(cov.shape),
        "dtype": str(mean.dtype),
        "mean_norm": mean_norm,
        "cov_norm": cov_norm,
        "cov_diag_min": float(cov_diag.min()),
        "cov_diag_max": float(cov_diag.max()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
