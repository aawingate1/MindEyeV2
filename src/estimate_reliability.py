#!/usr/bin/env python
import argparse
import csv
import io
import json
import os
import tarfile
from collections import defaultdict

import h5py
import numpy as np
import torch


EXPECTED_VOXELS = {5: 13039, 7: 12682}
EXPECTED_MASK_COUNTS = {
    5: {"early_vis": 3661, "higher_vis": 9378},
    7: {"early_vis": 3251, "higher_vis": 9431},
}


def read_behav_rows(wds_root, subj, split, shards):
    rows = []
    names = []
    for shard in shards:
        tar_path = os.path.join(wds_root, f"subj{subj:02d}", split, f"{shard}.tar")
        if not os.path.exists(tar_path):
            raise FileNotFoundError(tar_path)
        with tarfile.open(tar_path) as tar:
            for member in tar:
                if not member.name.endswith(".behav.npy"):
                    continue
                with tar.extractfile(member) as fh:
                    arr = np.load(io.BytesIO(fh.read()))
                rows.append(arr[0])
                names.append(f"{split}/{shard}.tar:{member.name}")
    if not rows:
        raise RuntimeError(f"No behav.npy rows found for subj{subj:02d} {split} shards {shards}")
    return np.stack(rows), names


def split_half_reliability(beta_matrix, image_ids, beta_indices):
    groups = defaultdict(list)
    for image_id, beta_idx in zip(image_ids.tolist(), beta_indices.tolist()):
        groups[int(image_id)].append(int(beta_idx))
    repeat_groups = {k: sorted(v) for k, v in groups.items() if len(v) >= 2}
    if not repeat_groups:
        raise RuntimeError("No repeated image IDs found in the requested training split")

    left_rows = []
    right_rows = []
    repeat_trial_count = 0
    for beta_rows in repeat_groups.values():
        repeat_trial_count += len(beta_rows)
        left_idx = beta_rows[0::2]
        right_idx = beta_rows[1::2]
        if len(right_idx) == 0:
            continue
        left_rows.append(beta_matrix[left_idx].mean(axis=0))
        right_rows.append(beta_matrix[right_idx].mean(axis=0))

    left = np.stack(left_rows).astype(np.float64)
    right = np.stack(right_rows).astype(np.float64)
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    denom = np.sqrt((left * left).sum(axis=0) * (right * right).sum(axis=0))
    raw = np.full(left.shape[1], np.nan, dtype=np.float32)
    valid = denom > 0
    raw[valid] = ((left * right).sum(axis=0)[valid] / denom[valid]).astype(np.float32)
    raw = np.clip(raw, -1.0, 1.0)
    return raw, repeat_groups, repeat_trial_count


def shrink_by_roi(raw, masks, shrinkage):
    valid = np.isfinite(raw)
    global_mean = float(np.nanmean(raw[valid])) if valid.any() else 0.0
    shrunk = raw.copy()
    roi_means = {}

    assigned = np.zeros(raw.shape[0], dtype=bool)
    for roi_name in ["early_vis", "higher_vis"]:
        mask = masks[roi_name].astype(bool)
        assigned |= mask
        roi_valid = valid & mask
        roi_mean = float(np.nanmean(raw[roi_valid])) if roi_valid.any() else global_mean
        roi_means[roi_name] = roi_mean
        fill = np.where(valid & mask, raw, roi_mean)
        shrunk[mask] = (1.0 - shrinkage) * fill[mask] + shrinkage * roi_mean

    other = ~assigned
    if other.any():
        other_valid = valid & other
        other_mean = float(np.nanmean(raw[other_valid])) if other_valid.any() else global_mean
        roi_means["other"] = other_mean
        fill = np.where(valid & other, raw, other_mean)
        shrunk[other] = (1.0 - shrinkage) * fill[other] + shrinkage * other_mean

    shrunk = np.nan_to_num(shrunk, nan=global_mean, posinf=global_mean, neginf=global_mean)
    return np.clip(shrunk.astype(np.float32), -1.0, 1.0), roi_means


def summarize_vector(vec, prefix):
    q = np.quantile(vec, [0.0, 0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99, 1.0])
    keys = ["min", "p01", "p05", "p10", "p25", "median", "p75", "p90", "p95", "p99", "max"]
    out = {f"{prefix}_{k}": float(v) for k, v in zip(keys, q)}
    out[f"{prefix}_mean"] = float(np.mean(vec))
    out[f"{prefix}_std"] = float(np.std(vec))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/src")
    parser.add_argument("--subjects", nargs="+", type=int, default=[5, 7])
    parser.add_argument("--num_sessions", type=int, default=1)
    parser.add_argument("--shrinkage", type=float, default=0.25)
    parser.add_argument("--out_dir", default="/src/reliability")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows_for_csv = []
    summaries = {}

    with h5py.File(os.path.join(args.data_path, "COCO_73k_subj_indices.hdf5"), "r") as index_h5, \
            h5py.File(os.path.join(args.data_path, "brain_region_masks.hdf5"), "r") as mask_h5:
        for subj in args.subjects:
            subj_key = f"subj{subj:02d}"
            train_rows, _ = read_behav_rows(os.path.join(args.data_path, "wds"), subj, "train", range(args.num_sessions))
            new_test_rows, _ = read_behav_rows(os.path.join(args.data_path, "wds"), subj, "new_test", [0])
            old_test_rows, _ = read_behav_rows(os.path.join(args.data_path, "wds"), subj, "test", [0])

            image_ids = train_rows[:, 0].astype(np.int64)
            beta_indices = train_rows[:, 5].astype(np.int64)
            new_test_ids = set(new_test_rows[:, 0].astype(np.int64).tolist())
            old_test_ids = set(old_test_rows[:, 0].astype(np.int64).tolist())
            repeated_train_ids = {int(k) for k, v in defaultdict(list, {}).items()}
            overlap_new = sorted(set(image_ids.tolist()) & new_test_ids)
            overlap_old = sorted(set(image_ids.tolist()) & old_test_ids)

            subj_indices = index_h5[subj_key][:]
            if not np.all(subj_indices[beta_indices] == image_ids):
                raise RuntimeError(f"{subj_key}: COCO_73k_subj_indices does not match behav image_id/beta_idx rows")

            beta_path = os.path.join(args.data_path, f"betas_all_{subj_key}_fp32_renorm.hdf5")
            with h5py.File(beta_path, "r") as beta_h5:
                betas = beta_h5["betas"]
                num_voxels = int(betas.shape[1])
                if num_voxels != EXPECTED_VOXELS[subj]:
                    raise RuntimeError(f"{subj_key}: expected {EXPECTED_VOXELS[subj]} voxels, found {num_voxels}")
                needed = np.unique(beta_indices)
                beta_subset = {int(i): betas[int(i)] for i in needed.tolist()}
                beta_matrix = np.zeros((int(beta_indices.max()) + 1, num_voxels), dtype=np.float32)
                for i, beta in beta_subset.items():
                    beta_matrix[i] = beta

            raw, repeat_groups, repeat_trial_count = split_half_reliability(beta_matrix, image_ids, beta_indices)
            repeated_train_ids = set(repeat_groups.keys())
            repeat_overlap_new = sorted(repeated_train_ids & new_test_ids)
            repeat_overlap_old = sorted(repeated_train_ids & old_test_ids)
            if repeat_overlap_new:
                raise RuntimeError(f"{subj_key}: repeated train IDs overlap new_test: {repeat_overlap_new[:10]}")
            if repeat_overlap_old:
                raise RuntimeError(f"{subj_key}: repeated train IDs overlap old test: {repeat_overlap_old[:10]}")

            masks = {name: mask_h5[subj_key][name][:].astype(bool) for name in ["early_vis", "higher_vis"]}
            for name, expected_count in EXPECTED_MASK_COUNTS[subj].items():
                actual_count = int(masks[name].sum())
                if actual_count != expected_count:
                    raise RuntimeError(f"{subj_key}: {name} mask count {actual_count}, expected {expected_count}")
                if masks[name].shape[0] != num_voxels:
                    raise RuntimeError(f"{subj_key}: {name} mask length mismatch")

            reliability, roi_raw_means = shrink_by_roi(raw, masks, args.shrinkage)
            invalid_count = int((~np.isfinite(raw)).sum())
            degenerate_count = invalid_count
            summary = {
                "subject": subj,
                "num_sessions": args.num_sessions,
                "train_row_count": int(train_rows.shape[0]),
                "unique_train_image_count": int(len(set(image_ids.tolist()))),
                "repeat_image_count": int(len(repeat_groups)),
                "repeat_trial_count": int(repeat_trial_count),
                "num_voxels": num_voxels,
                "invalid_voxel_count": invalid_count,
                "degenerate_voxel_count": degenerate_count,
                "shrinkage": float(args.shrinkage),
                "train_new_test_overlap_count": int(len(overlap_new)),
                "train_old_test_overlap_count": int(len(overlap_old)),
                "repeat_new_test_overlap_count": int(len(repeat_overlap_new)),
                "repeat_old_test_overlap_count": int(len(repeat_overlap_old)),
                "early_mask_count": int(masks["early_vis"].sum()),
                "higher_mask_count": int(masks["higher_vis"].sum()),
                "raw_early_mean": float(np.nanmean(raw[masks["early_vis"]])),
                "raw_higher_mean": float(np.nanmean(raw[masks["higher_vis"]])),
                "shrunk_early_mean": float(np.mean(reliability[masks["early_vis"]])),
                "shrunk_higher_mean": float(np.mean(reliability[masks["higher_vis"]])),
                "roi_raw_means": roi_raw_means,
                "provenance": "split_half_train_repeat_behav_col0_image_id_col5_beta_idx_wds_train0_only",
            }
            summary.update(summarize_vector(raw[np.isfinite(raw)], "raw_valid"))
            summary.update(summarize_vector(reliability, "shrunk"))
            summaries[subj_key] = summary
            rows_for_csv.append({k: json.dumps(v) if isinstance(v, dict) else v for k, v in summary.items()})

            tensor_path = os.path.join(args.out_dir, f"{subj_key}_trainrepeat_reliability.pt")
            torch.save(
                {
                    "reliability": torch.from_numpy(reliability),
                    "raw_reliability": torch.from_numpy(np.nan_to_num(raw, nan=0.0)),
                    "early_vis_mask": torch.from_numpy(masks["early_vis"]),
                    "higher_vis_mask": torch.from_numpy(masks["higher_vis"]),
                    "summary": summary,
                },
                tensor_path,
            )
            summary["tensor_path"] = tensor_path
            print(json.dumps(summary, indent=2, sort_keys=True))

    json_path = os.path.join(args.out_dir, "trainrepeat_reliability_summary.json")
    csv_path = os.path.join(args.out_dir, "trainrepeat_reliability_summary.csv")
    with open(json_path, "w") as f:
        json.dump(summaries, f, indent=2, sort_keys=True)
    with open(csv_path, "w", newline="") as f:
        fieldnames = sorted({key for row in rows_for_csv for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_for_csv)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
