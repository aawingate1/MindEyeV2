import glob
import io
import json
import os
import tarfile
from typing import Dict, List, Optional, Tuple

import numpy as np


def numeric_tar_sort_key(path: str):
    stem = os.path.basename(path).split(".")[0]
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def sorted_tar_files(split_dir: str) -> List[str]:
    tar_files = sorted(glob.glob(os.path.join(split_dir, "*.tar")), key=numeric_tar_sort_key)
    if not tar_files:
        raise FileNotFoundError(f"No tar shards found in {split_dir}")
    return tar_files


def resolve_subject_wds_dir(wds_root: str, subject: str) -> str:
    subject_dir = os.path.join(wds_root, subject)
    if not os.path.isdir(subject_dir):
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")
    return subject_dir


def resolve_split_dir(subject_wds_dir: str, split_name: Optional[str], fallback_name: str) -> str:
    if split_name is None:
        split_dir = os.path.join(subject_wds_dir, fallback_name)
    elif os.path.isabs(split_name):
        split_dir = split_name
    else:
        split_dir = os.path.join(subject_wds_dir, split_name)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    return split_dir


def discover_train_eval_tar_files(
    wds_root: str,
    subject: str,
    train_split: Optional[str] = None,
    eval_split: Optional[str] = None,
    use_new_test: bool = True,
    num_sessions: Optional[int] = None,
) -> Tuple[str, str, List[str], List[str]]:
    subject_wds_dir = resolve_subject_wds_dir(wds_root, subject)
    train_dir = resolve_split_dir(subject_wds_dir, train_split, "train")
    eval_dir = resolve_split_dir(subject_wds_dir, eval_split, "new_test" if use_new_test else "test")

    train_tar_files = sorted_tar_files(train_dir)
    if train_split is None and num_sessions is not None:
        train_tar_files = train_tar_files[:num_sessions]
    eval_tar_files = sorted_tar_files(eval_dir)
    return train_dir, eval_dir, train_tar_files, eval_tar_files


def collect_behav_records(tar_files: List[str]) -> Dict[str, np.ndarray]:
    global_trials = []
    run_ids = []
    sessions = []
    cocoidxs = []
    seen = set()

    for tar_path in tar_files:
        with tarfile.open(tar_path, "r") as tar:
            for member_name in tar.getnames():
                if not member_name.endswith(".behav.npy"):
                    continue
                with tar.extractfile(member_name) as fobj:
                    behav = np.load(io.BytesIO(fobj.read()))[0]
                global_trial = int(behav[5])
                if global_trial in seen:
                    continue
                seen.add(global_trial)
                sess = int(behav[2])
                run = int(behav[3])
                global_trials.append(global_trial)
                sessions.append(sess)
                run_ids.append((sess - 1) * 12 + run)
                cocoidxs.append(int(behav[0]))

    if not global_trials:
        raise ValueError("No behav.npy records found in provided tar files")

    order = np.argsort(np.asarray(global_trials, dtype=np.int64))
    return {
        "global_trials": np.asarray(global_trials, dtype=np.int64)[order],
        "run_ids": np.asarray(run_ids, dtype=np.int64)[order],
        "sessions": np.asarray(sessions, dtype=np.int64)[order],
        "cocoidxs": np.asarray(cocoidxs, dtype=np.int64)[order],
    }


def count_samples_in_tar_files(tar_files: List[str]) -> int:
    total = 0
    for tar_path in tar_files:
        with tarfile.open(tar_path, "r") as tar:
            total += sum(1 for member_name in tar.getnames() if member_name.endswith(".behav.npy"))
    return total


def materialize_subj01_three_way_split(
    source_wds_root: str,
    output_wds_root: str,
    subject: str = "subj01",
) -> str:
    if subject != "subj01":
        raise ValueError("The 3-way split materializer is currently scoped to subj01.")

    source_subject_dir = resolve_subject_wds_dir(source_wds_root, subject)
    output_subject_dir = os.path.join(output_wds_root, subject)
    os.makedirs(output_subject_dir, exist_ok=True)

    split_specs = {
        "set1_clipphate": [os.path.join(source_subject_dir, "train", f"{i}.tar") for i in range(20)],
        "set2_mindeye_train": [os.path.join(source_subject_dir, "train", f"{i}.tar") for i in range(20, 40)],
        "set3_mindeye_test": [os.path.join(source_subject_dir, "new_test", "0.tar")],
    }

    manifest = {
        "subject": subject,
        "source_wds_root": os.path.abspath(source_wds_root),
        "output_wds_root": os.path.abspath(output_wds_root),
        "splits": {},
        "overlap_checks": {},
    }

    split_trials = {}
    for split_name, source_paths in split_specs.items():
        split_dir = os.path.join(output_subject_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for source_path in source_paths:
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Expected source shard missing: {source_path}")
            link_path = os.path.join(split_dir, os.path.basename(source_path))
            if os.path.islink(link_path):
                if os.path.realpath(link_path) != os.path.realpath(source_path):
                    os.unlink(link_path)
                    os.symlink(source_path, link_path)
            elif os.path.exists(link_path):
                raise FileExistsError(f"Refusing to overwrite non-symlink path: {link_path}")
            else:
                os.symlink(source_path, link_path)

        split_tar_files = sorted_tar_files(split_dir)
        behav = collect_behav_records(split_tar_files)
        split_trials[split_name] = behav["global_trials"]
        manifest["splits"][split_name] = {
            "split_dir": os.path.abspath(split_dir),
            "source_tar_paths": [os.path.abspath(p) for p in source_paths],
            "shard_names": [os.path.basename(p) for p in split_tar_files],
            "session_numbers": sorted(np.unique(behav["sessions"]).tolist()),
            "unique_global_trial_count": int(len(behav["global_trials"])),
        }

    split_names = list(split_specs.keys())
    for i, left_name in enumerate(split_names):
        for right_name in split_names[i + 1:]:
            overlap = np.intersect1d(split_trials[left_name], split_trials[right_name])
            manifest["overlap_checks"][f"{left_name}__{right_name}"] = {
                "overlap_count": int(len(overlap)),
                "overlap_global_trials": overlap.tolist(),
            }

    manifest_path = os.path.join(output_subject_dir, "manifest.json")
    with open(manifest_path, "w", encoding="ascii") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path
