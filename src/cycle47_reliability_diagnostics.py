#!/usr/bin/env python
import argparse
import csv
import json
import os
import subprocess
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROWS = {
    5: {
        "margin0": "cycle44_subj05_margin0_1sess_150ep",
        "margin_low": "cycle44_subj05_margin_low_1sess_150ep",
    },
    7: {
        "margin0": "cycle44_subj07_margin0_1sess_150ep",
        "margin_low": "cycle44_subj07_margin_low_1sess_150ep",
    },
}

CSV_METRICS = [
    "PixCorr",
    "SSIM",
    "AlexNet(2)",
    "AlexNet(5)",
    "InceptionV3",
    "CLIP",
    "EffNet-B",
    "SwAV",
    "ImageRet",
    "BrainRet",
    "VC",
    "V1",
    "V2",
    "V3",
    "V4",
    "HigherVis",
]

OUTCOMES = [
    "delta_full_pool_rank",
    "delta_image_full_pool_rank",
    "delta_margin",
    "delta_sampled_top1_success_rate",
    "rank1_to_not1",
    "not1_to_rank1",
    "full_pool_rank",
    "image_full_pool_rank",
]

COVARIATES = [
    "train_clip_diag_mahalanobis",
    "train_clip_mean_cosine",
    "train_clip_mean_abs_z",
    "train_clip_max_abs_z",
    "eval_clip_nearest_neighbor_cosine",
    "eval_clip_top1_top2_gap",
    "eval_clip_density_top10",
    "global_reliability_mean",
    "early_reliability_mean",
    "higher_reliability_mean",
    "global_reliability_x_train_mahalanobis",
    "low_reliability_x_train_mahalanobis",
    "low_reliability_x_eval_density_top10",
]


def as_builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.tolist()
    return value


def require(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def scheduler_state():
    job_status = OrderedDict(path="/job-status.md", exists=os.path.exists("/job-status.md"))
    if job_status["exists"]:
        with open("/job-status.md") as f:
            job_status["content_head"] = f.read(4000)
    else:
        job_status["note"] = "No /job-status.md was visible; authenticated scheduler with squeue."
    result = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%.18i %.40j %.8T %.10M %.9l"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    active_jobs = [line for line in result.stdout.splitlines() if line.strip()]
    return OrderedDict(
        job_status=job_status,
        active_job_count=len(active_jobs),
        active_jobs=active_jobs,
        squeue_returncode=result.returncode,
        squeue_stderr=result.stderr.strip(),
    )


def load_teacher(data_path):
    candidates = [
        os.path.join(data_path, "tables/cycle39_failure_decomposition/all_images_openclip_bigG_flat_norm.pt"),
        os.path.join(data_path, "tables/cycle38_neighbor_diagnostics/all_images_openclip_bigG_flat_norm.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            teacher = torch.load(path, map_location="cpu").float()
            if teacher.ndim != 2 or teacher.shape[0] != 1000:
                raise RuntimeError(f"Unexpected teacher cache shape at {path}: {tuple(teacher.shape)}")
            return F.normalize(teacher, dim=-1), path
    raise FileNotFoundError("Missing existing OpenCLIP teacher cache.")


def pooled_clip_features(teacher):
    if teacher.shape[1] % 1664 != 0:
        raise RuntimeError(f"Cannot reshape teacher feature dim {teacher.shape[1]} into 1664-wide tokens")
    tokens = teacher.reshape(teacher.shape[0], teacher.shape[1] // 1664, 1664)
    return F.normalize(tokens.mean(dim=1), dim=-1)


def ranks_from_similarity(sim):
    order = torch.argsort(sim, dim=1, descending=True)
    labels = torch.arange(sim.shape[0], device=sim.device)
    return (order == labels[:, None]).nonzero()[:, 1].float() + 1.0


def load_student(data_path, model_name):
    path = require(os.path.join(data_path, "evals", model_name, f"{model_name}_all_clipvoxels.pt"))
    raw = torch.load(path, map_location="cpu")
    if not torch.isfinite(raw).all():
        raise RuntimeError(f"Non-finite all_clipvoxels tensor: {path}")
    return F.normalize(raw.float().flatten(1), dim=-1), path


def image_retrieval_rank(data_path, model_name, teacher):
    student, path = load_student(data_path, model_name)
    sim = teacher @ student.T
    ranks = ranks_from_similarity(sim).cpu().numpy()
    del student, sim
    return ranks, path


def load_final_csv(data_path, model_name):
    path = require(os.path.join(data_path, "tables", f"{model_name}_all_enhancedrecons.csv"))
    df = pd.read_csv(path, sep="\t")
    values = [float(v) for v in df["Value"].tolist()]
    if len(values) != len(CSV_METRICS):
        raise RuntimeError(f"Unexpected metric count in {path}: {len(values)}")
    return OrderedDict(zip(CSV_METRICS, values)), path


def load_cycle44_replay_check(data_path, subject):
    path = require(os.path.join(data_path, "tables/cycle44_rank_margin_diagnostics/cycle44_rank_margin_summary.json"))
    with open(path) as f:
        summary = json.load(f)
    for row in summary.get("replay_checks", []):
        if int(row["subject"]) == int(subject):
            return OrderedDict(
                source=path,
                csv_brainret_delta=float(row["csv_brainret_delta"]),
                replayed_sampled_top1_delta=float(row["replayed_sampled_top1_delta"]),
                replay_minus_csv=float(row["replay_minus_csv_delta"]),
                within_0p002_tolerance=bool(row["within_0p002_tolerance"]),
            )
    raise RuntimeError(f"Missing Cycle 44 replay check for subject {subject} in {path}")


def artifact_availability(data_path):
    suffixes = ["all_clipvoxels", "all_recons", "all_enhancedrecons"]
    out = OrderedDict()
    missing = []
    for subject, rows in ROWS.items():
        for label, model_name in rows.items():
            key = f"subj{subject:02d}_{label}"
            base = os.path.join(data_path, "evals", model_name)
            row = OrderedDict(eval_dir=base, eval_dir_exists=os.path.isdir(base))
            if not row["eval_dir_exists"]:
                missing.append(base)
            for suffix in suffixes:
                path = os.path.join(base, f"{model_name}_{suffix}.pt")
                row[suffix] = OrderedDict(path=path, exists=os.path.exists(path))
                if os.path.exists(path):
                    row[suffix]["size_bytes"] = int(os.path.getsize(path))
                else:
                    missing.append(path)
            csv_path = os.path.join(data_path, "tables", f"{model_name}_all_enhancedrecons.csv")
            row["final_csv"] = OrderedDict(path=csv_path, exists=os.path.exists(csv_path))
            if not os.path.exists(csv_path):
                missing.append(csv_path)
            out[key] = row
    diag_dir = os.path.join(data_path, "tables/cycle44_rank_margin_diagnostics")
    for subject in ROWS:
        path = os.path.join(diag_dir, f"subj{subject:02d}_margin_low_vs_margin0_rank_margin_delta.csv")
        out[f"subj{subject:02d}_rank_margin_delta_csv"] = OrderedDict(path=path, exists=os.path.exists(path))
        if not os.path.exists(path):
            missing.append(path)
    return out, missing


def load_reliability(data_path, subject):
    json_path = require(os.path.join(data_path, "reliability/trainrepeat_reliability_summary.json"))
    with open(json_path) as f:
        summary = json.load(f)[f"subj{subject:02d}"]
    required_zero = [
        "train_new_test_overlap_count",
        "train_old_test_overlap_count",
        "repeat_new_test_overlap_count",
        "repeat_old_test_overlap_count",
    ]
    for key in required_zero:
        if int(summary[key]) != 0:
            raise RuntimeError(f"Reliability leakage check failed subj{subject:02d}: {key}={summary[key]}")
    return summary, json_path


def train_clip_covariates(data_path, subject, eval_pooled):
    path = require(os.path.join(data_path, "tables", f"cycle30_subj{subject:02d}_clip_train_stats.pt"))
    stats = torch.load(path, map_location="cpu")
    if not bool(stats.get("training_only")):
        raise RuntimeError(f"Training CLIP stats are not marked training_only: {path}")
    mean = stats["mean"].float()
    cov = stats["cov"].float()
    if mean.shape[0] != eval_pooled.shape[1] or cov.shape[0] != mean.shape[0]:
        raise RuntimeError(f"Train stats shape mismatch for subj{subject:02d}")
    diag_var = torch.diag(cov).clamp_min(1e-6)
    centered = eval_pooled - mean[None, :]
    z = centered / torch.sqrt(diag_var)[None, :]
    return OrderedDict(
        path=path,
        training_only=bool(stats.get("training_only")),
        source=stats.get("source", ""),
        count=int(stats.get("count", -1)),
        train_clip_diag_mahalanobis=(z.square().mean(dim=1)).cpu().numpy(),
        train_clip_mean_abs_z=(z.abs().mean(dim=1)).cpu().numpy(),
        train_clip_max_abs_z=(z.abs().max(dim=1).values).cpu().numpy(),
        train_clip_mean_cosine=F.cosine_similarity(eval_pooled, mean[None, :], dim=1).cpu().numpy(),
    )


def eval_clip_crowding(eval_pooled):
    sim = eval_pooled @ eval_pooled.T
    sim.fill_diagonal_(-float("inf"))
    top = torch.topk(sim, k=10, dim=1).values
    return OrderedDict(
        eval_clip_nearest_neighbor_cosine=top[:, 0].cpu().numpy(),
        eval_clip_top1_top2_gap=(top[:, 0] - top[:, 1]).cpu().numpy(),
        eval_clip_density_top10=top.mean(dim=1).cpu().numpy(),
    )


def correlation(x, y, method):
    sx = pd.Series(x, dtype="float64")
    sy = pd.Series(y, dtype="float64")
    mask = sx.notna() & sy.notna()
    if int(mask.sum()) < 3:
        return None
    sx = sx[mask]
    sy = sy[mask]
    if float(sx.std()) == 0.0 or float(sy.std()) == 0.0:
        return None
    return float(sx.corr(sy, method=method))


def stratified_summary(df, covariate, outcomes):
    q25 = float(df[covariate].quantile(0.25))
    q75 = float(df[covariate].quantile(0.75))
    low = df[df[covariate] <= q25]
    high = df[df[covariate] >= q75]
    out = OrderedDict(covariate=covariate, q25=q25, q75=q75, low_n=int(len(low)), high_n=int(len(high)))
    for outcome in outcomes:
        out[f"{outcome}_low_mean"] = float(low[outcome].mean())
        out[f"{outcome}_high_mean"] = float(high[outcome].mean())
        out[f"{outcome}_high_minus_low"] = float(high[outcome].mean() - low[outcome].mean())
    return out


def summarize_subject(data_path, outdir, subject, teacher, eval_pooled, crowding):
    reliability, reliability_path = load_reliability(data_path, subject)
    train_cov = train_clip_covariates(data_path, subject, eval_pooled)
    delta_path = require(
        os.path.join(data_path, "tables/cycle44_rank_margin_diagnostics", f"subj{subject:02d}_margin_low_vs_margin0_rank_margin_delta.csv")
    )
    df = pd.read_csv(delta_path)
    if len(df) != 1000:
        raise RuntimeError(f"Expected 1000 rank/margin rows, found {len(df)} in {delta_path}")

    image_rank_paths = {}
    image_ranks = {}
    for label, model_name in ROWS[subject].items():
        ranks, path = image_retrieval_rank(data_path, model_name, teacher)
        image_ranks[label] = ranks
        image_rank_paths[label] = path
    df["image_full_pool_rank_margin0"] = image_ranks["margin0"]
    df["image_full_pool_rank"] = image_ranks["margin_low"]
    df["delta_image_full_pool_rank"] = image_ranks["margin_low"] - image_ranks["margin0"]

    for key, values in crowding.items():
        df[key] = values
    for key in ["train_clip_diag_mahalanobis", "train_clip_mean_cosine", "train_clip_mean_abs_z", "train_clip_max_abs_z"]:
        df[key] = train_cov[key]

    global_rel = float(reliability["shrunk_mean"])
    early_rel = float(reliability["shrunk_early_mean"])
    higher_rel = float(reliability["shrunk_higher_mean"])
    df["global_reliability_mean"] = global_rel
    df["early_reliability_mean"] = early_rel
    df["higher_reliability_mean"] = higher_rel
    df["low_reliability_mean"] = 1.0 - global_rel
    df["global_reliability_x_train_mahalanobis"] = global_rel * df["train_clip_diag_mahalanobis"]
    df["low_reliability_x_train_mahalanobis"] = (1.0 - global_rel) * df["train_clip_diag_mahalanobis"]
    df["low_reliability_x_eval_density_top10"] = (1.0 - global_rel) * df["eval_clip_density_top10"]
    df["heldout_or_newtest_labels_used_for_reliability"] = False
    df["heldout_or_newtest_labels_used_for_train_clip_stats"] = False
    df["heldout_or_newtest_labels_used_for_image_ambiguity"] = False

    out_csv = os.path.join(outdir, f"subj{subject:02d}_cycle47_reliability_diagnostics.csv")
    df.to_csv(out_csv, index=False)

    metric_values = {label: load_final_csv(data_path, model_name)[0] for label, model_name in ROWS[subject].items()}
    csv_brainret_delta = metric_values["margin_low"]["BrainRet"] - metric_values["margin0"]["BrainRet"]
    replay_check = load_cycle44_replay_check(data_path, subject)

    correlations = []
    for covariate in COVARIATES:
        for outcome in OUTCOMES:
            correlations.append(
                OrderedDict(
                    subject=subject,
                    covariate=covariate,
                    outcome=outcome,
                    pearson=correlation(df[covariate], df[outcome], "pearson"),
                    spearman=correlation(df[covariate], df[outcome], "spearman"),
                )
            )
    corr_csv = os.path.join(outdir, f"subj{subject:02d}_cycle47_correlations.csv")
    with open(corr_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(correlations[0].keys()))
        writer.writeheader()
        writer.writerows(correlations)

    stratified = [
        stratified_summary(df, "train_clip_diag_mahalanobis", ["delta_full_pool_rank", "rank1_to_not1", "delta_margin"]),
        stratified_summary(df, "eval_clip_density_top10", ["delta_full_pool_rank", "rank1_to_not1", "delta_margin"]),
        stratified_summary(df, "low_reliability_x_train_mahalanobis", ["delta_full_pool_rank", "rank1_to_not1", "delta_margin"]),
    ]

    stress = df[(df["delta_image_full_pool_rank"] <= 0) & (df["delta_full_pool_rank"] > 0)]
    stress_summary = OrderedDict(
        image_rank_preserved_or_improved_brain_rank_worse_n=int(len(stress)),
        total_n=int(len(df)),
        mean_train_clip_diag_mahalanobis=float(stress["train_clip_diag_mahalanobis"].mean()) if len(stress) else None,
        mean_eval_clip_density_top10=float(stress["eval_clip_density_top10"].mean()) if len(stress) else None,
        rank1_to_not1_n=int(df["rank1_to_not1"].sum()),
        not1_to_rank1_n=int(df["not1_to_rank1"].sum()),
    )

    top_corr = sorted(
        [row for row in correlations if row["outcome"] in ("delta_full_pool_rank", "rank1_to_not1", "delta_margin") and row["spearman"] is not None],
        key=lambda row: abs(row["spearman"]),
        reverse=True,
    )[:12]

    return OrderedDict(
        subject=subject,
        output_csv=out_csv,
        correlation_csv=corr_csv,
        reliability_summary_path=reliability_path,
        reliability_method=reliability["provenance"],
        reliability=OrderedDict(
            repeat_image_count=int(reliability["repeat_image_count"]),
            repeat_trial_count=int(reliability["repeat_trial_count"]),
            train_row_count=int(reliability["train_row_count"]),
            unique_train_image_count=int(reliability["unique_train_image_count"]),
            global_mean=global_rel,
            early_mean=early_rel,
            higher_mean=higher_rel,
            heldout_or_newtest_used=False,
            overlap_counts=OrderedDict(
                train_new_test=int(reliability["train_new_test_overlap_count"]),
                train_old_test=int(reliability["train_old_test_overlap_count"]),
                repeat_new_test=int(reliability["repeat_new_test_overlap_count"]),
                repeat_old_test=int(reliability["repeat_old_test_overlap_count"]),
            ),
        ),
        train_clip_stats=OrderedDict(
            path=train_cov["path"],
            training_only=bool(train_cov["training_only"]),
            count=int(train_cov["count"]),
            source=train_cov["source"],
            heldout_or_newtest_labels_used=False,
        ),
        image_rank_paths=image_rank_paths,
        replay_check=replay_check,
        metric_deltas=OrderedDict(
            BrainRet=float(csv_brainret_delta),
            ImageRet=float(metric_values["margin_low"]["ImageRet"] - metric_values["margin0"]["ImageRet"]),
            CLIP=float(metric_values["margin_low"]["CLIP"] - metric_values["margin0"]["CLIP"]),
            PixCorr=float(metric_values["margin_low"]["PixCorr"] - metric_values["margin0"]["PixCorr"]),
            VC=float(metric_values["margin_low"]["VC"] - metric_values["margin0"]["VC"]),
            HigherVis=float(metric_values["margin_low"]["HigherVis"] - metric_values["margin0"]["HigherVis"]),
        ),
        rank_delta_means=OrderedDict(
            brain_full_pool=float(df["delta_full_pool_rank"].mean()),
            image_full_pool=float(df["delta_image_full_pool_rank"].mean()),
            margin=float(df["delta_margin"].mean()),
            sampled_top1=float(df["delta_sampled_top1_success_rate"].mean()),
        ),
        stress_summary=stress_summary,
        stratified=stratified,
        strongest_correlations=top_corr,
    )


def main():
    parser = argparse.ArgumentParser(description="Cycle 47 training-only reliability and ambiguity diagnostics.")
    parser.add_argument("--data_path", default="/src")
    parser.add_argument("--outdir", default="/src/tables/cycle47_reliability_diagnostics")
    parser.add_argument("--subjects", nargs="+", type=int, default=[5, 7], choices=[5, 7])
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sched = scheduler_state()
    if sched["active_job_count"]:
        raise RuntimeError(f"Active jobs found; stop per plan: {sched['active_jobs']}")
    artifacts, missing = artifact_availability(args.data_path)
    if missing:
        raise FileNotFoundError(json.dumps(missing, indent=2))

    teacher, teacher_path = load_teacher(args.data_path)
    eval_pooled = pooled_clip_features(teacher)
    crowding = eval_clip_crowding(eval_pooled)

    subjects = []
    for subject in args.subjects:
        print(f"cycle47 diagnostics: subject {subject} start", flush=True)
        subjects.append(summarize_subject(args.data_path, args.outdir, subject, teacher, eval_pooled, crowding))
        print(f"cycle47 diagnostics: subject {subject} complete", flush=True)

    summary = OrderedDict(
        provenance=OrderedDict(
            cycle=47,
            scope="read-only post-hoc diagnostic; no training, evaluator relaunch, checkpoint write, or model edit",
            data_path=args.data_path,
            outdir=args.outdir,
            teacher_cache=teacher_path,
            reliability_source="/src/reliability/trainrepeat_reliability_summary.json",
            training_clip_stats_source="/src/tables/cycle30_subj0{5,7}_clip_train_stats.pt",
            heldout_or_newtest_labels_used_for_reliability=False,
            heldout_or_newtest_labels_used_for_train_clip_stats=False,
            heldout_or_newtest_labels_used_for_image_ambiguity=False,
            image_ambiguity_note="Uses saved evaluator image OpenCLIP features for unlabeled image-content crowding plus training-only CLIP mean/covariance distance.",
        ),
        preflight=OrderedDict(scheduler=sched, artifacts=artifacts, missing=missing),
        subjects=subjects,
    )
    summary_path = os.path.join(args.outdir, "cycle47_reliability_diagnostics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=as_builtin)
    print(json.dumps(summary, indent=2, default=as_builtin)[:30000])
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
