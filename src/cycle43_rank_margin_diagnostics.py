import argparse
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
        "prior0": "cycle40_subj05_prior0_1sess_150ep",
        "prior_low": "cycle40_subj05_prior_low_1sess_150ep",
    },
    7: {
        "prior0": "cycle40_subj07_prior0_1sess_150ep",
        "prior_low": "cycle40_subj07_prior_low_1sess_150ep",
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

REPORT_METRICS = ["BrainRet", "ImageRet", "CLIP", "InceptionV3", "VC", "HigherVis", "EffNet-B", "SwAV"]


def as_builtin(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x.tolist()
    return x


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
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return OrderedDict(
        job_status=job_status,
        active_job_count=len(lines),
        active_jobs=lines,
        squeue_returncode=result.returncode,
    )


def load_teacher(data_path, explicit_path=None):
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    candidates.extend(
        [
            os.path.join(data_path, "tables/cycle39_failure_decomposition/all_images_openclip_bigG_flat_norm.pt"),
            os.path.join(data_path, "tables/cycle38_neighbor_diagnostics/all_images_openclip_bigG_flat_norm.pt"),
        ]
    )
    for path in candidates:
        if path and os.path.exists(path):
            teacher = torch.load(path, map_location="cpu").float()
            return F.normalize(teacher, dim=-1), path
    raise FileNotFoundError("Missing existing all_images_openclip_bigG_flat_norm.pt teacher cache.")


def load_student(data_path, model_name):
    path = require(os.path.join(data_path, "evals", model_name, f"{model_name}_all_clipvoxels.pt"))
    raw = torch.load(path, map_location="cpu")
    if not torch.isfinite(raw).all():
        raise RuntimeError(f"Non-finite all_clipvoxels tensor: {path}")
    flat = raw.float().flatten(1)
    student = F.normalize(flat, dim=-1)
    del raw, flat
    return student, path


def load_final_csv(data_path, model_name):
    path = require(os.path.join(data_path, "tables", f"{model_name}_all_enhancedrecons.csv"))
    df = pd.read_csv(path, sep="\t")
    values = [float(v) for v in df["Value"].tolist()]
    if len(values) != len(CSV_METRICS):
        raise RuntimeError(f"Unexpected metric count in {path}: {len(values)}")
    return OrderedDict(zip(CSV_METRICS, values)), path


def image_ids(data_path, subject, n):
    c41_path = os.path.join(
        data_path,
        "tables/cycle41_prior_collapse_diagnostic",
        f"subj{subject:02d}_prior_low_vs_prior0_per_image.csv",
    )
    if os.path.exists(c41_path):
        df = pd.read_csv(c41_path, usecols=["nsd_image_id"])
        if len(df) == n:
            return df["nsd_image_id"].to_numpy(), c41_path
    return np.array(["NA"] * n, dtype=object), None


def ranks_from_similarity(sim):
    order = torch.argsort(sim, dim=1, descending=True)
    labels = torch.arange(sim.shape[0], device=sim.device)
    return (order == labels[:, None]).nonzero()[:, 1].float() + 1.0


def full_pool_brainret(student, teacher):
    sim = student.float() @ teacher.float().T
    rank = ranks_from_similarity(sim).numpy()
    eye = torch.eye(sim.shape[0], dtype=torch.bool)
    off = sim.masked_fill(eye, -float("inf"))
    hard_vals, hard_idx = off.max(dim=1)
    pos = sim.diag()
    return OrderedDict(
        positive_similarity=pos.numpy(),
        hardest_impostor_similarity=hard_vals.numpy(),
        hardest_impostor_index=hard_idx.numpy().astype(np.int64),
        margin=(pos - hard_vals).numpy(),
        full_pool_rank=rank,
        full_pool_top1=(rank == 1).astype(np.float32),
    )


def full_pool_imageret(student, teacher):
    sim = teacher.float() @ student.float().T
    return ranks_from_similarity(sim).numpy()


def sampled_brainret(student, teacher, seed=42, loops=30, sample_size=300):
    rng = np.random.RandomState(seed)
    n = student.shape[0]
    appearances = np.zeros(n, dtype=np.int32)
    success = np.zeros(n, dtype=np.int32)
    rank_lists = [[] for _ in range(n)]

    for _ in range(loops):
        idx = rng.choice(np.arange(n), size=sample_size, replace=False)
        sim = student[idx].float() @ teacher[idx].float().T
        rank = ranks_from_similarity(sim).numpy()
        appearances[idx] += 1
        success[idx] += (rank == 1).astype(np.int32)
        for local_i, global_i in enumerate(idx):
            rank_lists[global_i].append(float(rank[local_i]))

    denom = np.maximum(appearances, 1)
    rank_mean = np.array([np.mean(x) if x else np.nan for x in rank_lists], dtype=np.float64)
    rank_median = np.array([np.median(x) if x else np.nan for x in rank_lists], dtype=np.float64)
    return OrderedDict(
        appearances=appearances,
        sampled_top1_success_rate=success / denom,
        sampled_rank_mean=rank_mean,
        sampled_rank_median=rank_median,
        aggregate_top1=float(success.sum() / appearances.sum()),
    )


def summary_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    return OrderedDict(mean=float(np.nanmean(arr)), median=float(np.nanmedian(arr)))


def rank_summary(rank):
    arr = np.asarray(rank, dtype=np.float64)
    return OrderedDict(
        mean=float(np.nanmean(arr)),
        median=float(np.nanmedian(arr)),
        top1_fraction=float(np.nanmean(arr == 1)),
    )


def correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    x = x[mask]
    y = y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def authenticate_artifacts(data_path, subjects):
    missing = []
    artifacts = OrderedDict()
    suffixes = ["all_clipvoxels", "all_blurryrecons", "all_recons", "all_enhancedrecons"]
    for subj in subjects:
        for label, model_name in ROWS[subj].items():
            base = os.path.join(data_path, "evals", model_name)
            row = OrderedDict(eval_dir=base, eval_dir_exists=os.path.isdir(base))
            if not os.path.isdir(base):
                missing.append(base)
            for suffix in suffixes:
                path = os.path.join(base, f"{model_name}_{suffix}.pt")
                row[suffix] = OrderedDict(path=path, exists=os.path.exists(path))
                if not os.path.exists(path):
                    missing.append(path)
                else:
                    row[suffix]["size_bytes"] = int(os.path.getsize(path))
            csv_path = os.path.join(data_path, "tables", f"{model_name}_all_enhancedrecons.csv")
            row["final_csv"] = OrderedDict(path=csv_path, exists=os.path.exists(csv_path))
            if not os.path.exists(csv_path):
                missing.append(csv_path)
            artifacts[f"subj{subj:02d}_{label}"] = row
    return missing, artifacts


def load_optional_pixcorr_deltas(data_path, subject):
    path = os.path.join(
        data_path,
        "tables/cycle41_prior_collapse_diagnostic",
        f"subj{subject:02d}_prior_low_vs_prior0_per_image.csv",
    )
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    return df, path


def write_row(args, subject, label, model_name, teacher, ids, ids_source):
    csv_metrics, csv_path = load_final_csv(args.data_path, model_name)
    print(f"rank-margin diagnostics: loading student {model_name}", flush=True)
    student, clipvoxel_path = load_student(args.data_path, model_name)
    print(f"rank-margin diagnostics: full-pool BrainRet {model_name}", flush=True)
    full = full_pool_brainret(student, teacher)
    print(f"rank-margin diagnostics: sampled BrainRet {model_name}", flush=True)
    sampled = sampled_brainret(student, teacher, args.seed, args.eval_loops, args.eval_sample_size)
    print(f"rank-margin diagnostics: full-pool ImageRet {model_name}", flush=True)
    image_rank = full_pool_imageret(student, teacher)

    df = pd.DataFrame(
        OrderedDict(
            subject=subject,
            model_name=model_name,
            eval_index=np.arange(student.shape[0]),
            image_id=ids,
            positive_similarity=full["positive_similarity"],
            hardest_impostor_similarity=full["hardest_impostor_similarity"],
            hardest_impostor_index=full["hardest_impostor_index"],
            margin=full["margin"],
            full_pool_rank=full["full_pool_rank"],
            sampled_top1_success_rate=sampled["sampled_top1_success_rate"],
            sampled_rank_mean=sampled["sampled_rank_mean"],
            sampled_rank_median=sampled["sampled_rank_median"],
        )
    )
    per_model_path = os.path.join(args.outdir, f"{model_name}_rank_margin.csv")
    df.to_csv(per_model_path, index=False)

    summary = OrderedDict(
        subject=subject,
        row_label=label,
        model_name=model_name,
        per_image_csv=per_model_path,
        provenance=OrderedDict(
            clipvoxel_path=clipvoxel_path,
            final_csv_path=csv_path,
            image_id_source=ids_source or "NA",
        ),
        final_csv_metric_values=OrderedDict((k, csv_metrics[k]) for k in REPORT_METRICS),
        brainret_rank_summary=rank_summary(full["full_pool_rank"]),
        imageret_rank_summary=rank_summary(image_rank),
        full_pool_rank_mean=float(np.mean(full["full_pool_rank"])),
        full_pool_rank_median=float(np.median(full["full_pool_rank"])),
        full_pool_top1_fraction=float(np.mean(full["full_pool_rank"] == 1)),
        sampled_rank_mean=float(np.nanmean(sampled["sampled_rank_mean"])),
        sampled_rank_median=float(np.nanmedian(sampled["sampled_rank_median"])),
        sampled_top1_success_mean=float(np.nanmean(sampled["sampled_top1_success_rate"])),
        sampled_aggregate_top1=sampled["aggregate_top1"],
        margin_mean=float(np.mean(full["margin"])),
        margin_median=float(np.median(full["margin"])),
    )
    del student
    return df, summary


def summarize_delta(subject, control_df, intervention_df, pixcorr_df):
    delta = intervention_df.copy()
    delta["delta_positive_similarity"] = intervention_df["positive_similarity"] - control_df["positive_similarity"]
    delta["delta_hardest_impostor_similarity"] = (
        intervention_df["hardest_impostor_similarity"] - control_df["hardest_impostor_similarity"]
    )
    delta["delta_margin"] = intervention_df["margin"] - control_df["margin"]
    delta["delta_full_pool_rank"] = intervention_df["full_pool_rank"] - control_df["full_pool_rank"]
    delta["delta_sampled_top1_success_rate"] = (
        intervention_df["sampled_top1_success_rate"] - control_df["sampled_top1_success_rate"]
    )
    delta["rank1_to_not1"] = (control_df["full_pool_rank"].eq(1) & intervention_df["full_pool_rank"].ne(1)).astype(int)
    delta["not1_to_rank1"] = (control_df["full_pool_rank"].ne(1) & intervention_df["full_pool_rank"].eq(1)).astype(int)

    cols = [
        "subject",
        "model_name",
        "eval_index",
        "image_id",
        "positive_similarity",
        "hardest_impostor_similarity",
        "hardest_impostor_index",
        "margin",
        "full_pool_rank",
        "sampled_top1_success_rate",
        "sampled_rank_mean",
        "sampled_rank_median",
        "delta_positive_similarity",
        "delta_hardest_impostor_similarity",
        "delta_margin",
        "delta_full_pool_rank",
        "delta_sampled_top1_success_rate",
        "rank1_to_not1",
        "not1_to_rank1",
    ]
    corr = OrderedDict(
        sampled_success_delta_vs_margin_delta=correlation(
            delta["delta_sampled_top1_success_rate"], delta["delta_margin"]
        ),
        sampled_success_delta_vs_full_pool_rank_delta=correlation(
            delta["delta_sampled_top1_success_rate"], delta["delta_full_pool_rank"]
        ),
    )
    if pixcorr_df is not None and "enhanced_evalmix_pixcorr_delta_prior_low_minus_prior0" in pixcorr_df:
        corr["sampled_success_delta_vs_enhanced_evalmix_pixcorr_delta"] = correlation(
            delta["delta_sampled_top1_success_rate"],
            pixcorr_df["enhanced_evalmix_pixcorr_delta_prior_low_minus_prior0"],
        )

    summary = OrderedDict(
        subject=subject,
        row_label="prior_low_vs_prior0",
        row_count=int(len(delta)),
        rank1_to_not1_count=int(delta["rank1_to_not1"].sum()),
        rank1_to_not1_fraction=float(delta["rank1_to_not1"].mean()),
        not1_to_rank1_count=int(delta["not1_to_rank1"].sum()),
        not1_to_rank1_fraction=float(delta["not1_to_rank1"].mean()),
        delta_positive_similarity=summary_stats(delta["delta_positive_similarity"]),
        delta_hardest_impostor_similarity=summary_stats(delta["delta_hardest_impostor_similarity"]),
        delta_margin=summary_stats(delta["delta_margin"]),
        delta_full_pool_rank=summary_stats(delta["delta_full_pool_rank"]),
        delta_sampled_top1_success_rate=summary_stats(delta["delta_sampled_top1_success_rate"]),
        correlations=corr,
    )
    return delta[cols], summary


def main():
    parser = argparse.ArgumentParser(
        description="Read-only evaluator-side rank/margin diagnostics for saved all_clipvoxels artifacts."
    )
    parser.add_argument("--data_path", default="/src")
    parser.add_argument("--outdir", default="/src/tables/cycle43_rank_margin_diagnostics")
    parser.add_argument("--teacher_cache", default=None)
    parser.add_argument("--subjects", nargs="+", type=int, default=[5, 7], choices=[5, 7])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_loops", type=int, default=30)
    parser.add_argument("--eval_sample_size", type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sched = scheduler_state()
    if sched["active_job_count"]:
        raise RuntimeError(f"Active jobs found: {sched['active_jobs']}")
    missing, artifacts = authenticate_artifacts(args.data_path, args.subjects)
    if missing:
        raise FileNotFoundError(json.dumps(missing, indent=2))

    teacher, teacher_path = load_teacher(args.data_path, args.teacher_cache)
    subject_summaries = []
    replay_checks = []
    delta_paths = OrderedDict()

    for subject in args.subjects:
        ids, ids_source = image_ids(args.data_path, subject, teacher.shape[0])
        row_dfs = OrderedDict()
        row_summaries = OrderedDict()
        for label, model_name in ROWS[subject].items():
            row_df, row_summary = write_row(args, subject, label, model_name, teacher, ids, ids_source)
            row_dfs[label] = row_df
            row_summaries[label] = row_summary

        pixcorr_df, pixcorr_path = load_optional_pixcorr_deltas(args.data_path, subject)
        delta_df, delta_summary = summarize_delta(subject, row_dfs["prior0"], row_dfs["prior_low"], pixcorr_df)
        delta_path = os.path.join(args.outdir, f"subj{subject:02d}_prior_low_vs_prior0_rank_margin_delta.csv")
        delta_df.to_csv(delta_path, index=False)
        delta_paths[f"subj{subject:02d}"] = delta_path

        csv_delta = (
            row_summaries["prior_low"]["final_csv_metric_values"]["BrainRet"]
            - row_summaries["prior0"]["final_csv_metric_values"]["BrainRet"]
        )
        replay_delta = (
            row_summaries["prior_low"]["sampled_aggregate_top1"]
            - row_summaries["prior0"]["sampled_aggregate_top1"]
        )
        replay_checks.append(
            OrderedDict(
                subject=subject,
                csv_brainret_delta=float(csv_delta),
                replayed_sampled_top1_delta=float(replay_delta),
                replay_minus_csv_delta=float(replay_delta - csv_delta),
                within_0p002_tolerance=bool(abs(replay_delta - csv_delta) <= 0.002),
            )
        )

        subject_summaries.append(
            OrderedDict(
                subject=subject,
                rows=row_summaries,
                control_delta=delta_summary,
                control_delta_csv=delta_path,
                optional_pixcorr_delta_source=pixcorr_path or "not available",
                roi_category_diagnostics=OrderedDict(
                    available=False,
                    reason=(
                        "Saved Cycle 40 artifacts expose aggregate final_evaluations.py ROI correlations only; "
                        "no per-image ROI/category tensor is available without rerunning or rewriting GNet inference."
                    ),
                ),
            )
        )

    out = OrderedDict(
        provenance=OrderedDict(
            cycle=43,
            scope="read-only evaluator-side rank/margin diagnostics",
            data_path=args.data_path,
            outdir=args.outdir,
            teacher_cache=teacher_path,
            seed=args.seed,
            eval_loops=args.eval_loops,
            eval_sample_size=args.eval_sample_size,
            no_training_or_model_behavior_change=True,
        ),
        preflight=OrderedDict(scheduler=sched, missing=missing, artifacts=artifacts),
        evaluator_path_readout=OrderedDict(
            protected_brainret_source="all_clipvoxels vs true image OpenCLIP embeddings using final_evaluations.py 30x300 sampled top-1 protocol",
            final_csv_schema_changed=False,
            implementation="companion script; final_evaluations.py was not modified",
        ),
        replay_checks=replay_checks,
        delta_csvs=delta_paths,
        subjects=subject_summaries,
    )
    json_path = os.path.join(args.outdir, "cycle43_rank_margin_summary.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=as_builtin)
    print(json.dumps(out, indent=2, default=as_builtin)[:30000])
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
