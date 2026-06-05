import argparse
import glob
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

REFERENCE_CYCLE41 = {
    5: {
        "image_rank_delta_mean": 142.922,
        "brain_rank_delta_mean": -1.580,
        "effective_rank_prior0": 47.662,
        "effective_rank_prior_low": 41.410,
        "self_offdiag_prior0": 0.4800,
        "self_offdiag_prior_low": 0.8976,
    },
    7: {
        "image_rank_delta_mean": 148.302,
        "brain_rank_delta_mean": 2.166,
        "effective_rank_prior0": 46.638,
        "effective_rank_prior_low": 41.094,
        "self_offdiag_prior0": 0.5138,
        "self_offdiag_prior_low": 0.9128,
    },
}


def as_builtin(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x.tolist()
    return x


def percentile_summary(values):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    pct = np.percentile(arr, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    return OrderedDict(
        mean=float(arr.mean()),
        std=float(arr.std()),
        min=float(pct[0]),
        p01=float(pct[1]),
        p05=float(pct[2]),
        p25=float(pct[3]),
        median=float(pct[4]),
        p75=float(pct[5]),
        p95=float(pct[6]),
        p99=float(pct[7]),
        max=float(pct[8]),
    )


def summarize_signed_delta(values):
    arr = np.asarray(values, dtype=np.float64)
    return OrderedDict(
        mean=float(arr.mean()),
        median=float(np.median(arr)),
        worse_fraction=float(np.mean(arr > 0)),
        improved_fraction=float(np.mean(arr < 0)),
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


def require(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


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


def load_final_csv(data_path, model_name):
    path = require(os.path.join(data_path, "tables", f"{model_name}_all_enhancedrecons.csv"))
    df = pd.read_csv(path, sep="\t")
    values = [float(v) for v in df["Value"].tolist()]
    if len(values) != len(CSV_METRICS):
        raise RuntimeError(f"Unexpected metric count in {path}: {len(values)}")
    return OrderedDict(zip(CSV_METRICS, values)), path


def load_student(data_path, model_name):
    path = require(os.path.join(data_path, "evals", model_name, f"{model_name}_all_clipvoxels.pt"))
    raw = torch.load(path, map_location="cpu")
    flat = raw.float().flatten(1)
    norm = flat.norm(dim=1).numpy()
    student = F.normalize(flat, dim=-1)
    del raw, flat
    return student, norm, path


def ranks_from_similarity(sim):
    order = torch.argsort(sim, dim=1, descending=True)
    labels = torch.arange(sim.shape[0], device=sim.device)
    return (order == labels[:, None]).nonzero()[:, 1].float() + 1.0


def full_pool_geometry(student, teacher):
    sim = student.float() @ teacher.float().T
    rank = ranks_from_similarity(sim)
    eye = torch.eye(sim.shape[0], dtype=torch.bool)
    off = sim.masked_fill(eye, -float("inf"))
    hard = off.max(dim=1).values
    top5 = off.topk(5, dim=1).values.mean(dim=1)
    mean_imp = sim.masked_fill(eye, 0).sum(dim=1) / (sim.shape[1] - 1)
    pos = sim.diag()
    return OrderedDict(
        rank=rank.numpy(),
        positive_similarity=pos.numpy(),
        mean_impostor_similarity=mean_imp.numpy(),
        hardest_impostor_similarity=hard.numpy(),
        positive_minus_mean_impostor=(pos - mean_imp).numpy(),
        positive_minus_hardest_impostor=(pos - hard).numpy(),
        positive_minus_top5_impostor=(pos - top5).numpy(),
        top1_success=(rank.numpy() == 1).astype(np.float32),
    )


def sampled_eval_geometry(student, teacher, seed=42, loops=30, sample_size=300):
    rng = np.random.RandomState(seed)
    n = student.shape[0]
    appearances = np.zeros(n, dtype=np.int32)
    success = np.zeros(n, dtype=np.int32)
    pos_sum = np.zeros(n, dtype=np.float64)
    mean_imp_sum = np.zeros(n, dtype=np.float64)
    hard_imp_sum = np.zeros(n, dtype=np.float64)
    margin_mean_sum = np.zeros(n, dtype=np.float64)
    margin_hard_sum = np.zeros(n, dtype=np.float64)
    rank_sum = np.zeros(n, dtype=np.float64)

    for _ in range(loops):
        idx = rng.choice(np.arange(n), size=sample_size, replace=False)
        sim = student[idx].float() @ teacher[idx].float().T
        rank = ranks_from_similarity(sim).numpy()
        eye = torch.eye(sample_size, dtype=torch.bool)
        pos = sim.diag().numpy()
        sim_np = sim.numpy()
        off_np = np.where(eye.numpy(), np.nan, sim_np)
        mean_imp = np.nanmean(off_np, axis=1)
        hard_imp = np.nanmax(off_np, axis=1)

        appearances[idx] += 1
        success[idx] += (rank == 1).astype(np.int32)
        pos_sum[idx] += pos
        mean_imp_sum[idx] += mean_imp
        hard_imp_sum[idx] += hard_imp
        margin_mean_sum[idx] += pos - mean_imp
        margin_hard_sum[idx] += pos - hard_imp
        rank_sum[idx] += rank

    denom = np.maximum(appearances, 1)
    return OrderedDict(
        appearances=appearances,
        success_count=success,
        success_rate=success / denom,
        sampled_rank_mean=rank_sum / denom,
        sampled_positive_similarity=pos_sum / denom,
        sampled_mean_impostor_similarity=mean_imp_sum / denom,
        sampled_hardest_impostor_similarity=hard_imp_sum / denom,
        sampled_positive_minus_mean_impostor=margin_mean_sum / denom,
        sampled_positive_minus_hardest_impostor=margin_hard_sum / denom,
        aggregate_top1=float(success.sum() / appearances.sum()),
        expected_csv_value=float(success.sum() / appearances.sum()),
    )


def load_cycle41_summary(data_path):
    path = require(os.path.join(data_path, "tables/cycle41_prior_collapse_diagnostic/cycle41_prior_collapse_summary.json"))
    with open(path) as f:
        data = json.load(f)
    by_subject = {int(s["subject"]): s for s in data["subjects"]}
    return by_subject, path


def authenticate_artifacts(data_path, subjects):
    missing = []
    artifacts = OrderedDict()
    suffixes = ["all_clipvoxels", "all_blurryrecons", "all_recons", "all_enhancedrecons"]
    for subj in subjects:
        for label, model_name in ROWS[subj].items():
            row = OrderedDict()
            base = os.path.join(data_path, "evals", model_name)
            row["eval_dir"] = base
            row["eval_dir_exists"] = os.path.isdir(base)
            if not os.path.isdir(base):
                missing.append(base)
            for suffix in suffixes:
                path = os.path.join(base, f"{model_name}_{suffix}.pt")
                row[suffix] = OrderedDict(path=path, exists=os.path.exists(path))
                if not os.path.exists(path):
                    missing.append(path)
                else:
                    row[suffix].update(size_bytes=int(os.path.getsize(path)))
            csv_path = os.path.join(data_path, "tables", f"{model_name}_all_enhancedrecons.csv")
            row["final_csv"] = OrderedDict(path=csv_path, exists=os.path.exists(csv_path))
            if not os.path.exists(csv_path):
                missing.append(csv_path)
            artifacts[f"subj{subj:02d}_{label}"] = row
    return missing, artifacts


def evaluator_log_evidence(model_name):
    text = ""
    files = sorted(glob.glob("/src/slurms/c40_prior_eval_s57_9239770_*.out"))
    for path in files:
        with open(path, errors="replace") as f:
            body = f.read()
        if model_name in body:
            for line in body.splitlines():
                if "all_recons_path:" in line or model_name in line:
                    text += line + "\n"
    expected = f"evals/{model_name}/{model_name}_all_enhancedrecons.pt"
    return OrderedDict(
        expected_all_recons_path=expected,
        found_expected_path=expected in text,
        matching_lines=text.strip().splitlines()[:20],
    )


def load_cycle41_per_image(data_path, subj):
    path = require(
        os.path.join(
            data_path,
            "tables/cycle41_prior_collapse_diagnostic",
            f"subj{subj:02d}_prior_low_vs_prior0_per_image.csv",
        )
    )
    return pd.read_csv(path), path


def check_cycle41_anchor(subj, c41_df, c41_subject):
    ref = REFERENCE_CYCLE41[subj]
    cal0 = c41_subject["feature_calibration"]["prior0"]
    cal_low = c41_subject["feature_calibration"]["prior_low"]
    observed = OrderedDict(
        image_rank_delta_mean=float(c41_df["image_rank_delta_prior_low_minus_prior0"].mean()),
        brain_rank_delta_mean=float(c41_df["brain_rank_delta_prior_low_minus_prior0"].mean()),
        effective_rank_prior0=cal0["effective_rank"],
        effective_rank_prior_low=cal_low["effective_rank"],
        self_offdiag_prior0=cal0["student_self_offdiag_cosine"]["mean"],
        self_offdiag_prior_low=cal_low["student_self_offdiag_cosine"]["mean"],
    )
    checks = OrderedDict()
    for key, val in observed.items():
        tol = 0.25 if "rank_delta_mean" in key else 0.01
        checks[key] = OrderedDict(
            observed=float(val),
            reference=float(ref[key]),
            abs_diff=float(abs(val - ref[key])),
            tolerance=tol,
            pass_=bool(abs(val - ref[key]) <= tol),
        )
    if not all(v["pass_"] for v in checks.values()):
        raise RuntimeError(f"Cycle 41 anchor disagreement for subject {subj}: {checks}")
    return checks


def summarize_subject(args, subj, teacher, c41_subject):
    prior0 = ROWS[subj]["prior0"]
    prior_low = ROWS[subj]["prior_low"]
    csv0, csv0_path = load_final_csv(args.data_path, prior0)
    csvlow, csvlow_path = load_final_csv(args.data_path, prior_low)
    c41_df, c41_path = load_cycle41_per_image(args.data_path, subj)

    student0, norm0, student0_path = load_student(args.data_path, prior0)
    student_low, norm_low, student_low_path = load_student(args.data_path, prior_low)
    cal0 = c41_subject["feature_calibration"]["prior0"]
    cal_low = c41_subject["feature_calibration"]["prior_low"]
    anchor_checks = check_cycle41_anchor(subj, c41_df, c41_subject)

    full0 = full_pool_geometry(student0, teacher)
    full_low = full_pool_geometry(student_low, teacher)
    samp0 = sampled_eval_geometry(student0, teacher, args.seed, args.eval_loops, args.eval_sample_size)
    samp_low = sampled_eval_geometry(student_low, teacher, args.seed, args.eval_loops, args.eval_sample_size)

    out_df = c41_df.copy()
    prefix_pairs = [
        ("final_fullpool", full0, full_low),
        ("final_sampled300", samp0, samp_low),
    ]
    for prefix, z, low in prefix_pairs:
        for key in z:
            if key == "aggregate_top1" or key == "expected_csv_value":
                continue
            out_df[f"prior0_{prefix}_{key}"] = z[key]
            out_df[f"prior_low_{prefix}_{key}"] = low[key]
            out_df[f"{prefix}_{key}_delta_prior_low_minus_prior0"] = np.asarray(low[key]) - np.asarray(z[key])

    out_df["final_feature_norm_delta_prior_low_minus_prior0"] = norm_low - norm0
    out_df["final_brainret_success_rate_delta_prior_low_minus_prior0"] = (
        out_df["prior_low_final_sampled300_success_rate"] - out_df["prior0_final_sampled300_success_rate"]
    )

    corr_cols = OrderedDict()
    target = "final_brainret_success_rate_delta_prior_low_minus_prior0"
    predictors = [
        "image_rank_delta_prior_low_minus_prior0",
        "brain_rank_delta_prior_low_minus_prior0",
        "flat_norm_delta_prior_low_minus_prior0",
        "brain_margin_hardest_delta_prior_low_minus_prior0",
        "image_margin_hardest_delta_prior_low_minus_prior0",
        "blurry_pixcorr_delta_prior_low_minus_prior0",
        "recons_pixcorr_delta_prior_low_minus_prior0",
        "enhanced_pixcorr_delta_prior_low_minus_prior0",
        "enhanced_evalmix_pixcorr_delta_prior_low_minus_prior0",
        "final_fullpool_positive_minus_hardest_impostor_delta_prior_low_minus_prior0",
        "final_sampled300_sampled_positive_minus_hardest_impostor_delta_prior_low_minus_prior0",
        "final_feature_norm_delta_prior_low_minus_prior0",
    ]
    for col in predictors:
        corr_cols[col] = correlation(out_df[col], out_df[target]) if col in out_df else None

    rank_target = "final_sampled300_sampled_rank_mean_delta_prior_low_minus_prior0"
    rank_correlations = OrderedDict(
        (col, correlation(out_df[col], out_df[rank_target])) for col in predictors if col in out_df
    )

    worst = out_df.sort_values(target, ascending=True).head(args.topn)
    per_image_path = os.path.join(args.outdir, f"subj{subj:02d}_brainret_amplification_per_image.csv")
    out_df.to_csv(per_image_path, index=False)

    final_delta = OrderedDict((k, csvlow[k] - csv0[k]) for k in CSV_METRICS)
    csv_brainret_delta = final_delta["BrainRet"]
    replay_brainret_delta = samp_low["aggregate_top1"] - samp0["aggregate_top1"]

    return OrderedDict(
        subject=subj,
        rows=ROWS[subj],
        per_image_csv=per_image_path,
        cycle41_per_image_csv=c41_path,
        final_csv_paths=OrderedDict(prior0=csv0_path, prior_low=csvlow_path),
        clipvoxel_paths=OrderedDict(prior0=student0_path, prior_low=student_low_path),
        evaluator_log_evidence=OrderedDict(
            prior0=evaluator_log_evidence(prior0),
            prior_low=evaluator_log_evidence(prior_low),
        ),
        cycle41_anchor_checks=anchor_checks,
        final_csv_prior_low_minus_prior0=final_delta,
        final_brainret_replay=OrderedDict(
            prior0_sampled_top1=samp0["aggregate_top1"],
            prior_low_sampled_top1=samp_low["aggregate_top1"],
            sampled_top1_delta_prior_low_minus_prior0=replay_brainret_delta,
            csv_brainret_prior0=csv0["BrainRet"],
            csv_brainret_prior_low=csvlow["BrainRet"],
            csv_brainret_delta_prior_low_minus_prior0=csv_brainret_delta,
            replay_minus_csv_delta=float(replay_brainret_delta - csv_brainret_delta),
            note=(
                "final_evaluations.py computes BrainRet from all_clipvoxels versus true image OpenCLIP "
                "inside repeated 300-way top-1 samples; it is not computed from enhanced reconstruction pixels."
            ),
        ),
        final_geometry_summaries=OrderedDict(
            fullpool_brain_rank_delta=summarize_signed_delta(
                out_df["final_fullpool_rank_delta_prior_low_minus_prior0"]
            ),
            sampled300_brain_rank_delta=summarize_signed_delta(
                out_df["final_sampled300_sampled_rank_mean_delta_prior_low_minus_prior0"]
            ),
            sampled300_success_rate_delta=percentile_summary(out_df[target]),
            fullpool_margin_hardest_delta=percentile_summary(
                out_df["final_fullpool_positive_minus_hardest_impostor_delta_prior_low_minus_prior0"]
            ),
            sampled300_margin_hardest_delta=percentile_summary(
                out_df["final_sampled300_sampled_positive_minus_hardest_impostor_delta_prior_low_minus_prior0"]
            ),
        ),
        feature_calibration=OrderedDict(prior0=cal0, prior_low=cal_low),
        correlations_with_final_brainret_success_delta=corr_cols,
        correlations_with_final_sampled_rank_delta=rank_correlations,
        worst_final_brainret_regressions=worst[
            [
                "subject",
                "eval_index",
                "nsd_image_id",
                "prior0_final_sampled300_success_rate",
                "prior_low_final_sampled300_success_rate",
                target,
                "prior0_final_fullpool_rank",
                "prior_low_final_fullpool_rank",
                "final_fullpool_rank_delta_prior_low_minus_prior0",
                "brain_rank_delta_prior_low_minus_prior0",
                "image_rank_delta_prior_low_minus_prior0",
                "final_sampled300_sampled_positive_minus_hardest_impostor_delta_prior_low_minus_prior0",
                "enhanced_evalmix_pixcorr_delta_prior_low_minus_prior0",
            ]
        ].to_dict(orient="records"),
    )


def classify(subject_results):
    out = OrderedDict()
    for subj in subject_results:
        direct_brain_delta = subj["cycle41_anchor_checks"]["brain_rank_delta_mean"]["observed"]
        direct_image_delta = subj["cycle41_anchor_checks"]["image_rank_delta_mean"]["observed"]
        final_success_delta = subj["final_brainret_replay"]["sampled_top1_delta_prior_low_minus_prior0"]
        margin_delta = subj["final_geometry_summaries"]["sampled300_margin_hardest_delta"]["mean"]
        if final_success_delta < -0.25 and abs(direct_brain_delta) < 10 and direct_image_delta > 100:
            label = "evaluator top-1 amplification of upstream predicted-CLIP separability collapse"
        elif final_success_delta < -0.25 and margin_delta < 0:
            label = "mixed predicted-CLIP margin collapse and evaluator amplification"
        elif final_success_delta < -0.25:
            label = "downstream/evaluator amplification unresolved"
        else:
            label = "no large final BrainRet amplification detected"
        out[f"subj{subj['subject']:02d}"] = OrderedDict(
            label=label,
            direct_brain_rank_mean_delta=direct_brain_delta,
            direct_image_rank_mean_delta=direct_image_delta,
            sampled_brainret_delta=final_success_delta,
            sampled_margin_hardest_delta_mean=margin_delta,
            rationale=(
                "The protected BrainRet gate is a repeated subset top-1 metric. Small full-pool rank "
                "mean movement can cause a large score drop when many images move from rank 1 to rank 2+."
            ),
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/src")
    parser.add_argument("--outdir", default="/src/tables/cycle42_brainret_amplification_diagnostic")
    parser.add_argument("--teacher_cache", default=None)
    parser.add_argument("--subjects", nargs="+", type=int, default=[5, 7], choices=[5, 7])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_loops", type=int, default=30)
    parser.add_argument("--eval_sample_size", type=int, default=300)
    parser.add_argument("--topn", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sched = scheduler_state()
    if sched["active_job_count"]:
        raise RuntimeError(f"Active jobs found: {sched['active_jobs']}")
    missing, artifacts = authenticate_artifacts(args.data_path, args.subjects)
    if missing:
        raise FileNotFoundError(json.dumps(missing, indent=2))
    teacher, teacher_path = load_teacher(args.data_path, args.teacher_cache)
    c41_by_subject, c41_summary_path = load_cycle41_summary(args.data_path)
    subjects = [summarize_subject(args, subj, teacher, c41_by_subject[subj]) for subj in args.subjects]
    out = OrderedDict(
        provenance=OrderedDict(
            cycle=42,
            scope="artifact-only downstream BrainRet amplification diagnostic",
            data_path=args.data_path,
            outdir=args.outdir,
            teacher_cache=teacher_path,
            cycle41_summary=c41_summary_path,
            commands="python /src/cycle42_brainret_amplification_diagnostic.py --data_path=/src",
            no_training_or_new_evaluation_branch=True,
        ),
        preflight=OrderedDict(scheduler=sched, missing=missing, artifacts=artifacts),
        evaluator_path_readout=OrderedDict(
            protected_brainret_source="all_clipvoxels vs evals/all_images.pt OpenCLIP teacher cache",
            enhanced_reconstruction_brainret_source=False,
            explanation=(
                "In /src/final_evaluations.py, BrainRet is the BwdRetrieval value computed before pixel "
                "and GNet reconstruction metrics, using saved predicted CLIP voxels. all_recons_path is "
                "authenticated for the run but does not feed the BrainRet calculation."
            ),
        ),
        roi_feasibility=OrderedDict(
            per_image_roi_available=False,
            reason=(
                "Cycle 40 saved only aggregate GNet ROI correlations. Per-image ROI tensors are not present, "
                "and extensive evaluator instrumentation is outside this artifact-only cycle."
            ),
        ),
        subjects=subjects,
        classification=classify(subjects),
    )
    json_path = os.path.join(args.outdir, "cycle42_brainret_amplification_summary.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=as_builtin)
    print(json.dumps(out, indent=2, default=as_builtin)[:30000])
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
