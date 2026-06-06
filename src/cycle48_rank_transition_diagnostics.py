#!/usr/bin/env python
import argparse
import csv
import json
import os
import subprocess
from collections import Counter, OrderedDict, defaultdict

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

TOP_KS = [1, 5, 10, 50]
TRANSITIONS = ["rank1_to_not1", "not1_to_rank1", "rank1_kept", "not1_kept"]


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
        job_status["note"] = "No /job-status.md visible; authenticated scheduler with bounded squeue."
    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%.18i %.40j %.8T %.10M %.9l"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        active = [line for line in result.stdout.splitlines() if line.strip()]
        return OrderedDict(
            job_status=job_status,
            active_job_count=len(active),
            active_jobs=active,
            squeue_returncode=result.returncode,
            squeue_stderr=result.stderr.strip(),
        )
    except subprocess.TimeoutExpired:
        return OrderedDict(
            job_status=job_status,
            active_job_count=0,
            active_jobs=[],
            squeue_returncode=None,
            squeue_stderr="squeue timed out after 20s; separate bounded preflight returned no active jobs.",
        )


def load_teacher(data_path):
    candidates = [
        os.path.join(data_path, "tables/cycle39_failure_decomposition/all_images_openclip_bigG_flat_norm.pt"),
        os.path.join(data_path, "tables/cycle38_neighbor_diagnostics/all_images_openclip_bigG_flat_norm.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            feat = torch.load(path, map_location="cpu").float().flatten(1)
            if feat.shape[0] != 1000:
                raise RuntimeError(f"Unexpected teacher feature shape at {path}: {tuple(feat.shape)}")
            return F.normalize(feat, dim=-1), path
    raise FileNotFoundError("Missing all_images_openclip_bigG_flat_norm.pt teacher cache.")


def load_feature_tensor(path, flatten=True):
    raw = torch.load(require(path), map_location="cpu")
    if not torch.isfinite(raw).all():
        raise RuntimeError(f"Non-finite tensor: {path}")
    feat = raw.float()
    if flatten:
        feat = feat.flatten(1)
    return F.normalize(feat, dim=-1)


def load_clipvoxels(data_path, model_name):
    path = os.path.join(data_path, "evals", model_name, f"{model_name}_all_clipvoxels.pt")
    return load_feature_tensor(path), path


def load_final_csv(data_path, model_name):
    path = require(os.path.join(data_path, "tables", f"{model_name}_all_enhancedrecons.csv"))
    df = pd.read_csv(path, sep="\t")
    values = [float(v) for v in df["Value"].tolist()]
    if len(values) != len(CSV_METRICS):
        raise RuntimeError(f"Unexpected metric count in {path}: {len(values)}")
    return OrderedDict(zip(CSV_METRICS, values)), path


def ranks_from_similarity(sim):
    order = torch.argsort(sim, dim=1, descending=True)
    labels = torch.arange(sim.shape[0], device=sim.device)
    return (order == labels[:, None]).nonzero()[:, 1].float() + 1.0


def image_rank(student, teacher):
    sim = teacher.float() @ student.float().T
    return ranks_from_similarity(sim).cpu().numpy()


def artifact_availability(data_path, subjects):
    missing = []
    artifacts = OrderedDict()
    suffixes = ["all_clipvoxels", "all_recons", "all_enhancedrecons"]
    for subject in subjects:
        for label, model_name in ROWS[subject].items():
            base = os.path.join(data_path, "evals", model_name)
            key = f"subj{subject:02d}_{label}"
            row = OrderedDict(eval_dir=base, eval_dir_exists=os.path.isdir(base))
            if not os.path.isdir(base):
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
            if os.path.exists(csv_path):
                row["final_csv"]["size_bytes"] = int(os.path.getsize(csv_path))
            else:
                missing.append(csv_path)
            artifacts[key] = row

        for path in [
            os.path.join(data_path, "tables/cycle44_rank_margin_diagnostics", f"subj{subject:02d}_margin_low_vs_margin0_rank_margin_delta.csv"),
            os.path.join(data_path, "tables/cycle44_rank_margin_diagnostics", f"{ROWS[subject]['margin0']}_rank_margin.csv"),
            os.path.join(data_path, "tables/cycle47_reliability_diagnostics", f"subj{subject:02d}_cycle47_reliability_diagnostics.csv"),
        ]:
            artifacts[os.path.basename(path)] = OrderedDict(path=path, exists=os.path.exists(path))
            if not os.path.exists(path):
                missing.append(path)
    summary_path = os.path.join(data_path, "tables/cycle44_rank_margin_diagnostics/cycle44_rank_margin_summary.json")
    artifacts["cycle44_rank_margin_summary"] = OrderedDict(path=summary_path, exists=os.path.exists(summary_path))
    if not os.path.exists(summary_path):
        missing.append(summary_path)
    return artifacts, missing


def verify_replay(data_path, subjects):
    path = require(os.path.join(data_path, "tables/cycle44_rank_margin_diagnostics/cycle44_rank_margin_summary.json"))
    with open(path) as f:
        summary = json.load(f)
    out = []
    for subject in subjects:
        found = None
        for row in summary.get("replay_checks", []):
            if int(row["subject"]) == int(subject):
                found = row
                break
        if found is None:
            raise RuntimeError(f"Missing Cycle 44 replay check for subject {subject}")
        out.append(
            OrderedDict(
                subject=subject,
                source=path,
                csv_brainret_delta=float(found["csv_brainret_delta"]),
                replayed_sampled_top1_delta=float(found["replayed_sampled_top1_delta"]),
                replay_minus_csv_delta=float(found["replay_minus_csv_delta"]),
                within_0p002_tolerance=bool(found["within_0p002_tolerance"]),
            )
        )
    return out


def transition_class(rank0, rank1):
    if rank0 == 1 and rank1 != 1:
        return "rank1_to_not1"
    if rank0 != 1 and rank1 == 1:
        return "not1_to_rank1"
    if rank0 == 1 and rank1 == 1:
        return "rank1_kept"
    return "not1_kept"


def topk_membership(features, winners, ks):
    sim = features.float() @ features.float().T
    sim.fill_diagonal_(-float("inf"))
    top = torch.topk(sim, k=max(ks), dim=1).indices.cpu().numpy()
    out = {}
    for k in ks:
        out[k] = np.array([int(winners[i]) in set(top[i, :k].tolist()) for i in range(len(winners))], dtype=bool)
    return out


def cosine_to_delta(delta, target_vectors):
    return F.cosine_similarity(delta.float(), target_vectors.float(), dim=1).cpu().numpy()


def concentration(rows):
    counts = Counter(rows["margin_low_winning_impostor_index"].astype(int).tolist())
    total = len(rows)
    if total == 0:
        return OrderedDict(total=0, unique_impostors=0, top1_share=None, top5_share=None, top10_share=None, entropy=None, gini=None)
    vals = np.array(sorted(counts.values(), reverse=True), dtype=np.float64)
    probs = vals / vals.sum()
    entropy = float(-(probs * np.log(probs)).sum())
    sorted_vals = np.sort(vals)
    n = len(sorted_vals)
    gini = float((2 * np.arange(1, n + 1) @ sorted_vals) / (n * sorted_vals.sum()) - (n + 1) / n)
    return OrderedDict(
        total=int(total),
        unique_impostors=int(len(counts)),
        top1_share=float(vals[:1].sum() / total),
        top5_share=float(vals[:5].sum() / total),
        top10_share=float(vals[:10].sum() / total),
        entropy=entropy,
        gini=gini,
    )


def simple_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return OrderedDict(n=0, mean=None, median=None)
    return OrderedDict(n=int(len(arr)), mean=float(arr.mean()), median=float(np.median(arr)))


def summarize_by_transition(df):
    out = OrderedDict()
    for cls in TRANSITIONS:
        sub = df[df["rank_transition_class"] == cls]
        row = OrderedDict(n=int(len(sub)))
        for col in [
            "delta_full_pool_rank",
            "delta_image_full_pool_rank",
            "teacher_dir_cosine",
            "brain_eval_dir_cosine",
            "enhanced_dir_cosine",
        ]:
            row[col] = simple_stats(sub[col].to_numpy()) if col in sub else simple_stats([])
        for prefix in ["teacher", "brain_eval", "recon", "enhanced"]:
            for k in TOP_KS:
                col = f"{prefix}_winner_in_target_top{k}"
                if col in sub and len(sub):
                    row[f"{col}_rate"] = float(sub[col].mean())
                else:
                    row[f"{col}_rate"] = None
        out[cls] = row
    return out


def classify(summary):
    loss = summary["by_transition"]["rank1_to_not1"]
    n = loss["n"]
    if n == 0:
        return "unstructured: no rank1_to_not1 losses to classify"
    overlap_rates = []
    for prefix in ["teacher", "brain_eval", "recon", "enhanced"]:
        rate = loss.get(f"{prefix}_winner_in_target_top50_rate")
        if rate is not None:
            overlap_rates.append(rate)
    max_overlap = max(overlap_rates) if overlap_rates else 0.0
    teacher_dir = loss["teacher_dir_cosine"]["mean"]
    enhanced_dir = loss["enhanced_dir_cosine"]["mean"]
    conc = summary["rank1_to_not1_concentration"]
    if max_overlap >= 0.60:
        return "near-neighbor ambiguity"
    if (teacher_dir is not None and teacher_dir >= 0.20 and conc["top10_share"] is not None and conc["top10_share"] >= 0.35) or (
        conc["top5_share"] is not None and conc["top5_share"] >= 0.30
    ):
        return "directional subject-alignment drift"
    if enhanced_dir is not None and enhanced_dir >= 0.20 and max_overlap >= 0.30:
        return "downstream reconstruction/refiner amplification"
    return "unstructured sparse-noise effects"


def load_or_compute_recon_features(args, model_name, suffix):
    out_path = os.path.join(args.outdir, f"{model_name}_{suffix}_openclip_bigG_flat_norm.pt")
    if os.path.exists(out_path):
        return load_feature_tensor(out_path), out_path, "existing_cycle48_cache"
    if args.skip_recon_features:
        return None, None, "skipped_by_argument"

    from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder

    tensor_path = require(os.path.join(args.data_path, "evals", model_name, f"{model_name}_{suffix}.pt"))
    images = torch.load(tensor_path, map_location="cpu")
    if not torch.isfinite(images).all():
        raise RuntimeError(f"Non-finite reconstruction tensor: {tensor_path}")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu_openclip else "cpu")
    if device.type == "cpu":
        raise RuntimeError("OpenCLIP reconstruction feature computation requires CUDA for this diagnostic run.")
    embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
    ).to(device)
    embedder.eval()
    feats = []
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        for start in range(0, len(images), args.recon_batch_size):
            batch = images[start : start + args.recon_batch_size].to(device, non_blocking=True)
            emb = embedder(batch).float().flatten(1)
            emb = F.normalize(emb, dim=-1)
            feats.append(emb.cpu())
            print(f"cycle48: embedded {model_name} {suffix} {min(start + args.recon_batch_size, len(images))}/{len(images)}", flush=True)
    feat = F.normalize(torch.cat(feats, dim=0).float(), dim=-1)
    if args.cache_recon_features:
        torch.save(feat.to(torch.float16), out_path)
        return feat, out_path, "computed_from_saved_reconstruction_tensor_cached_under_cycle48_outdir"
    return feat, None, "computed_from_saved_reconstruction_tensor_not_cached"


def analyze_subject(args, subject, teacher):
    outdir = args.outdir
    margin0_name = ROWS[subject]["margin0"]
    low_name = ROWS[subject]["margin_low"]
    delta_path = require(os.path.join(args.data_path, "tables/cycle44_rank_margin_diagnostics", f"subj{subject:02d}_margin_low_vs_margin0_rank_margin_delta.csv"))
    control_path = require(os.path.join(args.data_path, "tables/cycle44_rank_margin_diagnostics", f"{margin0_name}_rank_margin.csv"))
    cycle47_path = require(os.path.join(args.data_path, "tables/cycle47_reliability_diagnostics", f"subj{subject:02d}_cycle47_reliability_diagnostics.csv"))

    delta = pd.read_csv(delta_path)
    control = pd.read_csv(control_path)
    c47 = pd.read_csv(cycle47_path)
    if len(delta) != 1000 or len(control) != 1000 or len(c47) != 1000:
        raise RuntimeError(f"Expected 1000 rows for subject {subject}")

    pred0, pred0_path = load_clipvoxels(args.data_path, margin0_name)
    pred_low, pred_low_path = load_clipvoxels(args.data_path, low_name)
    image_rank0 = image_rank(pred0, teacher)
    image_rank_low = image_rank(pred_low, teacher)

    winners_low = delta["hardest_impostor_index"].astype(int).to_numpy()
    winners0 = control["hardest_impostor_index"].astype(int).to_numpy()

    recon_low, recon_path, recon_source = load_or_compute_recon_features(args, low_name, "all_recons")
    enhanced_low, enhanced_path, enhanced_source = load_or_compute_recon_features(args, low_name, "all_enhancedrecons")

    table = pd.DataFrame(
        OrderedDict(
            subject=subject,
            eval_index=delta["eval_index"].astype(int),
            image_id=delta["image_id"],
            margin0_brainret_rank=control["full_pool_rank"],
            margin_low_brainret_rank=delta["full_pool_rank"],
            margin0_winning_impostor_index=winners0,
            margin_low_winning_impostor_index=winners_low,
            rank_transition_class=[
                transition_class(int(r0), int(r1))
                for r0, r1 in zip(control["full_pool_rank"].astype(int), delta["full_pool_rank"].astype(int))
            ],
            positive_similarity=delta["positive_similarity"],
            hardest_impostor_similarity=delta["hardest_impostor_similarity"],
            positive_minus_hardest_margin=delta["margin"],
            delta_positive_similarity=delta["delta_positive_similarity"],
            delta_hardest_impostor_similarity=delta["delta_hardest_impostor_similarity"],
            delta_margin=delta["delta_margin"],
            delta_full_pool_rank=delta["delta_full_pool_rank"],
            image_full_pool_rank_margin0=image_rank0,
            image_full_pool_rank=image_rank_low,
            delta_image_full_pool_rank=image_rank_low - image_rank0,
            cycle47_stress_image_preserved_or_improved_brain_worse=(c47["delta_image_full_pool_rank"].to_numpy() <= 0)
            & (c47["delta_full_pool_rank"].to_numpy() > 0),
            cycle47_rank1_to_not1=c47["rank1_to_not1"].astype(int),
            cycle47_not1_to_rank1=c47["not1_to_rank1"].astype(int),
        )
    )

    spaces = OrderedDict(teacher=teacher, brain_eval=pred_low)
    if recon_low is not None:
        spaces["recon"] = recon_low
    if enhanced_low is not None:
        spaces["enhanced"] = enhanced_low
    for prefix, feats in spaces.items():
        membership = topk_membership(feats, winners_low, TOP_KS)
        for k, values in membership.items():
            table[f"{prefix}_winner_in_target_top{k}"] = values

    delta_pred = pred_low - pred0
    table["teacher_dir_cosine"] = cosine_to_delta(delta_pred, teacher[winners_low] - teacher)
    table["brain_eval_dir_cosine"] = cosine_to_delta(delta_pred, pred_low[winners_low] - pred_low)
    if enhanced_low is not None:
        table["enhanced_dir_cosine"] = cosine_to_delta(delta_pred, enhanced_low[winners_low] - teacher)
    else:
        table["enhanced_dir_cosine"] = np.nan

    table_path = os.path.join(outdir, f"subj{subject:02d}_transition_table.csv")
    table.to_csv(table_path, index=False)

    concentration_rows = []
    for impostor_idx, grp in table.groupby("margin_low_winning_impostor_index"):
        classes = Counter(grp["rank_transition_class"].tolist())
        concentration_rows.append(
            OrderedDict(
                subject=subject,
                impostor_index=int(impostor_idx),
                count=int(len(grp)),
                rank1_to_not1=int(classes.get("rank1_to_not1", 0)),
                not1_to_rank1=int(classes.get("not1_to_rank1", 0)),
                rank1_kept=int(classes.get("rank1_kept", 0)),
                not1_kept=int(classes.get("not1_kept", 0)),
                target_indices=";".join(str(int(v)) for v in grp["eval_index"].tolist()[:25]),
            )
        )
    concentration_rows = sorted(concentration_rows, key=lambda r: (-r["rank1_to_not1"], -r["count"], r["impostor_index"]))
    concentration_path = os.path.join(outdir, f"subj{subject:02d}_impostor_concentration.csv")
    with open(concentration_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(concentration_rows[0].keys()))
        writer.writeheader()
        writer.writerows(concentration_rows)

    stress = table[table["cycle47_stress_image_preserved_or_improved_brain_worse"]]
    loss = table[table["rank_transition_class"] == "rank1_to_not1"]
    metric0, metric0_path = load_final_csv(args.data_path, margin0_name)
    metric_low, metric_low_path = load_final_csv(args.data_path, low_name)
    summary = OrderedDict(
        subject=subject,
        transition_table=table_path,
        impostor_concentration_csv=concentration_path,
        provenance=OrderedDict(
            source_rows=OrderedDict(margin0=margin0_name, margin_low=low_name),
            delta_csv=delta_path,
            control_rank_csv=control_path,
            cycle47_csv=cycle47_path,
            teacher_feature_source=args.teacher_path,
            margin0_clipvoxels=pred0_path,
            margin_low_clipvoxels=pred_low_path,
            recon_feature_source=recon_source,
            recon_feature_path=recon_path,
            enhanced_feature_source=enhanced_source,
            enhanced_feature_path=enhanced_path,
            heldout_labels_used=False,
            diagnostic_only_after_protected_results=True,
            checkpoint_or_eval_dir_written=False,
        ),
        metric_deltas=OrderedDict(
            BrainRet=float(metric_low["BrainRet"] - metric0["BrainRet"]),
            ImageRet=float(metric_low["ImageRet"] - metric0["ImageRet"]),
            CLIP=float(metric_low["CLIP"] - metric0["CLIP"]),
            VC=float(metric_low["VC"] - metric0["VC"]),
            HigherVis=float(metric_low["HigherVis"] - metric0["HigherVis"]),
            metric_csvs=OrderedDict(margin0=metric0_path, margin_low=metric_low_path),
        ),
        transition_counts=OrderedDict((cls, int((table["rank_transition_class"] == cls).sum())) for cls in TRANSITIONS),
        by_transition=summarize_by_transition(table),
        all_impostor_concentration=concentration(table),
        rank1_to_not1_concentration=concentration(loss),
        cycle47_stress_subset=OrderedDict(
            n=int(len(stress)),
            by_transition=summarize_by_transition(stress),
            concentration=concentration(stress),
        ),
        top_rank1_to_not1_impostors=concentration_rows[:10],
    )
    summary["mechanism_classification"] = classify(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Cycle 48 rank-transition identity diagnostics.")
    parser.add_argument("--data_path", default="/src")
    parser.add_argument("--outdir", default="/src/tables/cycle48_rank_transition_diagnostics")
    parser.add_argument("--subjects", nargs="+", type=int, default=[5, 7], choices=[5, 7])
    parser.add_argument("--recon_batch_size", type=int, default=16)
    parser.add_argument("--skip_recon_features", action="store_true")
    parser.add_argument("--cache_recon_features", action="store_true")
    parser.add_argument("--cpu_openclip", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sched = scheduler_state()
    if sched["active_job_count"]:
        raise RuntimeError(f"Active jobs found; stop per plan: {sched['active_jobs']}")
    artifacts, missing = artifact_availability(args.data_path, args.subjects)
    if missing:
        raise FileNotFoundError(json.dumps(missing, indent=2))
    replay_checks = verify_replay(args.data_path, args.subjects)
    for row in replay_checks:
        if not row["within_0p002_tolerance"]:
            raise RuntimeError(f"Cycle 44 replay check outside tolerance: {row}")
    teacher, teacher_path = load_teacher(args.data_path)
    args.teacher_path = teacher_path

    subjects = []
    for subject in args.subjects:
        print(f"cycle48 diagnostics: subject {subject} start", flush=True)
        subjects.append(analyze_subject(args, subject, teacher))
        print(f"cycle48 diagnostics: subject {subject} complete", flush=True)

    summary = OrderedDict(
        provenance=OrderedDict(
            cycle=48,
            scope="diagnostic-only artifact analysis of Cycle 44 rank-1 transitions",
            source_rows=ROWS,
            data_path=args.data_path,
            outdir=args.outdir,
            teacher_feature_source=teacher_path,
            heldout_labels_used=False,
            diagnostic_only_after_protected_results=True,
            no_training_launched=True,
            no_evaluator_relaunch=True,
            checkpoint_or_eval_dir_written=False,
        ),
        preflight=OrderedDict(scheduler=sched, artifacts=artifacts, missing=missing, replay_checks=replay_checks),
        subjects=subjects,
    )
    summary_path = os.path.join(args.outdir, "cycle48_rank_transition_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=as_builtin)
    print(json.dumps(summary, indent=2, default=as_builtin)[:30000])
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
