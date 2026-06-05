import argparse
import json
import os
import subprocess
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
import webdataset as wds


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


def my_split_by_node(urls):
    return urls


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


def summarize_delta(values):
    arr = np.asarray(values, dtype=np.float64)
    return OrderedDict(
        mean=float(arr.mean()),
        median=float(np.median(arr)),
        worse_fraction=float(np.mean(arr > 0)),
        improved_fraction=float(np.mean(arr < 0)),
    )


def summarize_tensor(path):
    tensor = torch.load(path, map_location="cpu")
    x = tensor.float()
    out = OrderedDict(
        path=path,
        shape=list(tensor.shape),
        dtype=str(tensor.dtype),
        finite=bool(torch.isfinite(x).all().item()),
        min=float(x.min().item()),
        max=float(x.max().item()),
        mean=float(x.mean().item()),
    )
    del tensor, x
    return out


def load_new_test_image_ids(data_path, subj):
    num_test = 3000 if subj not in (3, 4, 6, 8) else (2371 if subj in (3, 6) else 2188)
    test_url = f"{data_path}/wds/subj0{subj}/new_test/0.tar"
    test_data = (
        wds.WebDataset(test_url, resampled=False, nodesplitter=my_split_by_node)
        .decode("torch")
        .rename(
            behav="behav.npy",
            past_behav="past_behav.npy",
            future_behav="future_behav.npy",
            olds_behav="olds_behav.npy",
        )
        .to_tuple("behav", "past_behav", "future_behav", "olds_behav")
    )
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=True)
    image_ids = []
    for behav, *_ in test_dl:
        image_ids = np.append(image_ids, behav[:, 0, 0].cpu().numpy())
    return np.unique(image_ids.astype(int))


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
    raise FileNotFoundError(
        "Missing OpenCLIP teacher cache. Expected an existing all_images_openclip_bigG_flat_norm.pt "
        "from Cycle 38/39; this diagnostic intentionally does not regenerate it."
    )


def ranks_from_similarity(sim):
    order = torch.argsort(sim, dim=1, descending=True)
    labels = torch.arange(sim.shape[0], device=sim.device)
    return (order == labels[:, None]).nonzero()[:, 1].float() + 1.0


def retrieval_from_features(query, gallery):
    sim = query.float() @ gallery.float().T
    rank = ranks_from_similarity(sim)
    score = sim.diag()
    return rank, score, sim


def load_clipvoxels(data_path, model_name):
    path = os.path.join(data_path, "evals", model_name, f"{model_name}_all_clipvoxels.pt")
    raw = torch.load(path, map_location="cpu")
    return raw, path


def feature_stats(raw_clipvoxels):
    x = raw_clipvoxels.float()
    flat = x.flatten(1)
    flat_norm = flat.norm(dim=1)
    token_norm = x.norm(dim=2)
    feature_std = flat.std(dim=0)
    token_var = x.var(dim=0).mean(dim=1)
    centered = flat - flat.mean(dim=0, keepdim=True)
    gram = centered @ centered.T
    eig = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
    total = eig.sum().clamp_min(1e-12)
    prob = eig / total
    entropy = -(prob[prob > 0] * torch.log(prob[prob > 0])).sum()
    effective_rank = torch.exp(entropy)
    top = eig[: min(50, eig.numel())]
    student = F.normalize(flat, dim=-1)
    self_sim = student @ student.T
    off_mask = ~torch.eye(self_sim.shape[0], dtype=torch.bool)
    out = OrderedDict(
        flat_norm=percentile_summary(flat_norm.numpy()),
        per_image_mean_token_norm=percentile_summary(token_norm.mean(dim=1).numpy()),
        per_image_token_norm_std=percentile_summary(token_norm.std(dim=1).numpy()),
        token_mean_norm=percentile_summary(token_norm.mean(dim=0).numpy()),
        token_variance_mean=percentile_summary(token_var.numpy()),
        feature_std=percentile_summary(feature_std.numpy()),
        effective_rank=float(effective_rank.item()),
        pca_top10_variance_fraction=[float(v) for v in (top[:10] / total).numpy()],
        pca_top50_variance_fraction_sum=float((top / total).sum().item()),
        student_self_offdiag_cosine=percentile_summary(self_sim[off_mask].numpy()),
    )
    del x, flat, centered, gram, eig, student, self_sim
    return out


def similarity_diagnostics(student, teacher):
    image_rank, image_score, image_sim = retrieval_from_features(student, teacher)
    brain_rank, brain_score, brain_sim = retrieval_from_features(teacher, student)
    n = image_sim.shape[0]
    eye = torch.eye(n, dtype=torch.bool)
    image_off = image_sim.masked_fill(eye, -float("inf"))
    brain_off = brain_sim.masked_fill(eye, -float("inf"))
    image_top1 = image_off.max(dim=1).values
    brain_top1 = brain_off.max(dim=1).values
    image_top5 = image_off.topk(5, dim=1).values.mean(dim=1)
    brain_top5 = brain_off.topk(5, dim=1).values.mean(dim=1)
    return OrderedDict(
        image_rank=image_rank,
        image_score=image_score,
        brain_rank=brain_rank,
        brain_score=brain_score,
        image_positive_margin_hardest=(image_score - image_top1),
        brain_positive_margin_hardest=(brain_score - brain_top1),
        image_positive_margin_top5=(image_score - image_top5),
        brain_positive_margin_top5=(brain_score - brain_top5),
        summary=OrderedDict(
            true_pair_similarity=percentile_summary(image_score.numpy()),
            image_impostor_similarity=percentile_summary(image_sim[~eye].numpy()),
            brain_impostor_similarity=percentile_summary(brain_sim[~eye].numpy()),
            image_positive_margin_hardest=percentile_summary((image_score - image_top1).numpy()),
            brain_positive_margin_hardest=percentile_summary((brain_score - brain_top1).numpy()),
            image_positive_margin_top5=percentile_summary((image_score - image_top5).numpy()),
            brain_positive_margin_top5=percentile_summary((brain_score - brain_top5).numpy()),
        ),
    )


def load_stage_images(data_path, model_name, stage):
    base = os.path.join(data_path, "evals", model_name)
    if stage == "blurry":
        return torch.load(os.path.join(base, f"{model_name}_all_blurryrecons.pt"), map_location="cpu").float()
    if stage == "recons":
        return torch.load(os.path.join(base, f"{model_name}_all_recons.pt"), map_location="cpu").float()
    if stage == "enhanced":
        return torch.load(os.path.join(base, f"{model_name}_all_enhancedrecons.pt"), map_location="cpu").float()
    if stage == "enhanced_evalmix":
        enhanced = torch.load(os.path.join(base, f"{model_name}_all_enhancedrecons.pt"), map_location="cpu").float()
        blurry = torch.load(os.path.join(base, f"{model_name}_all_blurryrecons.pt"), map_location="cpu").float()
        if enhanced.shape[-1] != 256:
            enhanced = transforms.Resize((256, 256))(enhanced)
        if blurry.shape[-1] != 256:
            blurry = transforms.Resize((256, 256))(blurry)
        return enhanced * 0.75 + blurry * 0.25
    raise ValueError(stage)


def pixcorr_per_image(data_path, model_name, stage):
    all_images = torch.load(os.path.join(data_path, "evals/all_images.pt"), map_location="cpu").float()
    recons = load_stage_images(data_path, model_name, stage).float()
    if all_images.shape[-1] != 425:
        all_images = transforms.Resize(425)(all_images)
    if recons.shape[-1] != 425:
        recons = transforms.Resize(425)(recons)
    flat_images = all_images.reshape(len(all_images), -1).numpy()
    flat_recons = recons.reshape(len(recons), -1).numpy()
    out = np.empty(len(flat_images), dtype=np.float32)
    for i in range(len(flat_images)):
        out[i] = np.corrcoef(flat_images[i], flat_recons[i])[0, 1]
    del all_images, recons, flat_images, flat_recons
    return out


def load_final_csv(data_path, model_name):
    path = os.path.join(data_path, "tables", f"{model_name}_all_enhancedrecons.csv")
    df = pd.read_csv(path, sep="\t")
    values = [float(v) for v in df["Value"].tolist()]
    if len(values) != len(CSV_METRICS):
        raise RuntimeError(f"Unexpected metric count in {path}: {len(values)}")
    return OrderedDict(zip(CSV_METRICS, values)), path


def correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def subset_summary(df, mask, delta_cols, top_col, topn):
    sub = df.loc[mask].copy()
    out = OrderedDict(count=int(len(sub)), fraction=float(len(sub) / len(df)))
    for col in delta_cols:
        out[f"{col}_mean"] = float(sub[col].mean()) if len(sub) else None
        out[f"{col}_median"] = float(sub[col].median()) if len(sub) else None
    out["top_worst_examples"] = (
        sub.sort_values(top_col, ascending=False).head(topn).to_dict(orient="records") if len(sub) else []
    )
    return out


def scheduler_state():
    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%.18i %.40j %.8T %.10M %.9l"],
            check=False,
            capture_output=True,
            text=True,
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        return OrderedDict(active_job_count=len(lines), active_jobs=lines, squeue_returncode=result.returncode)
    except FileNotFoundError:
        return OrderedDict(active_job_count=None, active_jobs=[], note="squeue not available")


def preflight(data_path, subjects):
    artifact_summaries = OrderedDict()
    missing = []
    for subj in subjects:
        for label, model_name in ROWS[subj].items():
            base = os.path.join(data_path, "evals", model_name)
            row = OrderedDict(eval_dir=base, eval_dir_exists=os.path.isdir(base))
            if not os.path.isdir(base):
                missing.append(base)
            for suffix in ["all_clipvoxels", "all_blurryrecons", "all_recons", "all_enhancedrecons"]:
                path = os.path.join(base, f"{model_name}_{suffix}.pt")
                if not os.path.exists(path):
                    missing.append(path)
                else:
                    row[suffix] = summarize_tensor(path)
            csv_path = os.path.join(data_path, "tables", f"{model_name}_all_enhancedrecons.csv")
            row["final_csv"] = OrderedDict(path=csv_path, exists=os.path.exists(csv_path))
            if not os.path.exists(csv_path):
                missing.append(csv_path)
            artifact_summaries[f"subj{subj:02d}_{label}"] = row
    return OrderedDict(
        job_status=OrderedDict(
            path="/job-status.md",
            exists=os.path.exists("/job-status.md"),
            note="No /job-status.md was visible." if not os.path.exists("/job-status.md") else "",
        ),
        scheduler=scheduler_state(),
        missing=missing,
        artifacts=artifact_summaries,
    )


def summarize_subject(args, subj, teacher):
    ids = load_new_test_image_ids(args.data_path, subj)
    prior0_name = ROWS[subj]["prior0"]
    prior_low_name = ROWS[subj]["prior_low"]
    prior0_csv, prior0_csv_path = load_final_csv(args.data_path, prior0_name)
    prior_low_csv, prior_low_csv_path = load_final_csv(args.data_path, prior_low_name)

    raw0, raw0_path = load_clipvoxels(args.data_path, prior0_name)
    raw_low, raw_low_path = load_clipvoxels(args.data_path, prior_low_name)
    student0 = F.normalize(raw0.float().flatten(1), dim=-1)
    student_low = F.normalize(raw_low.float().flatten(1), dim=-1)
    stats0 = feature_stats(raw0)
    stats_low = feature_stats(raw_low)
    sim0 = similarity_diagnostics(student0, teacher)
    sim_low = similarity_diagnostics(student_low, teacher)

    df = pd.DataFrame(
        {
            "subject": subj,
            "eval_index": np.arange(len(ids)),
            "nsd_image_id": ids,
            "prior0_image_rank": sim0["image_rank"].numpy(),
            "prior_low_image_rank": sim_low["image_rank"].numpy(),
            "image_rank_delta_prior_low_minus_prior0": (sim_low["image_rank"] - sim0["image_rank"]).numpy(),
            "prior0_brain_rank": sim0["brain_rank"].numpy(),
            "prior_low_brain_rank": sim_low["brain_rank"].numpy(),
            "brain_rank_delta_prior_low_minus_prior0": (sim_low["brain_rank"] - sim0["brain_rank"]).numpy(),
            "image_score_delta_prior_low_minus_prior0": (sim_low["image_score"] - sim0["image_score"]).numpy(),
            "brain_score_delta_prior_low_minus_prior0": (sim_low["brain_score"] - sim0["brain_score"]).numpy(),
            "flat_norm_delta_prior_low_minus_prior0": (
                raw_low.float().flatten(1).norm(dim=1) - raw0.float().flatten(1).norm(dim=1)
            ).numpy(),
            "mean_token_norm_delta_prior_low_minus_prior0": (
                raw_low.float().norm(dim=2).mean(dim=1) - raw0.float().norm(dim=2).mean(dim=1)
            ).numpy(),
            "brain_margin_hardest_delta_prior_low_minus_prior0": (
                sim_low["brain_positive_margin_hardest"] - sim0["brain_positive_margin_hardest"]
            ).numpy(),
            "image_margin_hardest_delta_prior_low_minus_prior0": (
                sim_low["image_positive_margin_hardest"] - sim0["image_positive_margin_hardest"]
            ).numpy(),
            "brain_margin_top5_delta_prior_low_minus_prior0": (
                sim_low["brain_positive_margin_top5"] - sim0["brain_positive_margin_top5"]
            ).numpy(),
            "image_margin_top5_delta_prior_low_minus_prior0": (
                sim_low["image_positive_margin_top5"] - sim0["image_positive_margin_top5"]
            ).numpy(),
        }
    )

    stage_summaries = OrderedDict()
    for stage in ["blurry", "recons", "enhanced", "enhanced_evalmix"]:
        z_pix = pixcorr_per_image(args.data_path, prior0_name, stage)
        l_pix = pixcorr_per_image(args.data_path, prior_low_name, stage)
        col = f"{stage}_pixcorr_delta_prior_low_minus_prior0"
        df[f"{stage}_pixcorr_prior0"] = z_pix
        df[f"{stage}_pixcorr_prior_low"] = l_pix
        df[col] = l_pix - z_pix
        stage_summaries[stage] = OrderedDict(
            pixcorr_delta_mean=float(df[col].mean()),
            pixcorr_delta_median=float(df[col].median()),
            pixcorr_delta_improved_fraction=float((df[col] > 0).mean()),
            pixcorr_delta_worse_fraction=float((df[col] < 0).mean()),
        )

    corr_targets = [
        "flat_norm_delta_prior_low_minus_prior0",
        "mean_token_norm_delta_prior_low_minus_prior0",
        "brain_margin_hardest_delta_prior_low_minus_prior0",
        "image_margin_hardest_delta_prior_low_minus_prior0",
        "brain_margin_top5_delta_prior_low_minus_prior0",
        "image_margin_top5_delta_prior_low_minus_prior0",
        "enhanced_evalmix_pixcorr_delta_prior_low_minus_prior0",
    ]
    correlations = OrderedDict(
        (col, correlation(df[col], df["brain_rank_delta_prior_low_minus_prior0"])) for col in corr_targets
    )

    per_image_path = os.path.join(args.outdir, f"subj{subj:02d}_prior_low_vs_prior0_per_image.csv")
    df.to_csv(per_image_path, index=False)
    aggregate_deltas = OrderedDict((k, prior_low_csv[k] - prior0_csv[k]) for k in CSV_METRICS)
    stress_cols = [
        "image_rank_delta_prior_low_minus_prior0",
        "brain_rank_delta_prior_low_minus_prior0",
        "enhanced_evalmix_pixcorr_delta_prior_low_minus_prior0",
        "flat_norm_delta_prior_low_minus_prior0",
        "brain_margin_hardest_delta_prior_low_minus_prior0",
    ]
    brain_worse = df["brain_rank_delta_prior_low_minus_prior0"] > 0
    stress = OrderedDict(
        pixcorr_improved_while_brain_rank_worsened=subset_summary(
            df,
            (df["enhanced_evalmix_pixcorr_delta_prior_low_minus_prior0"] > 0) & brain_worse,
            stress_cols,
            "brain_rank_delta_prior_low_minus_prior0",
            args.topn,
        ),
        image_rank_preserved_while_brain_rank_worsened=subset_summary(
            df,
            (df["image_rank_delta_prior_low_minus_prior0"] <= 0) & brain_worse,
            stress_cols,
            "brain_rank_delta_prior_low_minus_prior0",
            args.topn,
        ),
        brain_rank_improved_despite_image_rank_worse=subset_summary(
            df,
            (df["brain_rank_delta_prior_low_minus_prior0"] < 0)
            & (df["image_rank_delta_prior_low_minus_prior0"] > 0),
            stress_cols,
            "image_rank_delta_prior_low_minus_prior0",
            args.topn,
        ),
    )

    result = OrderedDict(
        subject=subj,
        rows=ROWS[subj],
        num_images=int(len(df)),
        per_image_csv=per_image_path,
        clipvoxel_paths=OrderedDict(prior0=raw0_path, prior_low=raw_low_path),
        final_csv_paths=OrderedDict(prior0=prior0_csv_path, prior_low=prior_low_csv_path),
        final_csv_prior_low_minus_prior0=aggregate_deltas,
        direct_predicted_clip=OrderedDict(
            image_rank_delta=summarize_delta(df["image_rank_delta_prior_low_minus_prior0"]),
            brain_rank_delta=summarize_delta(df["brain_rank_delta_prior_low_minus_prior0"]),
            worst_brain_rank_regressions=df.sort_values(
                "brain_rank_delta_prior_low_minus_prior0", ascending=False
            )
            .head(args.topn)
            .to_dict(orient="records"),
        ),
        feature_calibration=OrderedDict(prior0=stats0, prior_low=stats_low),
        pairwise_similarity=OrderedDict(prior0=sim0["summary"], prior_low=sim_low["summary"]),
        delta_correlations_with_brain_rank_delta=correlations,
        stage_pixcorr=stage_summaries,
        stress_subsets=stress,
    )
    del raw0, raw_low, student0, student_low
    return result


def roi_feasibility(data_path):
    script_path = os.path.join(data_path, "final_evaluations.py")
    text = open(script_path).read() if os.path.exists(script_path) else ""
    return OrderedDict(
        per_image_roi_available=False,
        reason=(
            "Saved Cycle 40 artifacts expose aggregate ROI CSV values only. final_evaluations.py "
            "computes GNet ROI correlations and writes averaged metric values, not per-image ROI tensors."
        ),
        evidence=OrderedDict(
            final_evaluations_exists=os.path.exists(script_path),
            uses_GNet8_Encoder="GNet8_Encoder" in text,
            writes_only_metric_value_csv='df["Value"].to_csv' in text,
            region_keys=["nsd_general", "V1", "V2", "V3", "V4", "higher_vis"],
        ),
    )


def classify(subjects):
    out = OrderedDict()
    for subj in subjects:
        brain_mean = subj["direct_predicted_clip"]["brain_rank_delta"]["mean"]
        image_mean = subj["direct_predicted_clip"]["image_rank_delta"]["mean"]
        margin0 = subj["pairwise_similarity"]["prior0"]["brain_positive_margin_hardest"]["mean"]
        margin_low = subj["pairwise_similarity"]["prior_low"]["brain_positive_margin_hardest"]["mean"]
        er0 = subj["feature_calibration"]["prior0"]["effective_rank"]
        erlow = subj["feature_calibration"]["prior_low"]["effective_rank"]
        if brain_mean > 50:
            label = "upstream predicted-CLIP collapse"
        elif brain_mean > 5:
            label = "upstream predicted-CLIP regression"
        elif margin_low < margin0 * 0.5 or erlow < er0 * 0.75:
            label = "calibration/separability collapse"
        elif brain_mean > 0:
            label = "wrong-neighborhood or subject-alignment rotation"
        else:
            label = "downstream amplification or aggregate BrainRet mismatch"
        out[f"subj{subj['subject']:02d}"] = OrderedDict(
            label=label,
            direct_brain_rank_mean_delta=brain_mean,
            direct_image_rank_mean_delta=image_mean,
            brain_margin_hardest_mean_prior0=margin0,
            brain_margin_hardest_mean_prior_low=margin_low,
            effective_rank_prior0=er0,
            effective_rank_prior_low=erlow,
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/src")
    parser.add_argument("--outdir", default="/src/tables/cycle41_prior_collapse_diagnostic")
    parser.add_argument("--teacher_cache", default=None)
    parser.add_argument("--topn", type=int, default=20)
    parser.add_argument("--subjects", nargs="+", type=int, default=[5, 7], choices=[5, 7])
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    preflight_info = preflight(args.data_path, args.subjects)
    if preflight_info["missing"]:
        raise FileNotFoundError(json.dumps(preflight_info["missing"], indent=2))
    if preflight_info["scheduler"]["active_job_count"]:
        raise RuntimeError(f"Active jobs found: {preflight_info['scheduler']['active_jobs']}")
    teacher, teacher_path = load_teacher(args.data_path, args.teacher_cache)
    subjects = [summarize_subject(args, subj, teacher) for subj in args.subjects]
    out = OrderedDict(
        provenance=OrderedDict(
            cycle=41,
            scope="artifact-only diagnostic of Cycle 40 prior-preservation collapse",
            data_path=args.data_path,
            outdir=args.outdir,
            teacher_cache=teacher_path,
            notes=[
                "No training, checkpoint modification, reconstruction regeneration, evaluator rewrite, or new branch was run.",
                "Rank deltas are prior_low minus prior0; positive rank deltas mean worse rank.",
                "enhanced_evalmix PixCorr uses the evaluator's 0.75 * enhanced + 0.25 * blurry image path.",
            ],
        ),
        preflight=preflight_info,
        roi_feasibility=roi_feasibility(args.data_path),
        subjects=subjects,
        classification=classify(subjects),
    )
    json_path = os.path.join(args.outdir, "cycle41_prior_collapse_summary.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=as_builtin)
    print(json.dumps(out, indent=2, default=as_builtin)[:30000])
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
