import argparse
import json
import os
from collections import OrderedDict

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import webdataset as wds

from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder


ROWS = {
    5: {
        "zero": "cycle38_subj05_neighbor0_1sess_150ep",
        "low": "cycle38_subj05_neighbor_low_1sess_150ep",
    },
    7: {
        "zero": "cycle38_subj07_neighbor0_1sess_150ep",
        "low": "cycle38_subj07_neighbor_low_1sess_150ep",
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


def as_float(x):
    if isinstance(x, (np.generic,)):
        return x.item()
    return float(x)


def summarize_tensor(path):
    tensor = torch.load(path, map_location="cpu")
    x = tensor.float()
    return {
        "path": path,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": bool(torch.isfinite(x).all().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
    }


def load_new_test_image_ids(data_path, subj):
    if subj in (3, 6):
        num_test = 2371
    elif subj in (4, 8):
        num_test = 2188
    else:
        num_test = 3000
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


def ranks_from_similarity(sim):
    order = torch.argsort(sim, dim=1, descending=True)
    labels = torch.arange(sim.shape[0], device=sim.device)
    return (order == labels[:, None]).nonzero()[:, 1].float() + 1.0


@torch.no_grad()
def embed_images(images, batch_size, device):
    embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
    ).to(device)
    embedder.eval()
    chunks = []
    for start in tqdm(range(0, len(images), batch_size), desc="openclip image batches"):
        image = images[start : start + batch_size].to(device).float()
        emb = embedder(image).float().flatten(1).cpu()
        chunks.append(emb)
    return F.normalize(torch.cat(chunks, dim=0), dim=-1)


def get_teacher_features(data_path, outdir, batch_size, device):
    cache_path = os.path.join(outdir, "all_images_openclip_bigG_flat_norm.pt")
    fallback = os.path.join(data_path, "tables/cycle38_neighbor_diagnostics/all_images_openclip_bigG_flat_norm.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu"), cache_path
    if os.path.exists(fallback):
        teacher = torch.load(fallback, map_location="cpu")
        os.makedirs(outdir, exist_ok=True)
        torch.save(teacher, cache_path)
        return teacher, cache_path
    all_images = torch.load(os.path.join(data_path, "evals/all_images.pt"), map_location="cpu")
    if all_images.shape[-1] != 256:
        all_images = transforms.Resize((256, 256))(all_images).float()
    teacher = embed_images(all_images, batch_size, device)
    os.makedirs(outdir, exist_ok=True)
    torch.save(teacher, cache_path)
    return teacher, cache_path


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


def get_stage_features(data_path, outdir, model_name, stage, batch_size, device, recompute=False):
    cache_path = os.path.join(outdir, f"{model_name}_{stage}_openclip_bigG_flat_norm.pt")
    if os.path.exists(cache_path) and not recompute:
        return torch.load(cache_path, map_location="cpu"), cache_path
    images = load_stage_images(data_path, model_name, stage)
    if images.shape[-1] != 256:
        images = transforms.Resize((256, 256))(images)
    features = embed_images(images, batch_size, device).half()
    torch.save(features, cache_path)
    return features, cache_path


def retrieval_from_features(query, gallery):
    sim = query.float() @ gallery.float().T
    rank = ranks_from_similarity(sim)
    score = sim.diag()
    return rank, score


def direct_pred_clip_retrieval(data_path, model_name, teacher):
    path = os.path.join(data_path, "evals", model_name, f"{model_name}_all_clipvoxels.pt")
    clipvoxels = torch.load(path, map_location="cpu")
    student = F.normalize(clipvoxels.float().flatten(1), dim=-1)
    image_rank, image_score = retrieval_from_features(student, teacher)
    brain_rank, brain_score = retrieval_from_features(teacher, student)
    return {
        "features": student,
        "image_rank": image_rank,
        "image_score": image_score,
        "brain_rank": brain_rank,
        "brain_score": brain_score,
        "path": path,
    }


def pixcorr_per_image(data_path, model_name, stage):
    all_images = torch.load(os.path.join(data_path, "evals/all_images.pt"), map_location="cpu").float()
    recons = load_stage_images(data_path, model_name, stage).float()
    if all_images.shape[-1] != 425:
        all_images = transforms.Resize(425)(all_images)
    if recons.shape[-1] != 425:
        recons = transforms.Resize(425)(recons)
    flat_images = all_images.reshape(len(all_images), -1).numpy()
    flat_recons = recons.reshape(len(recons), -1).numpy()
    out = []
    for i in range(len(flat_images)):
        out.append(float(np.corrcoef(flat_images[i], flat_recons[i])[0, 1]))
    return np.asarray(out, dtype=np.float32)


def load_final_csv(data_path, model_name):
    path = os.path.join(data_path, "tables", f"{model_name}_all_enhancedrecons.csv")
    df = pd.read_csv(path, sep="\t")
    values = [float(v) for v in df["Value"].tolist()]
    if len(values) != len(CSV_METRICS):
        raise RuntimeError(f"Unexpected metric count in {path}: {len(values)}")
    return OrderedDict(zip(CSV_METRICS, values)), path


def summarize_delta(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "worse_fraction": float(np.mean(values > 0)),
        "improved_fraction": float(np.mean(values < 0)),
    }


def subset_summary(df, mask, delta_cols, top_col, topn):
    sub = df.loc[mask].copy()
    out = {
        "count": int(len(sub)),
        "fraction": float(len(sub) / len(df)),
    }
    for col in delta_cols:
        out[f"{col}_mean"] = float(sub[col].mean()) if len(sub) else None
        out[f"{col}_median"] = float(sub[col].median()) if len(sub) else None
    out["top_worst_examples"] = (
        sub.sort_values(top_col, ascending=False).head(topn).to_dict(orient="records") if len(sub) else []
    )
    return out


def summarize_subject(args, subj, teacher, device):
    ids = load_new_test_image_ids(args.data_path, subj)
    zero_name = ROWS[subj]["zero"]
    low_name = ROWS[subj]["low"]
    zero_csv, zero_csv_path = load_final_csv(args.data_path, zero_name)
    low_csv, low_csv_path = load_final_csv(args.data_path, low_name)

    zero_direct = direct_pred_clip_retrieval(args.data_path, zero_name, teacher)
    low_direct = direct_pred_clip_retrieval(args.data_path, low_name, teacher)

    df = pd.DataFrame(
        {
            "eval_index": np.arange(len(ids)),
            "nsd_image_id": ids,
            "predclip_image_rank_neighbor0": zero_direct["image_rank"].numpy(),
            "predclip_image_rank_neighbor_low": low_direct["image_rank"].numpy(),
            "predclip_image_rank_delta_low_minus_zero": (
                low_direct["image_rank"] - zero_direct["image_rank"]
            ).numpy(),
            "predclip_brain_rank_neighbor0": zero_direct["brain_rank"].numpy(),
            "predclip_brain_rank_neighbor_low": low_direct["brain_rank"].numpy(),
            "predclip_brain_rank_delta_low_minus_zero": (
                low_direct["brain_rank"] - zero_direct["brain_rank"]
            ).numpy(),
            "predclip_image_score_delta_low_minus_zero": (
                low_direct["image_score"] - zero_direct["image_score"]
            ).numpy(),
            "predclip_brain_score_delta_low_minus_zero": (
                low_direct["brain_score"] - zero_direct["brain_score"]
            ).numpy(),
        }
    )

    stage_summaries = OrderedDict()
    for stage in args.stages:
        z_pix = pixcorr_per_image(args.data_path, zero_name, stage)
        l_pix = pixcorr_per_image(args.data_path, low_name, stage)
        prefix = f"{stage}_"
        feature_caches = None
        image_rank_summary = None
        brainlike_rank_summary = None
        if not args.skip_stage_clip_features:
            zero_feat, zero_cache = get_stage_features(
                args.data_path,
                args.outdir,
                zero_name,
                stage,
                args.batch_size,
                device,
                args.recompute_features,
            )
            low_feat, low_cache = get_stage_features(
                args.data_path,
                args.outdir,
                low_name,
                stage,
                args.batch_size,
                device,
                args.recompute_features,
            )
            z_img_rank, z_img_score = retrieval_from_features(zero_feat, teacher)
            l_img_rank, l_img_score = retrieval_from_features(low_feat, teacher)
            z_bwd_rank, z_bwd_score = retrieval_from_features(teacher, zero_feat)
            l_bwd_rank, l_bwd_score = retrieval_from_features(teacher, low_feat)
            df[f"{prefix}image_rank_neighbor0"] = z_img_rank.numpy()
            df[f"{prefix}image_rank_neighbor_low"] = l_img_rank.numpy()
            df[f"{prefix}image_rank_delta_low_minus_zero"] = (l_img_rank - z_img_rank).numpy()
            df[f"{prefix}brainlike_rank_neighbor0"] = z_bwd_rank.numpy()
            df[f"{prefix}brainlike_rank_neighbor_low"] = l_bwd_rank.numpy()
            df[f"{prefix}brainlike_rank_delta_low_minus_zero"] = (l_bwd_rank - z_bwd_rank).numpy()
            df[f"{prefix}image_score_delta_low_minus_zero"] = (l_img_score - z_img_score).numpy()
            df[f"{prefix}brainlike_score_delta_low_minus_zero"] = (l_bwd_score - z_bwd_score).numpy()
            feature_caches = {"neighbor0": zero_cache, "neighbor_low": low_cache}
            image_rank_summary = summarize_delta(df[f"{prefix}image_rank_delta_low_minus_zero"])
            brainlike_rank_summary = summarize_delta(df[f"{prefix}brainlike_rank_delta_low_minus_zero"])
        df[f"{prefix}pixcorr_neighbor0"] = z_pix
        df[f"{prefix}pixcorr_neighbor_low"] = l_pix
        df[f"{prefix}pixcorr_delta_low_minus_zero"] = l_pix - z_pix
        stage_summaries[stage] = {
            "feature_caches": feature_caches,
            "image_rank_delta": image_rank_summary,
            "brainlike_rank_delta": brainlike_rank_summary,
            "pixcorr_delta_mean": float(df[f"{prefix}pixcorr_delta_low_minus_zero"].mean()),
            "pixcorr_delta_median": float(df[f"{prefix}pixcorr_delta_low_minus_zero"].median()),
        }

    stress_cols = [col for col in [
        "predclip_image_rank_delta_low_minus_zero",
        "predclip_brain_rank_delta_low_minus_zero",
        "enhanced_evalmix_image_rank_delta_low_minus_zero",
        "enhanced_evalmix_brainlike_rank_delta_low_minus_zero",
        "enhanced_evalmix_pixcorr_delta_low_minus_zero",
    ] if col in df.columns]
    stress = OrderedDict()
    final_worse = df["predclip_brain_rank_delta_low_minus_zero"] > 0
    image_rank_col = (
        "enhanced_evalmix_image_rank_delta_low_minus_zero"
        if "enhanced_evalmix_image_rank_delta_low_minus_zero" in df.columns
        else "predclip_image_rank_delta_low_minus_zero"
    )
    stress["image_rank_preserved_or_improved_while_final_brainret_worsened"] = subset_summary(
        df,
        (df[image_rank_col] <= 0) & final_worse,
        stress_cols,
        "predclip_brain_rank_delta_low_minus_zero",
        args.topn,
    )
    stress["predclip_rank_preserved_or_improved_while_final_brainret_worsened"] = subset_summary(
        df,
        (df["predclip_image_rank_delta_low_minus_zero"] <= 0) & final_worse,
        stress_cols,
        "predclip_brain_rank_delta_low_minus_zero",
        args.topn,
    )
    stress["pixcorr_improved_while_final_brainret_worsened"] = subset_summary(
        df,
        (df["enhanced_evalmix_pixcorr_delta_low_minus_zero"] > 0) & final_worse,
        stress_cols,
        "predclip_brain_rank_delta_low_minus_zero",
        args.topn,
    )
    stress["final_brainret_improved_despite_image_side_worsening"] = subset_summary(
        df,
        (df["predclip_brain_rank_delta_low_minus_zero"] < 0)
        & (df[image_rank_col] > 0),
        stress_cols,
        image_rank_col,
        args.topn,
    )

    csv_path = os.path.join(args.outdir, f"subj{subj:02d}_neighbor_low_vs_neighbor0_per_image.csv")
    df.to_csv(csv_path, index=False)
    aggregate_deltas = OrderedDict((k, low_csv[k] - zero_csv[k]) for k in CSV_METRICS)
    direct_summary = {
        "image_rank_delta": summarize_delta(df["predclip_image_rank_delta_low_minus_zero"]),
        "brain_rank_delta": summarize_delta(df["predclip_brain_rank_delta_low_minus_zero"]),
        "worst_predclip_brain_rank_regressions": df.sort_values(
            "predclip_brain_rank_delta_low_minus_zero", ascending=False
        ).head(args.topn).to_dict(orient="records"),
    }
    return {
        "subject": subj,
        "rows": ROWS[subj],
        "num_images": int(len(df)),
        "per_image_csv": csv_path,
        "final_csv_paths": {"neighbor0": zero_csv_path, "neighbor_low": low_csv_path},
        "final_csv_neighbor_low_minus_neighbor0": aggregate_deltas,
        "direct_predicted_clip": direct_summary,
        "stage_summaries": stage_summaries,
        "stress_subsets": stress,
    }


def roi_feasibility(data_path):
    script_path = os.path.join(data_path, "final_evaluations.py")
    text = open(script_path).read()
    return {
        "per_image_roi_available": False,
        "reason": (
            "final_evaluations.py computes GNet ROI correlations by averaging PearsonCorrCoef "
            "over voxels and writes only aggregate CSV values; no per-image ROI tensor is saved."
        ),
        "evidence": {
            "uses_GNet8_Encoder": "GNet8_Encoder" in text,
            "writes_only_metric_value_csv": "df[\"Value\"].to_csv" in text,
            "region_keys": ["nsd_general", "V1", "V2", "V3", "V4", "higher_vis"],
        },
    }


def preflight(data_path, subjects=(5, 7)):
    artifact_summaries = OrderedDict()
    missing = []
    job_status_path = "/job-status.md"
    job_status = {
        "path": job_status_path,
        "exists": os.path.exists(job_status_path),
        "note": "No active-job content was available from /job-status.md." if not os.path.exists(job_status_path) else "",
    }
    for subj in subjects:
        rows = ROWS[subj]
        for label, model_name in rows.items():
            base = os.path.join(data_path, "evals", model_name)
            row = OrderedDict()
            for suffix in ["all_clipvoxels", "all_recons", "all_enhancedrecons", "all_blurryrecons"]:
                path = os.path.join(base, f"{model_name}_{suffix}.pt")
                if not os.path.exists(path):
                    missing.append(path)
                    continue
                row[suffix] = summarize_tensor(path)
            csv_path = os.path.join(data_path, "tables", f"{model_name}_all_enhancedrecons.csv")
            row["final_csv"] = {"path": csv_path, "exists": os.path.exists(csv_path)}
            if not os.path.exists(csv_path):
                missing.append(csv_path)
            artifact_summaries[f"subj{subj:02d}_{label}"] = row
    return {"job_status": job_status, "missing": missing, "artifacts": artifact_summaries}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/src")
    parser.add_argument("--outdir", default="/src/tables/cycle39_failure_decomposition")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--topn", type=int, default=20)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["blurry", "recons", "enhanced", "enhanced_evalmix"],
        choices=["blurry", "recons", "enhanced", "enhanced_evalmix"],
    )
    parser.add_argument("--recompute_features", action="store_true")
    parser.add_argument(
        "--skip_stage_clip_features",
        action="store_true",
        help="Fallback mode: compute direct predicted-CLIP ranks and stage PixCorr only.",
    )
    parser.add_argument("--subjects", nargs="+", type=int, default=[5, 7], choices=[5, 7])
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preflight_info = preflight(args.data_path, args.subjects)
    if preflight_info["missing"]:
        raise FileNotFoundError(json.dumps(preflight_info["missing"], indent=2))

    teacher, teacher_cache = get_teacher_features(args.data_path, args.outdir, args.batch_size, device)
    subjects = []
    for subj in args.subjects:
        subjects.append(summarize_subject(args, subj, teacher, device))

    out = OrderedDict(
        provenance={
            "cycle": 39,
            "scope": "post-hoc failure decomposition of completed Cycle 38 sparse-neighbor rows",
            "data_path": args.data_path,
            "teacher_cache": teacher_cache,
            "device": device,
            "stages": args.stages,
            "skip_stage_clip_features": args.skip_stage_clip_features,
            "notes": [
                "No checkpoints, training code, model weights, retrieval pool, masks, refiner settings, or evaluator settings are changed.",
                "Rank deltas are low-minus-zero; positive rank deltas mean worse rank.",
                "enhanced_evalmix matches final_evaluations.py's low-level weighted image used for enhanced low-level/perceptual metrics.",
            ],
        },
        preflight=preflight_info,
        roi_feasibility=roi_feasibility(args.data_path),
        subjects=subjects,
    )
    json_path = os.path.join(args.outdir, "cycle39_failure_decomposition_summary.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=as_float)
    print(json.dumps(out, indent=2, default=as_float)[:30000])
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
