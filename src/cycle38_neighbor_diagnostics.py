import argparse
import json
import os

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
        "zero": "cycle36_subj05_topo0_1sess_150ep",
        "low": "cycle36_subj05_topo_low_1sess_150ep",
    },
    7: {
        "zero": "cycle36_subj07_topo0_1sess_150ep",
        "low": "cycle36_subj07_topo_low_1sess_150ep",
    },
}


def my_split_by_node(urls):
    return urls


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


@torch.no_grad()
def get_teacher_features(data_path, cache_path, batch_size, device):
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu")

    all_images = torch.load(os.path.join(data_path, "evals/all_images.pt"), map_location="cpu")
    if all_images.shape[-1] != 256:
        all_images = transforms.Resize((256, 256))(all_images).float()
    embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
    ).to(device)
    chunks = []
    for start in tqdm(range(0, len(all_images), batch_size), desc="teacher openclip"):
        image = all_images[start : start + batch_size].to(device).float()
        emb = embedder(image).float().flatten(1).cpu()
        chunks.append(emb)
    features = F.normalize(torch.cat(chunks, dim=0), dim=-1)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(features, cache_path)
    return features


def ranks_from_similarity(sim):
    order = torch.argsort(sim, dim=1, descending=True)
    labels = torch.arange(sim.shape[0], device=sim.device)
    return (order == labels[:, None]).nonzero()[:, 1].float() + 1.0


def row_retrieval(data_path, model_name, teacher):
    clipvoxels = torch.load(
        os.path.join(data_path, "evals", model_name, f"{model_name}_all_clipvoxels.pt"),
        map_location="cpu",
    )
    student = F.normalize(clipvoxels.float().flatten(1), dim=-1)
    sim_student_to_teacher = student @ teacher.T
    sim_teacher_to_student = teacher @ student.T
    image_rank = ranks_from_similarity(sim_student_to_teacher)
    brain_rank = ranks_from_similarity(sim_teacher_to_student)
    diag_image_score = sim_student_to_teacher.diag()
    diag_brain_score = sim_teacher_to_student.diag()
    return {
        "student": student,
        "image_rank": image_rank,
        "brain_rank": brain_rank,
        "image_score": diag_image_score,
        "brain_score": diag_brain_score,
    }


def teacher_neighbor_stats(teacher, student, k=5):
    n = teacher.shape[0]
    eye = torch.eye(n, dtype=torch.bool)
    teacher_sim = (teacher @ teacher.T).masked_fill(eye, -float("inf"))
    student_sim = (student @ student.T).masked_fill(eye, -float("inf"))
    teacher_top = torch.topk(teacher_sim, k=k, dim=1).indices
    student_order = torch.argsort(student_sim, dim=1, descending=True)
    student_top = student_order[:, :k]
    teacher_top1 = teacher_top[:, 0]
    teacher_top1_rank = (student_order == teacher_top1[:, None]).nonzero()[:, 1].float() + 1.0
    overlap5 = (student_top[:, :, None] == teacher_top[:, None, :]).any(dim=2).float().sum(dim=1) / float(k)
    return teacher_top1_rank, overlap5


def pixcorr_per_image(data_path, model_name):
    all_images = torch.load(os.path.join(data_path, "evals/all_images.pt"), map_location="cpu")
    all_recons = torch.load(
        os.path.join(data_path, "evals", model_name, f"{model_name}_all_enhancedrecons.pt"),
        map_location="cpu",
    )
    blurry = torch.load(
        os.path.join(data_path, "evals", model_name, f"{model_name}_all_blurryrecons.pt"),
        map_location="cpu",
    )
    resize = transforms.Resize((256, 256))
    all_images = resize(all_images).float()
    all_recons = resize(all_recons).float() * 0.75 + resize(blurry).float() * 0.25
    flat_images = transforms.Resize(425)(all_images).reshape(len(all_images), -1).numpy()
    flat_recons = transforms.Resize(425)(all_recons).reshape(len(all_recons), -1).numpy()
    out = []
    for i in range(len(flat_images)):
        out.append(float(np.corrcoef(flat_images[i], flat_recons[i])[0, 1]))
    return np.array(out, dtype=np.float32)


def summarize_subject(data_path, outdir, subj, teacher, topn):
    ids = load_new_test_image_ids(data_path, subj)
    zero_name = ROWS[subj]["zero"]
    low_name = ROWS[subj]["low"]
    zero = row_retrieval(data_path, zero_name, teacher)
    low = row_retrieval(data_path, low_name, teacher)
    zero_top1_rank, zero_overlap5 = teacher_neighbor_stats(teacher, zero["student"], k=5)
    low_top1_rank, low_overlap5 = teacher_neighbor_stats(teacher, low["student"], k=5)
    zero_pix = pixcorr_per_image(data_path, zero_name)
    low_pix = pixcorr_per_image(data_path, low_name)

    df = pd.DataFrame(
        {
            "eval_index": np.arange(len(ids)),
            "nsd_image_id": ids,
            "image_rank_topo0": zero["image_rank"].numpy(),
            "image_rank_topo_low": low["image_rank"].numpy(),
            "image_rank_delta_low_minus_zero": (low["image_rank"] - zero["image_rank"]).numpy(),
            "brain_rank_topo0": zero["brain_rank"].numpy(),
            "brain_rank_topo_low": low["brain_rank"].numpy(),
            "brain_rank_delta_low_minus_zero": (low["brain_rank"] - zero["brain_rank"]).numpy(),
            "image_score_delta_low_minus_zero": (low["image_score"] - zero["image_score"]).numpy(),
            "brain_score_delta_low_minus_zero": (low["brain_score"] - zero["brain_score"]).numpy(),
            "teacher_top1_rank_topo0": zero_top1_rank.numpy(),
            "teacher_top1_rank_topo_low": low_top1_rank.numpy(),
            "teacher_top1_rank_delta_low_minus_zero": (low_top1_rank - zero_top1_rank).numpy(),
            "teacher_neighbor_overlap5_topo0": zero_overlap5.numpy(),
            "teacher_neighbor_overlap5_topo_low": low_overlap5.numpy(),
            "teacher_neighbor_overlap5_delta_low_minus_zero": (low_overlap5 - zero_overlap5).numpy(),
            "pixcorr_topo0": zero_pix,
            "pixcorr_topo_low": low_pix,
            "pixcorr_delta_low_minus_zero": low_pix - zero_pix,
        }
    )
    preserved_image_worse_brain = df[
        (df["image_rank_delta_low_minus_zero"] <= 0)
        & (df["brain_rank_delta_low_minus_zero"] > 0)
    ]
    broad_brain_worse_frac = float((df["brain_rank_delta_low_minus_zero"] > 0).mean())
    summary = {
        "subject": subj,
        "rows": ROWS[subj],
        "num_images": int(len(df)),
        "brain_rank_delta_mean": float(df["brain_rank_delta_low_minus_zero"].mean()),
        "brain_rank_delta_median": float(df["brain_rank_delta_low_minus_zero"].median()),
        "brain_rank_worse_fraction": broad_brain_worse_frac,
        "brain_rank_improved_fraction": float((df["brain_rank_delta_low_minus_zero"] < 0).mean()),
        "image_rank_delta_mean": float(df["image_rank_delta_low_minus_zero"].mean()),
        "image_rank_delta_median": float(df["image_rank_delta_low_minus_zero"].median()),
        "image_preserved_while_brain_worse_count": int(len(preserved_image_worse_brain)),
        "teacher_top1_rank_delta_mean": float(df["teacher_top1_rank_delta_low_minus_zero"].mean()),
        "teacher_overlap5_delta_mean": float(df["teacher_neighbor_overlap5_delta_low_minus_zero"].mean()),
        "pixcorr_delta_mean": float(df["pixcorr_delta_low_minus_zero"].mean()),
        "worst_brain_regressions": df.sort_values("brain_rank_delta_low_minus_zero", ascending=False)
        .head(topn)
        .to_dict(orient="records"),
        "largest_brain_improvements": df.sort_values("brain_rank_delta_low_minus_zero", ascending=True)
        .head(topn)
        .to_dict(orient="records"),
        "image_preserved_brain_worse": preserved_image_worse_brain.sort_values(
            "brain_rank_delta_low_minus_zero", ascending=False
        )
        .head(topn)
        .to_dict(orient="records"),
    }
    csv_path = os.path.join(outdir, f"subj{subj:02d}_topo_low_vs_topo0_per_image.csv")
    df.to_csv(csv_path, index=False)
    return summary, csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/src")
    parser.add_argument("--outdir", default="/src/tables/cycle38_neighbor_diagnostics")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--topn", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_path = os.path.join(args.outdir, "all_images_openclip_bigG_flat_norm.pt")
    teacher = get_teacher_features(args.data_path, cache_path, args.batch_size, device)

    summaries = []
    csv_paths = {}
    for subj in (5, 7):
        summary, csv_path = summarize_subject(args.data_path, args.outdir, subj, teacher, args.topn)
        summaries.append(summary)
        csv_paths[f"subj{subj:02d}"] = csv_path

    out = {
        "provenance": {
            "cycle36_rows": ROWS,
            "training_loss_sources_used": [],
            "diagnostic_sources": [
                "evals/all_images.pt",
                "evals/<cycle36_model>/<cycle36_model>_all_clipvoxels.pt",
                "evals/<cycle36_model>/<cycle36_model>_all_enhancedrecons.pt",
                "evals/<cycle36_model>/<cycle36_model>_all_blurryrecons.pt",
                "wds/subj0{subj}/new_test/0.tar image identifiers",
            ],
            "note": "Diagnostics read held-out evaluator artifacts only for post-hoc analysis; no diagnostic output is used by training.",
        },
        "csv_paths": csv_paths,
        "subjects": summaries,
    }
    json_path = os.path.join(args.outdir, "cycle38_neighbor_diagnostics_summary.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2)[:20000])
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
