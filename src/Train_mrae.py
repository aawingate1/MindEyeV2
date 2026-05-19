#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import random
import io
import tarfile
import numpy as np
import h5py
from tqdm import tqdm
import webdataset as wds

import torch
import torch.nn as nn
from accelerate import Accelerator

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append("generative_models/")
import sgm  # noqa: F401
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder

import utils
from models_mrae import MRAE2CLIPModel


torch.backends.cuda.matmul.allow_tf32 = True


def _sanitize_wandb_id(run_name: str) -> str:
    out = run_name
    for ch in ["/", ":", ";", ",", "#", "?", "'", '"', "\\", " "]:
        out = out.replace(ch, "_")
    return out


def _collect_run_ids_from_tar_files(tar_files, n_trials):
    run_ids = np.zeros(n_trials, dtype=np.int64)
    session_ids = np.zeros(n_trials, dtype=np.int64)

    for tar_path in tar_files:
        with tarfile.open(tar_path, "r") as tar:
            for member_name in tar.getnames():
                if not member_name.endswith(".behav.npy"):
                    continue
                with tar.extractfile(member_name) as fobj:
                    behav = np.load(io.BytesIO(fobj.read()))[0]
                global_trial = int(behav[5])
                sess = int(behav[2])
                run = int(behav[3])
                session_ids[global_trial] = sess
                run_ids[global_trial] = (sess - 1) * 12 + run

    return run_ids, session_ids


def _zscore_within_run(betas: np.ndarray, run_ids: np.ndarray) -> np.ndarray:
    betas = betas.copy()
    unique_runs = np.unique(run_ids[run_ids > 0])
    for rid in unique_runs:
        mask = run_ids == rid
        run_data = betas[mask]
        run_mean = run_data.mean(axis=0, keepdims=True)
        run_std = run_data.std(axis=0, keepdims=True)
        run_std[run_std == 0] = 1.0
        betas[mask] = (run_data - run_mean) / run_std
    return betas


### Multi-GPU config ###
local_rank = os.getenv("RANK")
if local_rank is None:
    local_rank = 0
else:
    local_rank = int(local_rank)
print("LOCAL RANK ", local_rank)

data_type = torch.float16
num_devices = torch.cuda.device_count()
if num_devices == 0:
    num_devices = 1

accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
if utils.is_interactive():
    global_batch_size = batch_size = 8
else:
    global_batch_size = os.environ["GLOBAL_BATCH_SIZE"]
    batch_size = int(os.environ["GLOBAL_BATCH_SIZE"]) // num_devices

print("PID of this process =", os.getpid())
device = accelerator.device
print("device:", device)
world_size = accelerator.state.num_processes
distributed = not accelerator.state.distributed_type == "NO"
num_devices = torch.cuda.device_count()
if num_devices == 0 or not distributed:
    num_devices = 1
num_workers = num_devices
print(accelerator.state)

print(
    "distributed =",
    distributed,
    "num_devices =",
    num_devices,
    "local rank =",
    local_rank,
    "world size =",
    world_size,
    "data_type =",
    data_type,
)
print = accelerator.print


if utils.is_interactive():
    model_name = "mrae_debug"
    print("model_name:", model_name)
    jupyter_args = f"--data_path=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src \
                    --cache_dir=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src \
                    --model_name={model_name} \
                    --no-multi_subject --subj=1 --batch_size={batch_size} --num_sessions=2 \
                    --clip_scale=1. --no-blurry_recon --blur_scale=.5 \
                    --no-use_prior --prior_scale=30 \
                    --max_lr=3e-4 --mixup_pct=.33 --num_epochs=2 --no-use_image_aug \
                    --mlp_dim1=512 --mlp_dim2=512 --mlp_dim3=512 --zscore_within_run \
                    --ckpt_interval=999 --no-ckpt_saving --no-wandb_log"
    print(jupyter_args)
    jupyter_args = jupyter_args.split()


parser = argparse.ArgumentParser(description="MRAE-backed MindEyeV2 training")
parser.add_argument(
    "--model_name", type=str, default="mrae/subj01_mrae2clip",
    help="name of model, used for ckpt saving and wandb logging (if enabled)",
)
parser.add_argument(
    "--data_path", type=str, default=os.getcwd(),
    help="Path to where NSD data is stored / where to download it to",
)
parser.add_argument(
    "--cache_dir", type=str, default=os.getcwd(),
    help="Path to where misc. files downloaded from huggingface are stored. Defaults to current src directory.",
)
parser.add_argument(
    "--subj", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8],
    help="Validate on which subject?",
)
parser.add_argument(
    "--multisubject_ckpt", type=str, default=None,
    help="Path to pre-trained multisubject model to finetune a single subject from. multisubject must be False.",
)
parser.add_argument(
    "--num_sessions", type=int, default=1,
    help="Number of training sessions to include",
)
parser.add_argument(
    "--use_prior", action=argparse.BooleanOptionalAction, default=False,
    help="whether to train diffusion prior (True) or just rely on retrieval part of the pipeline (False)",
)
parser.add_argument(
    "--batch_size", type=int, default=16,
    help="Batch size can be increased by 10x if only training retreival submodule and not diffusion prior",
)
parser.add_argument(
    "--wandb_log", action=argparse.BooleanOptionalAction, default=False,
    help="whether to log to wandb",
)
parser.add_argument(
    "--wandb_project", type=str, default="mrae_mindeye",
    help="wandb project name",
)
parser.add_argument(
    "--mixup_pct", type=float, default=.33,
    help="proportion of way through training when to switch from BiMixCo to SoftCLIP",
)
parser.add_argument(
    "--blurry_recon", action=argparse.BooleanOptionalAction, default=False,
    help="whether to output blurry reconstructions",
)
parser.add_argument(
    "--blur_scale", type=float, default=.5,
    help="multiply loss from blurry recons by this number",
)
parser.add_argument(
    "--clip_scale", type=float, default=1.,
    help="multiply contrastive loss by this number",
)
parser.add_argument(
    "--prior_scale", type=float, default=30,
    help="multiply diffusion prior loss by this",
)
parser.add_argument(
    "--use_image_aug", action=argparse.BooleanOptionalAction, default=False,
    help="whether to use image augmentation",
)
parser.add_argument(
    "--num_epochs", type=int, default=150,
    help="number of epochs of training",
)
parser.add_argument(
    "--multi_subject", action=argparse.BooleanOptionalAction, default=False,
)
parser.add_argument(
    "--new_test", action=argparse.BooleanOptionalAction, default=True,
)
parser.add_argument(
    "--n_blocks", type=int, default=4,
)
parser.add_argument(
    "--hidden_dim", type=int, default=1024,
)
parser.add_argument(
    "--lr_scheduler_type", type=str, default="cycle", choices=["cycle", "linear"],
)
parser.add_argument(
    "--ckpt_saving", action=argparse.BooleanOptionalAction, default=True,
)
parser.add_argument(
    "--ckpt_interval", type=int, default=5,
    help="save backup ckpt every x epochs",
)
parser.add_argument(
    "--seed", type=int, default=42,
)
parser.add_argument(
    "--max_lr", type=float, default=3e-4,
)
parser.add_argument(
    "--mrae_ckpt_path", type=str, required=True,
    help="Path to the frozen MRAE checkpoint to load for voxel preprocessing/encoding.",
)
parser.add_argument(
    "--mlp_dim1", type=int, default=512,
    help="Hidden size of the first learned MLP layer after the frozen MRAE encoder.",
)
parser.add_argument(
    "--mlp_dim2", type=int, default=512,
    help="Hidden size of the second learned MLP layer after the frozen MRAE encoder.",
)
parser.add_argument(
    "--mlp_dim3", type=int, default=512,
    help="Hidden size of the third learned MLP layer after the frozen MRAE encoder.",
)
parser.add_argument(
    "--mlp_dropout", type=float, default=0.1,
    help="Dropout probability for the learned MLP head.",
)
parser.add_argument(
    "--clip_seq_dim", type=int, default=256,
    help="Number of CLIP tokens to predict.",
)
parser.add_argument(
    "--clip_emb_dim", type=int, default=1664,
    help="Embedding dimension of each predicted CLIP token.",
)
parser.add_argument(
    "--zscore_within_run", action=argparse.BooleanOptionalAction, default=True,
    help="Apply extra within-run z-scoring to betas before MindEye training.",
)

if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()

for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)

if multi_subject:
    raise ValueError("Train_mrae.py currently supports single-subject only. Use --no-multi_subject.")
if subj != 1:
    raise ValueError("Train_mrae.py is currently scoped to subject 1 only (--subj=1).")
if multisubject_ckpt is not None:
    raise ValueError("No fine-tuning stage in this path. Leave --multisubject_ckpt unset.")
if use_prior:
    raise ValueError("This MRAE path is clip-only. Use --no-use_prior.")
if blurry_recon:
    raise ValueError("This MRAE path disables blurry reconstruction. Use --no-blurry_recon.")
if use_image_aug:
    raise ValueError("This MRAE path disables image augmentation. Use --no-use_image_aug.")

utils.seed_everything(seed)

outdir = os.path.abspath(f"../train_logs/{model_name}")
if not os.path.exists(outdir) and ckpt_saving:
    os.makedirs(outdir, exist_ok=True)

if multi_subject:
    subj_list = np.arange(1, 9)
    subj_list = subj_list[subj_list != subj]
else:
    subj_list = [subj]

print("subj_list", subj_list, "num_sessions", num_sessions)
print("Using MRAE checkpoint from", mrae_ckpt_path)


def my_split_by_node(urls):
    return urls


num_voxels_list = []

if multi_subject:
    nsessions_allsubj = np.array([40, 40, 32, 30, 40, 32, 40, 30])
    num_samples_per_epoch = (750 * 40) // num_devices
else:
    num_samples_per_epoch = (750 * num_sessions) // num_devices

print("dividing batch size by subj_list, which will then be concatenated across subj during training...")
batch_size = batch_size // len(subj_list)
num_iterations_per_epoch = num_samples_per_epoch // (batch_size * len(subj_list))
print("batch_size =", batch_size, "num_iterations_per_epoch =", num_iterations_per_epoch, "num_samples_per_epoch =", num_samples_per_epoch)

train_data = {}
train_dl = {}
num_voxels = {}
voxels = {}

subject_wds_dir = os.path.join(data_path, "wds", f"subj0{subj}")
train_tar_files = [os.path.join(subject_wds_dir, "train", f"{i}.tar") for i in range(num_sessions)]
eval_tar_files = [os.path.join(subject_wds_dir, "new_test" if new_test else "test", "0.tar")]

for tar_path in train_tar_files + eval_tar_files:
    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"Missing required WDS shard: {tar_path}")

for s in subj_list:
    print(f"Training with {num_sessions} sessions")
    if multi_subject:
        train_url = f"{data_path}/wds/subj0{s}/train/" + "{0.." + f"{nsessions_allsubj[s-1]-1}" + "}.tar"
    else:
        train_url = f"{data_path}/wds/subj0{s}/train/" + "{0.." + f"{num_sessions-1}" + "}.tar"
    print(train_url)

    train_data[f"subj0{s}"] = wds.WebDataset(train_url, resampled=True, nodesplitter=my_split_by_node) \
                        .shuffle(750, initial=1500, rng=random.Random(42)) \
                        .decode("torch") \
                        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy") \
                        .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
    train_dl[f"subj0{s}"] = torch.utils.data.DataLoader(
        train_data[f"subj0{s}"],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    with h5py.File(f"{data_path}/betas_all_subj0{s}_fp32_renorm.hdf5", "r") as f:
        betas = f["betas"][:]

    if zscore_within_run:
        if s != subj:
            raise ValueError("Within-run z-scoring is only implemented for the validated subject in this script.")
        print("Applying extra within-run z-scoring to betas to match MRAE preprocessing...")
        run_ids, _ = _collect_run_ids_from_tar_files(train_tar_files + eval_tar_files, betas.shape[0])
        betas = _zscore_within_run(betas, run_ids)
        print(f"Betas z-scored within run (mean={betas.mean():.4f}, std={betas.std():.4f})")
    else:
        print("Skipping extra within-run z-scoring to match MindEye baseline preprocessing.")

    betas = torch.Tensor(betas).to("cpu").to(data_type)
    num_voxels_list.append(betas[0].shape[-1])
    num_voxels[f"subj0{s}"] = betas[0].shape[-1]
    voxels[f"subj0{s}"] = betas
    print(f"num_voxels for subj0{s}: {num_voxels[f'subj0{s}']}")

print("Loaded all subj train dls and betas!\n")

if multi_subject:
    subj = subj_list[0]
if not new_test:
    if subj == 3:
        num_test = 2113
    elif subj == 4:
        num_test = 1985
    elif subj == 6:
        num_test = 2113
    elif subj == 8:
        num_test = 1985
    else:
        num_test = 2770
    test_url = f"{data_path}/wds/subj0{subj}/test/" + "0.tar"
elif new_test:
    if subj == 3:
        num_test = 2371
    elif subj == 4:
        num_test = 2188
    elif subj == 6:
        num_test = 2371
    elif subj == 8:
        num_test = 2188
    else:
        num_test = 3000
    test_url = f"{data_path}/wds/subj0{subj}/new_test/" + "0.tar"
print(test_url)
test_data = wds.WebDataset(test_url, resampled=False, nodesplitter=my_split_by_node) \
                    .shuffle(750, initial=1500, rng=random.Random(42)) \
                    .decode("torch") \
                    .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy") \
                    .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=True)
print(f"Loaded test dl for subj{subj}!\n")

images_h5 = h5py.File(f"{data_path}/coco_images_224_float16.hdf5", "r")
images = images_h5["images"]
print("Loaded all 73k possible NSD images to cpu!", images.shape)


clip_img_embedder = FrozenOpenCLIPImageEmbedder(
    arch="ViT-bigG-14",
    version="laion2b_s39b_b160k",
    output_tokens=True,
    only_tokens=True,
)
clip_img_embedder.to(device)


class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()

    def forward(self, x):
        return x


model = MindEyeModule()
model.backbone = MRAE2CLIPModel(
    mrae_ckpt_path=mrae_ckpt_path,
    mlp_dim1=mlp_dim1,
    mlp_dim2=mlp_dim2,
    mlp_dim3=mlp_dim3,
    clip_seq_dim=clip_seq_dim,
    clip_emb_dim=clip_emb_dim,
    mlp_dropout=mlp_dropout,
)
utils.count_params(model.backbone)
utils.count_params(model)

b = torch.randn((2, 1, num_voxels_list[0]))
backbone_, clip_ = model.backbone(b)
print("sanity shapes:", backbone_.shape, clip_.shape)


no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
head_named_parameters = list(model.backbone.head.named_parameters())
opt_grouped_parameters = [
    {"params": [p for n, p in head_named_parameters if not any(nd in n for nd in no_decay)], "weight_decay": 1e-2},
    {"params": [p for n, p in head_named_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]
optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)

if lr_scheduler_type == "linear":
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        total_iters=int(np.floor(num_epochs * num_iterations_per_epoch)),
        last_epoch=-1,
    )
elif lr_scheduler_type == "cycle":
    total_steps = int(np.floor(num_epochs * num_iterations_per_epoch))
    print("total_steps", total_steps)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        final_div_factor=1000,
        last_epoch=-1,
        pct_start=2 / num_epochs,
    )


def save_ckpt(tag):
    ckpt_path = outdir + f"/{tag}.pth"
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": unwrapped_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "train_losses": losses,
                "test_losses": test_losses,
                "lrs": lrs,
            },
            ckpt_path,
        )
    print(f"\n---saved {outdir}/{tag} ckpt!---\n")


def load_ckpt(tag, load_lr=True, load_optimizer=True, load_epoch=True, strict=True, outdir=outdir):
    ckpt_file = os.path.join(outdir, f"{tag}.pth")
    print(f"\n---loading {ckpt_file} ckpt---\n")
    checkpoint = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if load_epoch:
        globals()["epoch"] = checkpoint["epoch"]
        print("Epoch", epoch)
    if load_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if load_lr:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    del checkpoint


print("\nDone with model preparations!")
num_params = utils.count_params(model)


if local_rank == 0 and wandb_log:
    import wandb
    print(f"wandb {wandb_project} run {model_name}")
    wandb_config = {
      "model_name": model_name,
      "global_batch_size": global_batch_size,
      "batch_size": batch_size,
      "num_epochs": num_epochs,
      "num_sessions": num_sessions,
      "num_params": num_params,
      "clip_scale": clip_scale,
      "prior_scale": prior_scale,
      "blur_scale": blur_scale,
      "use_image_aug": use_image_aug,
      "max_lr": max_lr,
      "mixup_pct": mixup_pct,
      "num_samples_per_epoch": num_samples_per_epoch,
      "num_test": num_test,
      "ckpt_interval": ckpt_interval,
      "ckpt_saving": ckpt_saving,
      "seed": seed,
      "distributed": distributed,
      "num_devices": num_devices,
      "world_size": world_size,
      "train_url": train_url,
      "test_url": test_url,
      "mrae_ckpt_path": mrae_ckpt_path,
      "mlp_dim1": mlp_dim1,
      "mlp_dim2": mlp_dim2,
      "mlp_dim3": mlp_dim3,
      "mlp_dropout": mlp_dropout,
      "clip_seq_dim": clip_seq_dim,
      "clip_emb_dim": clip_emb_dim,
      "zscore_within_run": zscore_within_run,
    }
    print("wandb_config:\n", wandb_config)
    wandb_run_id = _sanitize_wandb_id(model_name)
    print("wandb_id:", wandb_run_id)
    wandb.init(
        id=wandb_run_id,
        project=wandb_project,
        name=model_name,
        config=wandb_config,
        resume="allow",
    )
else:
    wandb_log = False


epoch = 0
losses, test_losses, lrs = [], [], []
torch.cuda.empty_cache()

train_dls = [train_dl[f"subj0{s}"] for s in subj_list]
model, optimizer, *train_dls, lr_scheduler = accelerator.prepare(model, optimizer, *train_dls, lr_scheduler)

print(f"{model_name} starting with epoch {epoch} / {num_epochs}")
progress_bar = tqdm(range(epoch, num_epochs), ncols=1200, disable=(local_rank != 0))
test_image, test_voxel = None, None
soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))

for epoch in progress_bar:
    model.train()

    fwd_percent_correct = 0.
    bwd_percent_correct = 0.
    test_fwd_percent_correct = 0.
    test_bwd_percent_correct = 0.

    recon_cossim = 0.
    test_recon_cossim = 0.
    recon_mse = 0.
    test_recon_mse = 0.

    loss_clip_total = 0.
    loss_blurry_total = 0.
    loss_blurry_cont_total = 0.
    test_loss_clip_total = 0.

    loss_prior_total = 0.
    test_loss_prior_total = 0.

    blurry_pixcorr = 0.
    test_blurry_pixcorr = 0.

    voxel_iters = {}
    image_iters = torch.zeros(num_iterations_per_epoch, batch_size * len(subj_list), 3, 224, 224).float()
    perm_iters, betas_iters, select_iters = {}, {}, {}
    for s, train_dl in enumerate(train_dls):
        with torch.cuda.amp.autocast(dtype=data_type):
            iter_idx = -1
            for behav0, past_behav0, future_behav0, old_behav0 in train_dl:
                image_idx = behav0[:, 0, 0].cpu().long().numpy()
                image0, image_sorted_idx = np.unique(image_idx, return_index=True)
                if len(image0) != len(image_idx):
                    continue
                iter_idx += 1
                image0 = torch.tensor(images[image0], dtype=data_type)
                image_iters[iter_idx, s * batch_size:s * batch_size + batch_size] = image0

                voxel_idx = behav0[:, 0, 5].cpu().long().numpy()
                voxel_sorted_idx = voxel_idx[image_sorted_idx]
                voxel0 = voxels[f"subj0{subj_list[s]}"][voxel_sorted_idx]
                voxel0 = torch.Tensor(voxel0).unsqueeze(1)

                if epoch < int(mixup_pct * num_epochs):
                    voxel0, perm, betas, select = utils.mixco(voxel0)
                    perm_iters[f"subj0{subj_list[s]}_iter{iter_idx}"] = perm
                    betas_iters[f"subj0{subj_list[s]}_iter{iter_idx}"] = betas
                    select_iters[f"subj0{subj_list[s]}_iter{iter_idx}"] = select

                voxel_iters[f"subj0{subj_list[s]}_iter{iter_idx}"] = voxel0

                if iter_idx >= num_iterations_per_epoch - 1:
                    break

    for train_i in range(num_iterations_per_epoch):
        with torch.cuda.amp.autocast(dtype=data_type):
            optimizer.zero_grad()
            loss = 0.

            voxel_list = [voxel_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
            image = image_iters[train_i].detach()
            image = image.to(device)

            clip_target = clip_img_embedder(image)
            assert not torch.any(torch.isnan(clip_target))

            if epoch < int(mixup_pct * num_epochs):
                perm_list = [perm_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
                perm = torch.cat(perm_list, dim=0)
                betas_list = [betas_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
                betas = torch.cat(betas_list, dim=0)
                select_list = [select_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
                select = torch.cat(select_list, dim=0)

            backbone_list = []
            clip_voxels_list = []
            for si, s in enumerate(subj_list):
                backbone_si, clip_voxels_si = model.backbone(voxel_list[si])
                backbone_list.append(backbone_si)
                clip_voxels_list.append(clip_voxels_si)
            backbone = torch.cat(backbone_list, dim=0)
            clip_voxels = torch.cat(clip_voxels_list, dim=0)

            if clip_scale > 0:
                clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

            if clip_scale > 0:
                if epoch < int(mixup_pct * num_epochs):
                    loss_clip = utils.mixco_nce(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=.006,
                        perm=perm, betas=betas, select=select)
                else:
                    epoch_temp = soft_loss_temps[epoch - int(mixup_pct * num_epochs)]
                    loss_clip = utils.soft_clip_loss(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=epoch_temp)

                loss_clip_total += loss_clip.item()
                loss_clip *= clip_scale
                loss += loss_clip

            if clip_scale > 0:
                labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device)
                fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()

            utils.check_loss(loss)
            accelerator.backward(loss)
            optimizer.step()

            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]["lr"])

            if lr_scheduler_type is not None:
                lr_scheduler.step()

    model.eval()
    if local_rank == 0:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=data_type):
            for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):
                assert len(behav) == num_test

                if test_image is None:
                    voxel = voxels[f"subj0{subj}"][behav[:, 0, 5].cpu().long()].unsqueeze(1)
                    image = behav[:, 0, 0].cpu().long()

                    unique_image, sort_indices = torch.unique(image, return_inverse=True)
                    for im in unique_image:
                        locs = torch.where(im == image)[0]
                        if len(locs) == 1:
                            locs = locs.repeat(3)
                        elif len(locs) == 2:
                            locs = locs.repeat(2)[:3]
                        assert len(locs) == 3
                        if test_image is None:
                            test_image = torch.Tensor(images[im][None])
                            test_voxel = voxel[locs][None]
                        else:
                            test_image = torch.vstack((test_image, torch.Tensor(images[im][None])))
                            test_voxel = torch.vstack((test_voxel, voxel[locs][None]))

                loss = 0.

                test_indices = torch.arange(len(test_voxel))[:300]
                voxel = test_voxel[test_indices].to(device)
                image = test_image[test_indices].to(device)
                assert len(image) == 300

                clip_target = clip_img_embedder(image.float())

                for rep in range(3):
                    backbone0, clip_voxels0 = model.backbone(voxel[:, rep])
                    if rep == 0:
                        clip_voxels = clip_voxels0
                        backbone = backbone0
                    else:
                        clip_voxels += clip_voxels0
                        backbone += backbone0
                clip_voxels /= 3
                backbone /= 3
                _ = backbone

                if clip_scale > 0:
                    clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                    clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

                if clip_scale > 0:
                    loss_clip = utils.soft_clip_loss(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=.006)

                    test_loss_clip_total += loss_clip.item()
                    loss_clip = loss_clip * clip_scale
                    loss += loss_clip

                if clip_scale > 0:
                    labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device)
                    test_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                    test_bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()

                utils.check_loss(loss)
                test_losses.append(loss.item())

            assert (test_i + 1) == 1
            logs = {"train/loss": np.mean(losses[-(train_i + 1):]),
                "test/loss": np.mean(test_losses[-(test_i + 1):]),
                "train/lr": lrs[-1],
                "train/num_steps": len(losses),
                "test/num_steps": len(test_losses),
                "train/fwd_pct_correct": fwd_percent_correct / (train_i + 1),
                "train/bwd_pct_correct": bwd_percent_correct / (train_i + 1),
                "test/test_fwd_pct_correct": test_fwd_percent_correct / (test_i + 1),
                "test/test_bwd_pct_correct": test_bwd_percent_correct / (test_i + 1),
                "train/loss_clip_total": loss_clip_total / (train_i + 1),
                "train/loss_blurry_total": loss_blurry_total / (train_i + 1),
                "train/loss_blurry_cont_total": loss_blurry_cont_total / (train_i + 1),
                "test/loss_clip_total": test_loss_clip_total / (test_i + 1),
                "train/blurry_pixcorr": blurry_pixcorr / (train_i + 1),
                "test/blurry_pixcorr": test_blurry_pixcorr / (test_i + 1),
                "train/recon_cossim": recon_cossim / (train_i + 1),
                "test/recon_cossim": test_recon_cossim / (test_i + 1),
                "train/recon_mse": recon_mse / (train_i + 1),
                "test/recon_mse": test_recon_mse / (test_i + 1),
                "train/loss_prior": loss_prior_total / (train_i + 1),
                "test/loss_prior": test_loss_prior_total / (test_i + 1),
                "mrae/zscore_within_run": float(zscore_within_run),
                }

            progress_bar.set_postfix(**logs)

            if wandb_log:
                wandb.log(logs)

    if ckpt_saving and (epoch % ckpt_interval == 0):
        save_ckpt("last")

    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()

print("\n===Finished!===\n")
if ckpt_saving:
    save_ckpt("last")

images_h5.close()
