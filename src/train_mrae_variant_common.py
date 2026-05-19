#!/usr/bin/env python

import argparse
import io
import os
import random
import sys
import tarfile

import h5py
import numpy as np
import torch
import torch.nn as nn
import webdataset as wds
from accelerate import Accelerator
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATIVE_MODELS_DIR = os.path.join(THIS_DIR, "generative_models")
if GENERATIVE_MODELS_DIR not in sys.path:
    sys.path.append(GENERATIVE_MODELS_DIR)
import sgm  # noqa: F401
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder

import utils
from wds_3way_utils import count_samples_in_tar_files, discover_train_eval_tar_files


torch.backends.cuda.matmul.allow_tf32 = True


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=os.getcwd())
    parser.add_argument("--cache_dir", type=str, default=os.getcwd())
    parser.add_argument("--wds_root", type=str, default=None)
    parser.add_argument("--train_split", type=str, default=None)
    parser.add_argument("--eval_split", type=str, default=None)
    parser.add_argument("--subj", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--num_sessions", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--wandb_log", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb_project", type=str, default="mrae_mindeye")
    parser.add_argument("--mixup_pct", type=float, default=0.33)
    parser.add_argument("--clip_scale", type=float, default=1.0)
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--new_test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr_scheduler_type", type=str, default="cycle", choices=["cycle", "linear"])
    parser.add_argument("--ckpt_saving", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ckpt_interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--mrae_ckpt_path", type=str, required=True)
    parser.add_argument("--clip_seq_dim", type=int, default=256)
    parser.add_argument("--clip_emb_dim", type=int, default=1664)
    parser.add_argument("--zscore_within_run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--projector_tune_mode", type=str, default="freeze", choices=["freeze", "last", "full"])
    parser.add_argument("--projector_lr_scale", type=float, default=0.1)
    parser.add_argument("--eval_num_images", type=int, default=300)


def _sanitize_wandb_id(run_name: str) -> str:
    out = run_name
    for ch in ["/", ":", ";", ",", "#", "?", "'", '"', "\\", " "]:
        out = out.replace(ch, "_")
    return out


def _collect_run_ids_from_tar_files(tar_files, n_trials):
    run_ids = np.zeros(n_trials, dtype=np.int64)
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
                run_ids[global_trial] = (sess - 1) * 12 + run
    return run_ids


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


def _common_parser(description: str, add_variant_args_fn=None):
    parser = argparse.ArgumentParser(description=description)
    add_common_args(parser)
    if add_variant_args_fn is not None:
        add_variant_args_fn(parser)
    return parser


def run_training(description: str, build_model_fn, add_variant_args_fn=None):
    local_rank = os.getenv("RANK")
    if local_rank is None:
        local_rank = 0
    else:
        local_rank = int(local_rank)
    print("LOCAL RANK ", local_rank)

    accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
    if utils.is_interactive():
        args = _common_parser(description, add_variant_args_fn).parse_args(
            [
                "--model_name=mrae/debug_variant",
                "--data_path=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src",
                "--cache_dir=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src",
                "--subj=1",
                "--num_sessions=2",
                "--batch_size=8",
                "--num_epochs=2",
                "--max_lr=3e-4",
                "--mixup_pct=.33",
                "--mrae_ckpt_path=/scratch/gpfs/KNORMAN/aw1907/MRAE/MRAE/example/mrae_results_mindeye_subj01/mindeye_split_hidden200_manifold100_tauto_mrae_subj01.pt",
                "--no-wandb_log",
                "--no-ckpt_saving",
            ]
        )
        global_batch_size = args.batch_size
    else:
        args = _common_parser(description, add_variant_args_fn).parse_args()
        global_batch_size = int(os.environ["GLOBAL_BATCH_SIZE"])

    data_type = torch.float16
    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        num_devices = 1
    batch_size = global_batch_size // num_devices
    args.batch_size = batch_size

    print("PID of this process =", os.getpid())
    device = accelerator.device
    print("device:", device)
    world_size = accelerator.state.num_processes
    distributed = not accelerator.state.distributed_type == "NO"
    if num_devices == 0 or not distributed:
        num_devices = 1
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
    accel_print = accelerator.print

    utils.seed_everything(args.seed)

    outdir = os.path.abspath(f"../train_logs/{args.model_name}")
    if args.ckpt_saving:
        os.makedirs(outdir, exist_ok=True)

    subj = args.subj
    if subj != 1:
        raise ValueError("These additive MRAE variants are currently scoped to subject 1.")

    wds_root = args.wds_root or os.path.join(args.data_path, "wds")
    train_dir, eval_dir, train_tar_files, eval_tar_files = discover_train_eval_tar_files(
        wds_root=wds_root,
        subject=f"subj0{subj}",
        train_split=args.train_split,
        eval_split=args.eval_split,
        use_new_test=args.new_test,
        num_sessions=args.num_sessions,
    )
    num_train_samples = count_samples_in_tar_files(train_tar_files)
    num_eval_samples = count_samples_in_tar_files(eval_tar_files)
    args.num_sessions = len(train_tar_files)

    num_samples_per_epoch = num_train_samples // num_devices
    num_iterations_per_epoch = num_samples_per_epoch // batch_size
    accel_print("subj_list", [subj], "num_sessions", args.num_sessions)
    accel_print("batch_size =", batch_size, "num_iterations_per_epoch =", num_iterations_per_epoch, "num_samples_per_epoch =", num_samples_per_epoch)
    accel_print("wds_root =", wds_root)
    accel_print("train_dir =", train_dir)
    accel_print("eval_dir =", eval_dir)

    train_data = wds.WebDataset(train_tar_files, resampled=True, nodesplitter=lambda urls: urls) \
        .shuffle(750, initial=1500, rng=random.Random(42)) \
        .decode("torch") \
        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy") \
        .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
    train_dl = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    with h5py.File(f"{args.data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5", "r") as f:
        betas = f["betas"][:]

    if args.zscore_within_run:
        accel_print("Applying extra within-run z-scoring to betas...")
        run_ids = _collect_run_ids_from_tar_files(train_tar_files + eval_tar_files, betas.shape[0])
        betas = _zscore_within_run(betas, run_ids)
        accel_print(f"Betas z-scored within run (mean={betas.mean():.4f}, std={betas.std():.4f})")

    voxels = torch.tensor(betas, dtype=data_type)
    num_voxels = betas.shape[-1]
    accel_print(f"num_voxels for subj0{subj}: {num_voxels}")

    test_data = wds.WebDataset(eval_tar_files, resampled=False, nodesplitter=lambda urls: urls) \
        .shuffle(750, initial=1500, rng=random.Random(42)) \
        .decode("torch") \
        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy") \
        .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_eval_samples, shuffle=False, drop_last=True, pin_memory=True)

    images_h5 = h5py.File(f"{args.data_path}/coco_images_224_float16.hdf5", "r")
    images = images_h5["images"]
    accel_print("Loaded all 73k possible NSD images to cpu!", images.shape)

    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
    )
    clip_img_embedder.to(device)

    model = build_model_fn(args)
    utils.count_params(model)

    dummy = torch.randn((2, 1, num_voxels))
    sanity = model(dummy)
    accel_print("sanity clip_tokens shape:", sanity["clip_tokens"].shape)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    opt_grouped_parameters = []
    for group in model.param_groups(args.max_lr):
        decay_params = []
        no_decay_params = []
        for p in group["params"]:
            if p.ndim == 1:
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        if decay_params:
            opt_grouped_parameters.append({"params": decay_params, "weight_decay": 1e-2, "lr": group["lr"]})
        if no_decay_params:
            opt_grouped_parameters.append({"params": no_decay_params, "weight_decay": 0.0, "lr": group["lr"]})
    optimizer = torch.optim.AdamW(opt_grouped_parameters)

    total_steps = int(np.floor(args.num_epochs * num_iterations_per_epoch))
    if args.lr_scheduler_type == "linear":
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=total_steps, last_epoch=-1)
    else:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[group["lr"] for group in opt_grouped_parameters],
            total_steps=total_steps,
            final_div_factor=1000,
            last_epoch=-1,
            pct_start=2 / args.num_epochs,
        )

    if local_rank == 0 and args.wandb_log:
        import wandb
        wandb.init(
            id=_sanitize_wandb_id(args.model_name),
            project=args.wandb_project,
            name=args.model_name,
            config=vars(args),
            resume="allow",
        )
    else:
        args.wandb_log = False

    model, optimizer, train_dl, lr_scheduler = accelerator.prepare(model, optimizer, train_dl, lr_scheduler)

    def clip_loss(pred_tokens, target_tokens, epoch_idx, perm=None, betas_mix=None, select=None):
        pred_norm = nn.functional.normalize(pred_tokens.flatten(1), dim=-1)
        target_norm = nn.functional.normalize(target_tokens.flatten(1), dim=-1)
        if epoch_idx < int(args.mixup_pct * args.num_epochs) and perm is not None:
            return utils.mixco_nce(pred_norm, target_norm, temp=0.006, perm=perm, betas=betas_mix, select=select)
        soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, args.num_epochs - int(args.mixup_pct * args.num_epochs))
        temp_idx = max(0, epoch_idx - int(args.mixup_pct * args.num_epochs))
        return utils.soft_clip_loss(pred_norm, target_norm, temp=soft_loss_temps[temp_idx])

    losses = []
    test_losses = []
    lrs = []
    test_image = None
    test_voxel = None
    progress_bar = tqdm(range(args.num_epochs), ncols=1200, disable=(local_rank != 0))

    for epoch in progress_bar:
        model.train()
        train_fwd = 0.0
        train_bwd = 0.0
        train_loss_clip_total = 0.0
        train_loss_global_total = 0.0
        voxel_iters = {}
        image_iters = torch.zeros(num_iterations_per_epoch, batch_size, 3, 224, 224).float()
        perm_iters = {}
        beta_iters = {}
        select_iters = {}

        with torch.cuda.amp.autocast(dtype=data_type):
            iter_idx = -1
            for behav0, _, _, _ in train_dl:
                image_idx = behav0[:, 0, 0].cpu().long().numpy()
                image0, image_sorted_idx = np.unique(image_idx, return_index=True)
                if len(image0) != len(image_idx):
                    continue
                iter_idx += 1
                image0 = torch.tensor(images[image0], dtype=data_type)
                image_iters[iter_idx] = image0

                voxel_idx = behav0[:, 0, 5].cpu().long().numpy()
                voxel_sorted_idx = voxel_idx[image_sorted_idx]
                voxel0 = voxels[voxel_sorted_idx].unsqueeze(1)

                if epoch < int(args.mixup_pct * args.num_epochs):
                    voxel0, perm, betas_mix, select = utils.mixco(voxel0)
                    perm_iters[iter_idx] = perm
                    beta_iters[iter_idx] = betas_mix
                    select_iters[iter_idx] = select

                voxel_iters[iter_idx] = voxel0
                if iter_idx >= num_iterations_per_epoch - 1:
                    break

        for train_i in range(num_iterations_per_epoch):
            optimizer.zero_grad()
            voxel = voxel_iters[train_i].detach().to(device)
            image = image_iters[train_i].detach().to(device)
            with torch.cuda.amp.autocast(dtype=data_type):
                clip_target = clip_img_embedder(image)
                outputs = model(voxel)
                clip_tokens = outputs["clip_tokens"]
                loss = args.clip_scale * clip_loss(
                    clip_tokens,
                    clip_target,
                    epoch,
                    perm=perm_iters.get(train_i).to(device) if train_i in perm_iters else None,
                    betas_mix=beta_iters.get(train_i).to(device) if train_i in beta_iters else None,
                    select=select_iters.get(train_i).to(device) if train_i in select_iters else None,
                )
                train_loss_clip_total += loss.item()

                if "aux_clip_tokens" in outputs and getattr(model, "token_aux_scale", 0.0) > 0:
                    aux_loss = clip_loss(outputs["aux_clip_tokens"], clip_target, epoch)
                    loss = loss + model.token_aux_scale * aux_loss

                if "global_pred" in outputs and getattr(model, "global_scale", 0.0) > 0:
                    global_pred = nn.functional.normalize(outputs["global_pred"], dim=-1)
                    global_target = nn.functional.normalize(clip_target.mean(dim=1), dim=-1)
                    global_loss = utils.soft_clip_loss(global_pred, global_target, temp=0.02)
                    train_loss_global_total += global_loss.item()
                    loss = loss + model.global_scale * global_loss

                pred_norm = nn.functional.normalize(clip_tokens.flatten(1), dim=-1)
                target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
                labels = torch.arange(len(pred_norm), device=pred_norm.device)
                train_fwd += utils.topk(utils.batchwise_cosine_similarity(pred_norm, target_norm), labels, k=1).item()
                train_bwd += utils.topk(utils.batchwise_cosine_similarity(target_norm, pred_norm), labels, k=1).item()

            utils.check_loss(loss)
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()

            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]["lr"])

        model.eval()
        test_fwd = 0.0
        test_bwd = 0.0
        test_loss_clip_total = 0.0
        test_loss_global_total = 0.0

        if local_rank == 0:
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=data_type):
                for test_i, (behav, _, _, _) in enumerate(test_dl):
                    if test_image is None:
                        voxel = voxels[behav[:, 0, 5].cpu().long()].unsqueeze(1)
                        image = behav[:, 0, 0].cpu().long()
                        unique_image = torch.unique(image)
                        for im in unique_image:
                            locs = torch.where(im == image)[0]
                            if len(locs) == 1:
                                locs = locs.repeat(3)
                            elif len(locs) == 2:
                                locs = locs.repeat(2)[:3]
                            if test_image is None:
                                test_image = torch.tensor(images[im][None])
                                test_voxel = voxel[locs][None]
                            else:
                                test_image = torch.vstack((test_image, torch.tensor(images[im][None])))
                                test_voxel = torch.vstack((test_voxel, voxel[locs][None]))

                    idx = torch.arange(len(test_voxel))[:args.eval_num_images]
                    voxel = test_voxel[idx].to(device)
                    image = test_image[idx].to(device)
                    clip_target = clip_img_embedder(image.float())

                    summed = None
                    summed_global = None
                    for rep in range(3):
                        rep_outputs = model(voxel[:, rep])
                        rep_tokens = rep_outputs["clip_tokens"]
                        summed = rep_tokens if summed is None else (summed + rep_tokens)
                        if "global_pred" in rep_outputs:
                            rep_global = rep_outputs["global_pred"]
                            summed_global = rep_global if summed_global is None else (summed_global + rep_global)
                    clip_tokens = summed / 3.0
                    outputs = {"clip_tokens": clip_tokens}
                    if summed_global is not None:
                        outputs["global_pred"] = summed_global / 3.0

                    loss = args.clip_scale * utils.soft_clip_loss(
                        nn.functional.normalize(outputs["clip_tokens"].flatten(1), dim=-1),
                        nn.functional.normalize(clip_target.flatten(1), dim=-1),
                        temp=0.006,
                    )
                    test_loss_clip_total += loss.item()
                    if "global_pred" in outputs and getattr(model, "global_scale", 0.0) > 0:
                        global_pred = nn.functional.normalize(outputs["global_pred"], dim=-1)
                        global_target = nn.functional.normalize(clip_target.mean(dim=1), dim=-1)
                        global_loss = utils.soft_clip_loss(global_pred, global_target, temp=0.02)
                        test_loss_global_total += global_loss.item()
                        loss = loss + model.global_scale * global_loss

                    pred_norm = nn.functional.normalize(outputs["clip_tokens"].flatten(1), dim=-1)
                    target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
                    labels = torch.arange(len(pred_norm), device=pred_norm.device)
                    test_fwd += utils.topk(utils.batchwise_cosine_similarity(pred_norm, target_norm), labels, k=1).item()
                    test_bwd += utils.topk(utils.batchwise_cosine_similarity(target_norm, pred_norm), labels, k=1).item()
                    test_losses.append(loss.item())

                logs = {
                    "train/loss": np.mean(losses[-num_iterations_per_epoch:]),
                    "test/loss": np.mean(test_losses[-1:]),
                    "train/lr": lrs[-1],
                    "train/fwd_pct_correct": train_fwd / num_iterations_per_epoch,
                    "train/bwd_pct_correct": train_bwd / num_iterations_per_epoch,
                    "test/test_fwd_pct_correct": test_fwd / (test_i + 1),
                    "test/test_bwd_pct_correct": test_bwd / (test_i + 1),
                    "train/loss_clip_total": train_loss_clip_total / num_iterations_per_epoch,
                    "test/loss_clip_total": test_loss_clip_total / (test_i + 1),
                    "train/loss_global_total": train_loss_global_total / max(1, num_iterations_per_epoch),
                    "test/loss_global_total": test_loss_global_total / max(1, test_i + 1),
                }
                progress_bar.set_postfix(**logs)
                if args.wandb_log:
                    wandb.log(logs)

        if args.ckpt_saving and (epoch % args.ckpt_interval == 0) and accelerator.is_main_process:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_losses": losses,
                    "test_losses": test_losses,
                    "lrs": lrs,
                },
                os.path.join(outdir, "last.pth"),
            )

        accelerator.wait_for_everyone()

    images_h5.close()
