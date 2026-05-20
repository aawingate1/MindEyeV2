#!/usr/bin/env python
# coding: utf-8

# # Import packages & functions

# In[1]:


import os
import sys
import json
import argparse
import numpy as np
import math
import copy
from einops import rearrange
import time
import random
import string
import h5py
from tqdm import tqdm
import webdataset as wds

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder # bigG embedder

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom functions #
import utils


# In[2]:


### Multi-GPU config ###
local_rank = os.getenv('RANK')
if local_rank is None: 
    local_rank = 0
else:
    local_rank = int(local_rank)
print("LOCAL RANK ", local_rank)  

data_type = torch.float16 # change depending on your mixed_precision
num_devices = torch.cuda.device_count()
if num_devices==0: num_devices = 1

# First use "accelerate config" in terminal and setup using deepspeed stage 2 with CPU offloading!
accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
if utils.is_interactive(): # set batch size here if using interactive notebook instead of submitting job
    global_batch_size = batch_size = 8
else:
    global_batch_size = os.environ["GLOBAL_BATCH_SIZE"]
    batch_size = int(os.environ["GLOBAL_BATCH_SIZE"]) // num_devices


# In[3]:


print("PID of this process =",os.getpid())
device = accelerator.device
print("device:",device)
world_size = accelerator.state.num_processes
distributed = not accelerator.state.distributed_type == 'NO'
num_devices = torch.cuda.device_count()
if num_devices==0 or not distributed: num_devices = 1
num_workers = num_devices
print(accelerator.state)

print("distributed =",distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size, "data_type =", data_type)
print = accelerator.print # only print if local_rank=0


# # Configurations

# In[4]:


# if running this interactively, can specify jupyter_args here for argparser to use
if utils.is_interactive():
    model_name = "testing"
    print("model_name:", model_name)

    # global_batch_size and batch_size should already be defined in the 2nd cell block
    jupyter_args = f"--data_path=/weka/proj-medarc/shared/mindeyev2_dataset \
                    --cache_dir=/weka/proj-medarc/shared/cache \
                    --model_name={model_name} \
                    --no-multi_subject --subj=1 --batch_size={batch_size} --num_sessions=40 \
                    --hidden_dim=1024 --clip_scale=1. \
                    --no-blurry_recon --blur_scale=.5  \
                    --use_prior --prior_scale=30 \
                    --n_blocks=4 --max_lr=3e-4 --mixup_pct=.33 --num_epochs=150 --no-use_image_aug \
                    --ckpt_interval=999 --no-ckpt_saving --no-wandb_log"
    # --multisubject_ckpt=../train_logs/multisubject_subj01_1024_24bs_nolow

    print(jupyter_args)
    jupyter_args = jupyter_args.split()

    from IPython.display import clear_output # function to clear print outputs in cell
    get_ipython().run_line_magic('load_ext', 'autoreload')
    # this allows you to change functions in models.py or utils.py and have this notebook automatically update with your revisions
    get_ipython().run_line_magic('autoreload', '2')


# In[5]:


parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="testing",
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
    "--subj",type=int, default=1, choices=[1,2,3,4,5,6,7,8],
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
    "--use_prior",action=argparse.BooleanOptionalAction,default=True,
    help="whether to train diffusion prior (True) or just rely on retrieval part of the pipeline (False)",
)
parser.add_argument(
    "--batch_size", type=int, default=16,
    help="Batch size can be increased by 10x if only training retreival submodule and not diffusion prior",
)
parser.add_argument(
    "--wandb_log",action=argparse.BooleanOptionalAction,default=False,
    help="whether to log to wandb",
)
parser.add_argument(
    "--wandb_project",type=str,default="stability",
    help="wandb project name",
)
parser.add_argument(
    "--mixup_pct",type=float,default=.33,
    help="proportion of way through training when to switch from BiMixCo to SoftCLIP",
)
parser.add_argument(
    "--blurry_recon",action=argparse.BooleanOptionalAction,default=True,
    help="whether to output blurry reconstructions",
)
parser.add_argument(
    "--blur_scale",type=float,default=.5,
    help="multiply loss from blurry recons by this number",
)
parser.add_argument(
    "--clip_scale",type=float,default=1.,
    help="multiply contrastive loss by this number",
)
parser.add_argument(
    "--prior_scale",type=float,default=30,
    help="multiply diffusion prior loss by this",
)
parser.add_argument(
    "--use_image_aug",action=argparse.BooleanOptionalAction,default=False,
    help="whether to use image augmentation",
)
parser.add_argument(
    "--num_epochs",type=int,default=150,
    help="number of epochs of training",
)
parser.add_argument(
    "--multi_subject",action=argparse.BooleanOptionalAction,default=False,
)
parser.add_argument(
    "--new_test",action=argparse.BooleanOptionalAction,default=True,
)
parser.add_argument(
    "--n_blocks",type=int,default=4,
)
parser.add_argument(
    "--hidden_dim",type=int,default=1024,
)
parser.add_argument(
    "--lr_scheduler_type",type=str,default='cycle',choices=['cycle','linear'],
)
parser.add_argument(
    "--ckpt_saving",action=argparse.BooleanOptionalAction,default=True,
)
parser.add_argument(
    "--ckpt_interval",type=int,default=5,
    help="save backup ckpt and reconstruct every x epochs",
)
parser.add_argument(
    "--seed",type=int,default=42,
)
parser.add_argument(
    "--max_lr",type=float,default=3e-4,
)
parser.add_argument(
    "--reliability_mode", type=str, default="none", choices=["none", "soft_weight", "topk_mask"],
    help="Reliability-aware voxel input mode. Defaults to none to preserve baseline behavior.",
)
parser.add_argument(
    "--reliability_topk", type=int, default=0,
    help="Number of voxels to keep for reliability_mode=topk_mask. 0 keeps the top half.",
)
parser.add_argument(
    "--adapter_prior_weight", type=float, default=0.0,
    help="L2 drift penalty weight for the subject ridge adapter/projection.",
)
parser.add_argument(
    "--adapter_prior_type", type=str, default="weights", choices=["weights", "outputs"],
    help="Whether to regularize ridge adapter weights or ridge adapter outputs toward initialization.",
)
parser.add_argument(
    "--train_scope", type=str, default="all", choices=["all", "adapter_only", "adapter_head"],
    help="Which parameters to train. all preserves baseline behavior; adapter scopes freeze shared capacity.",
)
parser.add_argument(
    "--alignment_output_prior_weight", type=float, default=0.0,
    help="Penalty weight for keeping adapter outputs aligned to their initialized frozen outputs.",
)
parser.add_argument(
    "--alignment_output_prior_type", type=str, default="distill", choices=["distill", "moments"],
    help="Output alignment prior type after the ridge adapter and before the shared mapper.",
)
parser.add_argument(
    "--val_fraction", type=float, default=0.0,
    help="Optional deterministic cached train-shard validation fraction. 0 disables validation.",
)
parser.add_argument(
    "--heldout_val_sessions", type=int, default=0,
    help="Number of train-session tar shards to reserve only for cached validation. 0 uses the original train-shard cache.",
)
parser.add_argument(
    "--heldout_val_start_session", type=int, default=-1,
    help="First train-session tar shard for held-out validation. -1 starts immediately after num_sessions.",
)
parser.add_argument(
    "--heldout_val_max_samples", type=int, default=300,
    help="Maximum unique image/voxel pairs to cache from held-out validation shards.",
)
parser.add_argument(
    "--early_stop_patience", type=int, default=0,
    help="Stop after this many non-improving validation epochs. 0 disables early stopping.",
)
parser.add_argument(
    "--head_lr_scale", type=float, default=1.0,
    help="LR multiplier for backbone_linear and clip_proj when train_scope=adapter_head.",
)
parser.add_argument(
    "--ridge_lr_scale", type=float, default=1.0,
    help="LR multiplier for the subject ridge adapter.",
)
parser.add_argument(
    "--alignment_lora_rank", type=int, default=0,
    help="Rank for near-identity low-rank residual alignment after ridge. 0 disables it.",
)
parser.add_argument(
    "--alignment_lora_alpha", type=float, default=0.0,
    help="Scale numerator for low-rank alignment. 0 means use rank when rank > 0.",
)
parser.add_argument(
    "--alignment_lora_dropout", type=float, default=0.0,
    help="Dropout applied inside the low-rank alignment residual.",
)

if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)

# seed all random functions
utils.seed_everything(seed)

outdir = os.path.abspath(f'../train_logs/{model_name}')
if not os.path.exists(outdir) and ckpt_saving:
    os.makedirs(outdir,exist_ok=True)

if use_image_aug or blurry_recon:
    import kornia
    from kornia.augmentation.container import AugmentationSequential
if use_image_aug:
    img_augment = AugmentationSequential(
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.3),
        same_on_batch=False,
        data_keys=["input"],
    )

if multi_subject:
    subj_list = np.arange(1,9)
    subj_list = subj_list[subj_list != subj]
else:
    subj_list = [subj]

print("subj_list", subj_list, "num_sessions", num_sessions)


# # Prep data, models, and dataloaders

# ### Creating wds dataloader, preload betas and all 73k possible images

# In[6]:


def my_split_by_node(urls): return urls
num_voxels_list = []

if multi_subject:
    nsessions_allsubj=np.array([40, 40, 32, 30, 40, 32, 40, 30])
    num_samples_per_epoch = (750*40) // num_devices 
else:
    num_samples_per_epoch = (750*num_sessions) // num_devices 

print("dividing batch size by subj_list, which will then be concatenated across subj during training...") 
batch_size = batch_size // len(subj_list)

num_iterations_per_epoch = num_samples_per_epoch // (batch_size*len(subj_list))

print("batch_size =", batch_size, "num_iterations_per_epoch =",num_iterations_per_epoch, "num_samples_per_epoch =",num_samples_per_epoch)


# In[7]:


train_data = {}
train_dl = {}
num_voxels = {}
voxels = {}
reliability_inputs = {}
reliability_summaries = {}

def build_reliability_input(betas, mode, topk):
    """Builds a deterministic fallback reliability proxy when NCSNR metadata is unavailable."""
    if mode == "none":
        return None, {
            "mode": mode,
            "source": "disabled",
            "num_voxels": int(betas.shape[-1]),
            "active_voxels": int(betas.shape[-1]),
        }

    # Fallback proxy: voxelwise response variance across available training betas.
    # This is not NCSNR; it is logged as a proxy so results are interpreted accordingly.
    reliability = torch.std(betas.float(), dim=0)
    reliability = torch.nan_to_num(reliability, nan=0.0, posinf=0.0, neginf=0.0)
    reliability = torch.clamp(reliability, min=0.0)
    if float(reliability.max()) == 0.0:
        reliability = torch.ones_like(reliability)

    num_vox = int(reliability.numel())
    active_voxels = num_vox
    if mode == "soft_weight":
        weights = reliability / torch.clamp(reliability.mean(), min=1e-6)
        weights = torch.clamp(weights, 0.25, 4.0).to(data_type).reshape(1, 1, -1)
        transform = weights
    elif mode == "topk_mask":
        active_voxels = int(topk) if int(topk) > 0 else max(1, num_vox // 2)
        active_voxels = min(max(1, active_voxels), num_vox)
        keep_idx = torch.topk(reliability, k=active_voxels).indices
        mask = torch.zeros_like(reliability, dtype=data_type)
        mask[keep_idx] = 1
        transform = mask.reshape(1, 1, -1)
    else:
        raise ValueError(f"Unknown reliability_mode: {mode}")

    quantiles = torch.quantile(reliability, torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
    summary = {
        "mode": mode,
        "source": "voxelwise_train_beta_std_proxy_no_ncsnr_found",
        "num_voxels": num_vox,
        "active_voxels": active_voxels,
        "mean": float(reliability.mean()),
        "std": float(reliability.std()),
        "min": float(quantiles[0]),
        "p10": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p90": float(quantiles[5]),
        "max": float(quantiles[6]),
    }
    return transform, summary

def apply_reliability(voxel, subj_key):
    transform = reliability_inputs.get(subj_key)
    if transform is None:
        return voxel
    return voxel * transform.to(voxel.device, dtype=voxel.dtype)

for s in subj_list:
    print(f"Training with {num_sessions} sessions")
    if multi_subject:
        train_url = f"{data_path}/wds/subj0{s}/train/" + "{0.." + f"{nsessions_allsubj[s-1]-1}" + "}.tar"
    else:
        train_url = f"{data_path}/wds/subj0{s}/train/" + "{0.." + f"{num_sessions-1}" + "}.tar"
    print(train_url)

    train_data[f'subj0{s}'] = wds.WebDataset(train_url,resampled=True,nodesplitter=my_split_by_node)\
                        .shuffle(750, initial=1500, rng=random.Random(42))\
                        .decode("torch")\
                        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                        .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
    train_dl[f'subj0{s}'] = torch.utils.data.DataLoader(train_data[f'subj0{s}'], batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

    f = h5py.File(f'{data_path}/betas_all_subj0{s}_fp32_renorm.hdf5', 'r')
    betas = f['betas'][:]
    betas = torch.Tensor(betas).to("cpu").to(data_type)
    num_voxels_list.append(betas[0].shape[-1])
    num_voxels[f'subj0{s}'] = betas[0].shape[-1]
    voxels[f'subj0{s}'] = betas
    reliability_inputs[f'subj0{s}'], reliability_summaries[f'subj0{s}'] = build_reliability_input(
        betas, reliability_mode, reliability_topk)
    print(f"reliability summary for subj0{s}: {reliability_summaries[f'subj0{s}']}")
    print(f"num_voxels for subj0{s}: {num_voxels[f'subj0{s}']}")

print("Loaded all subj train dls and betas!\n")

if accelerator.is_main_process and ckpt_saving:
    with open(os.path.join(outdir, "reliability_summary.json"), "w") as f:
        json.dump(reliability_summaries, f, indent=2, sort_keys=True)
    print(f"Saved reliability summary to {os.path.join(outdir, 'reliability_summary.json')}")

# Validate only on one subject
if multi_subject: 
    subj = subj_list[0] # cant validate on the actual held out person so picking first in subj_list
if not new_test: # using old test set from before full dataset released (used in original MindEye paper)
    if subj==3:
        num_test=2113
    elif subj==4:
        num_test=1985
    elif subj==6:
        num_test=2113
    elif subj==8:
        num_test=1985
    else:
        num_test=2770
    test_url = f"{data_path}/wds/subj0{subj}/test/" + "0.tar"
elif new_test: # using larger test set from after full dataset released
    if subj==3:
        num_test=2371
    elif subj==4:
        num_test=2188
    elif subj==6:
        num_test=2371
    elif subj==8:
        num_test=2188
    else:
        num_test=3000
    test_url = f"{data_path}/wds/subj0{subj}/new_test/" + "0.tar"
print(test_url)
test_data = wds.WebDataset(test_url,resampled=False,nodesplitter=my_split_by_node)\
                    .shuffle(750, initial=1500, rng=random.Random(42))\
                    .decode("torch")\
                    .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                    .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=True)
print(f"Loaded test dl for subj{subj}!\n")

val_cache = None

def build_val_cache():
    if val_fraction <= 0 and heldout_val_sessions <= 0:
        return None
    if multi_subject:
        print("val_fraction is enabled only for the active evaluation subject cache; multi_subject validation is skipped.")
        return None

    if heldout_val_sessions > 0:
        val_start = num_sessions if heldout_val_start_session < 0 else heldout_val_start_session
        val_end = val_start + heldout_val_sessions - 1
        if val_start < num_sessions:
            raise ValueError(
                "heldout_val_start_session overlaps training sessions: "
                f"start={val_start}, num_sessions={num_sessions}"
            )
        missing = [
            f"{data_path}/wds/subj0{subj}/train/{session_i}.tar"
            for session_i in range(val_start, val_end + 1)
            if not os.path.exists(f"{data_path}/wds/subj0{subj}/train/{session_i}.tar")
        ]
        if missing:
            raise FileNotFoundError(f"held-out validation shard(s) missing: {missing}")
        val_url = f"{data_path}/wds/subj0{subj}/train/" + "{" + f"{val_start}..{val_end}" + "}.tar"
        source_sessions = heldout_val_sessions
        val_source = "heldout_train_sessions"
        val_fraction_for_count = val_fraction if val_fraction > 0 else 1.0
        val_samples = int(round(750 * source_sessions * val_fraction_for_count))
        val_samples = min(heldout_val_max_samples, max(batch_size, val_samples))
    else:
        val_url = f"{data_path}/wds/subj0{subj}/train/" + "{0.." + f"{num_sessions-1}" + "}.tar"
        source_sessions = num_sessions
        val_source = "train_shard_cache"
        val_samples = int(round(750 * source_sessions * val_fraction))
        val_samples = min(300, max(batch_size, val_samples))

    print(f"Building deterministic validation cache source={val_source} from {val_url} with up to {val_samples} samples")
    val_data = wds.WebDataset(val_url, resampled=False, nodesplitter=my_split_by_node)\
                        .shuffle(750, initial=750, rng=random.Random(seed + 1000))\
                        .decode("torch")\
                        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                        .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
    val_dl = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)
    val_images, val_voxels = [], []
    seen_images = set()
    for behav0, past_behav0, future_behav0, old_behav0 in val_dl:
        image_idx = behav0[:,0,0].cpu().long().numpy()
        voxel_idx = behav0[:,0,5].cpu().long().numpy()
        for image_i, voxel_i in zip(image_idx, voxel_idx):
            image_i = int(image_i)
            if image_i in seen_images:
                continue
            seen_images.add(image_i)
            val_images.append(torch.tensor(images[image_i], dtype=data_type))
            voxel = voxels[f'subj0{subj}'][int(voxel_i)].unsqueeze(0).unsqueeze(0)
            val_voxels.append(apply_reliability(voxel, f'subj0{subj}').squeeze(0))
            if len(val_images) >= val_samples:
                break
        if len(val_images) >= val_samples:
            break

    if len(val_images) < 2:
        print("Validation cache skipped because fewer than 2 unique samples were found.")
        return None
    cache = {
        "image": torch.stack(val_images).float(),
        "voxel": torch.stack(val_voxels).to(data_type),
        "source": val_source,
        "url": val_url,
    }
    print(f"Validation cache ready: source={val_source} n={len(val_images)} unique image/voxel pairs")
    return cache


# In[8]:


# Load 73k NSD images
f = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
images = f['images']
print("Loaded all 73k possible NSD images to cpu!", images.shape)


# ## Load models

# ### CLIP image embeddings  model

# In[9]:


clip_img_embedder = FrozenOpenCLIPImageEmbedder(
    arch="ViT-bigG-14",
    version="laion2b_s39b_b160k",
    output_tokens=True,
    only_tokens=True,
)
clip_img_embedder.to(device)

clip_seq_dim = 256
clip_emb_dim = 1664


# ### SD VAE

# In[10]:


if blurry_recon:
    from diffusers import AutoencoderKL    
    autoenc = AutoencoderKL(
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    ckpt = torch.load(f'{cache_dir}/sd_image_var_autoenc.pth')
    autoenc.load_state_dict(ckpt)

    autoenc.eval()
    autoenc.requires_grad_(False)
    autoenc.to(device)
    utils.count_params(autoenc)

    from autoencoder.convnext import ConvnextXL
    cnx = ConvnextXL(f'{cache_dir}/convnext_xlarge_alpha0.75_fullckpt.pth')
    cnx.requires_grad_(False)
    cnx.eval()
    cnx.to(device)

    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).reshape(1,3,1,1)
    std = torch.tensor([0.228, 0.224, 0.225]).to(device).reshape(1,3,1,1)

    blur_augs = AugmentationSequential(
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        kornia.augmentation.RandomGrayscale(p=0.1),
        kornia.augmentation.RandomSolarize(p=0.1),
        kornia.augmentation.RandomResizedCrop((224,224), scale=(.9,.9), ratio=(1,1), p=1.0),
        data_keys=["input"],
    )


# ### MindEye modules

# In[11]:


class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()
    def forward(self, x):
        return x

model = MindEyeModule()
model


# In[12]:


class RidgeRegression(torch.nn.Module):
    # make sure to add weight_decay when initializing optimizer to enable regularization
    def __init__(self, input_sizes, out_features): 
        super(RidgeRegression, self).__init__()
        self.out_features = out_features
        self.linears = torch.nn.ModuleList([
                torch.nn.Linear(input_size, out_features) for input_size in input_sizes
            ])
    def forward(self, x, subj_idx):
        out = self.linears[subj_idx](x[:,0]).unsqueeze(1)
        return out

model.ridge = RidgeRegression(num_voxels_list, out_features=hidden_dim)
utils.count_params(model.ridge)
utils.count_params(model)

class LowRankResidualAlignment(torch.nn.Module):
    def __init__(self, dim, rank=0, alpha=0.0, dropout=0.0):
        super().__init__()
        self.rank = int(rank)
        if self.rank <= 0:
            self.enabled = False
            self.scale = 0.0
            return
        self.enabled = True
        alpha = float(alpha) if float(alpha) > 0 else float(self.rank)
        self.scale = alpha / float(self.rank)
        self.norm = torch.nn.LayerNorm(dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.down = torch.nn.Linear(dim, self.rank, bias=False)
        self.up = torch.nn.Linear(self.rank, dim, bias=False)
        torch.nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.up.weight)

    def forward(self, x):
        if not self.enabled:
            return x
        return x + self.scale * self.up(self.dropout(self.down(self.norm(x))))

model.alignment_lora = LowRankResidualAlignment(
    hidden_dim,
    rank=alignment_lora_rank,
    alpha=alignment_lora_alpha,
    dropout=alignment_lora_dropout,
)
if alignment_lora_rank > 0:
    print(
        f"Initialized alignment_lora: rank={alignment_lora_rank}, "
        f"alpha={alignment_lora_alpha if alignment_lora_alpha > 0 else alignment_lora_rank}, "
        f"scale={model.alignment_lora.scale}, dropout={alignment_lora_dropout}"
    )
utils.count_params(model.alignment_lora)
utils.count_params(model)

# test on subject 1 with fake data
b = torch.randn((2,1,num_voxels_list[0]))
print(b.shape, model.alignment_lora(model.ridge(b,0)).shape)


# In[13]:


from models import BrainNetwork
model.backbone = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, seq_len=1, n_blocks=n_blocks,
                          clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim, 
                          blurry_recon=blurry_recon, clip_scale=clip_scale)
utils.count_params(model.backbone)
utils.count_params(model)

# test that the model works on some fake data
b = torch.randn((2,1,hidden_dim))
print("b.shape",b.shape)

backbone_, clip_, blur_ = model.backbone(b)
print(backbone_.shape, clip_.shape, blur_[0].shape, blur_[1].shape)


# ### Adding diffusion prior + unCLIP if use_prior=True

# In[14]:


if use_prior:
    from models import *

    # setup diffusion prior network
    out_dim = clip_emb_dim
    depth = 6
    dim_head = 52
    heads = clip_emb_dim//52 # heads * dim_head = clip_emb_dim
    timesteps = 100

    prior_network = PriorNetwork(
            dim=out_dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            causal=False,
            num_tokens = clip_seq_dim,
            learned_query_mode="pos_emb"
        )

    model.diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=out_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=0.2,
        image_embed_scale=None,
    )

    utils.count_params(model.diffusion_prior)
    utils.count_params(model)


# ### Setup optimizer / lr / ckpt saving

# In[15]:


def apply_train_scope(scope):
    for p in model.parameters():
        p.requires_grad = True
    if scope == "all":
        pass
    elif scope == "adapter_only":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.ridge.parameters():
            p.requires_grad = True
        for p in model.alignment_lora.parameters():
            p.requires_grad = True
    elif scope == "adapter_head":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.ridge.parameters():
            p.requires_grad = True
        for p in model.alignment_lora.parameters():
            p.requires_grad = True
        for p in model.backbone.backbone_linear.parameters():
            p.requires_grad = True
        for p in model.backbone.clip_proj.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown train_scope: {scope}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"train_scope={scope} total_params={total_params:,} trainable_params={trainable_params:,}")
    return total_params, trainable_params

total_params, trainable_params = apply_train_scope(train_scope)

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'norm.bias', 'norm.weight']

def named_param_group(name, named_params, weight_decay, lr_scale):
    params = [p for n, p in named_params if p.requires_grad]
    return {
        'name': name,
        'params': params,
        'weight_decay': weight_decay,
        'lr': max_lr * lr_scale,
        'max_lr': max_lr * lr_scale,
        'param_count': sum(p.numel() for p in params),
    }

head_scale = head_lr_scale if train_scope == "adapter_head" else 1.0
opt_grouped_parameters = [
    named_param_group('ridge', model.ridge.named_parameters(), 1e-2, ridge_lr_scale),
    named_param_group(
        'alignment_lora_decay',
        [(n, p) for n, p in model.alignment_lora.named_parameters() if not any(nd in n for nd in no_decay)],
        1e-2,
        ridge_lr_scale,
    ),
    named_param_group(
        'alignment_lora_nodecay',
        [(n, p) for n, p in model.alignment_lora.named_parameters() if any(nd in n for nd in no_decay)],
        0.0,
        ridge_lr_scale,
    ),
    named_param_group(
        'backbone_linear',
        model.backbone.backbone_linear.named_parameters(),
        1e-2,
        head_scale,
    ),
    named_param_group(
        'clip_proj_decay',
        [(n, p) for n, p in model.backbone.clip_proj.named_parameters() if not any(nd in n for nd in no_decay)],
        1e-2,
        head_scale,
    ),
    named_param_group(
        'clip_proj_nodecay',
        [(n, p) for n, p in model.backbone.clip_proj.named_parameters() if any(nd in n for nd in no_decay)],
        0.0,
        head_scale,
    ),
    named_param_group(
        'backbone_other_decay',
        [(n, p) for n, p in model.backbone.named_parameters()
         if p.requires_grad
         and not n.startswith('backbone_linear.')
         and not n.startswith('clip_proj.')
         and not any(nd in n for nd in no_decay)],
        1e-2,
        1.0,
    ),
    named_param_group(
        'backbone_other_nodecay',
        [(n, p) for n, p in model.backbone.named_parameters()
         if p.requires_grad
         and not n.startswith('backbone_linear.')
         and not n.startswith('clip_proj.')
         and any(nd in n for nd in no_decay)],
        0.0,
        1.0,
    ),
]
if use_prior:
    opt_grouped_parameters.extend([
        named_param_group(
            'diffusion_prior_decay',
            [(n, p) for n, p in model.diffusion_prior.named_parameters() if not any(nd in n for nd in no_decay)],
            1e-2,
            1.0,
        ),
        named_param_group(
            'diffusion_prior_nodecay',
            [(n, p) for n, p in model.diffusion_prior.named_parameters() if any(nd in n for nd in no_decay)],
            0.0,
            1.0,
        )
    ])
opt_grouped_parameters = [group for group in opt_grouped_parameters if len(group['params']) > 0]
if len(opt_grouped_parameters) == 0:
    raise ValueError(f"train_scope={train_scope} left no trainable parameters")

optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)
print("optimizer parameter groups:")
for group in opt_grouped_parameters:
    print(
        f"  {group['name']}: params={group['param_count']:,} "
        f"weight_decay={group['weight_decay']} lr={group['lr']}"
    )

if lr_scheduler_type == 'linear':
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        total_iters=int(np.floor(num_epochs*num_iterations_per_epoch)),
        last_epoch=-1
    )
elif lr_scheduler_type == 'cycle':
    total_steps=int(np.floor(num_epochs*num_iterations_per_epoch))
    print("total_steps", total_steps)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[group['max_lr'] for group in opt_grouped_parameters],
        total_steps=total_steps,
        final_div_factor=1000,
        last_epoch=-1, pct_start=2/num_epochs
    )

def save_ckpt(tag):
    ckpt_path = outdir+f'/{tag}.pth'
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save({
            'epoch': epoch,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_losses': losses,
            'test_losses': test_losses,
            'lrs': lrs,
            }, ckpt_path)
    print(f"\n---saved {outdir}/{tag} ckpt!---\n")

def load_ckpt(tag,load_lr=True,load_optimizer=True,load_epoch=True,strict=True,outdir=outdir,multisubj_loading=False): 
    print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")
    checkpoint = torch.load(outdir+'/last.pth', map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    if multisubj_loading: # remove incompatible ridge layer that will otherwise error
        state_dict.pop('ridge.linears.0.weight',None)
    model.load_state_dict(state_dict, strict=strict)
    if load_epoch:
        globals()["epoch"] = checkpoint['epoch']
        print("Epoch",epoch)
    if load_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if load_lr:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    del checkpoint

print("\nDone with model preparations!")
num_params = utils.count_params(model)
val_cache = build_val_cache()


# # Weights and Biases

# In[16]:


if local_rank==0 and wandb_log: # only use main process for wandb logging
    import wandb
    wandb_project = 'mindeye'
    print(f"wandb {wandb_project} run {model_name}")
    # need to configure wandb beforehand in terminal with "wandb init"!
    wandb_config = {
      "model_name": model_name,
      "global_batch_size": global_batch_size,
      "batch_size": batch_size,
      "num_epochs": num_epochs,
      "num_sessions": num_sessions,
      "num_params": num_params,
      "total_params": total_params,
      "trainable_params": trainable_params,
      "clip_scale": clip_scale,
      "prior_scale": prior_scale,
      "blur_scale": blur_scale,
      "use_image_aug": use_image_aug,
      "max_lr": max_lr,
      "reliability_mode": reliability_mode,
      "reliability_topk": reliability_topk,
      "adapter_prior_weight": adapter_prior_weight,
      "adapter_prior_type": adapter_prior_type,
      "train_scope": train_scope,
      "alignment_output_prior_weight": alignment_output_prior_weight,
      "alignment_output_prior_type": alignment_output_prior_type,
      "head_lr_scale": head_lr_scale,
      "ridge_lr_scale": ridge_lr_scale,
      "alignment_lora_rank": alignment_lora_rank,
      "alignment_lora_alpha": alignment_lora_alpha,
      "alignment_lora_dropout": alignment_lora_dropout,
      "val_fraction": val_fraction,
      "heldout_val_sessions": heldout_val_sessions,
      "heldout_val_start_session": heldout_val_start_session,
      "heldout_val_max_samples": heldout_val_max_samples,
      "early_stop_patience": early_stop_patience,
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
    }
    print("wandb_config:\n",wandb_config)
    print("wandb_id:",model_name)
    wandb.init(
        id=model_name,
        project=wandb_project,
        name=model_name,
        config=wandb_config,
        resume="allow",
    )
else:
    wandb_log = False


# # Main

# In[17]:


epoch = 0
losses, test_losses, lrs = [], [], []
best_test_loss = 1e9
best_val_loss = 1e9
epochs_since_best_val = 0
torch.cuda.empty_cache()


# In[18]:


# load multisubject stage1 ckpt if set
if multisubject_ckpt is not None:
    load_ckpt("last",outdir=multisubject_ckpt,load_lr=False,load_optimizer=False,load_epoch=False,strict=False,multisubj_loading=True)

adapter_prior_state = None
adapter_prior_ridge = None
if adapter_prior_weight > 0:
    if adapter_prior_type == "weights":
        adapter_prior_state = {n: p.detach().clone() for n, p in model.ridge.named_parameters() if p.requires_grad}
    elif adapter_prior_type == "outputs":
        adapter_prior_ridge = copy.deepcopy(model.ridge).eval()
        adapter_prior_ridge.requires_grad_(False)
    print(f"Initialized adapter prior: type={adapter_prior_type}, weight={adapter_prior_weight}")

alignment_output_prior_ridge = None
if alignment_output_prior_weight > 0:
    alignment_output_prior_ridge = copy.deepcopy(model.ridge).eval()
    alignment_output_prior_ridge.requires_grad_(False)
    print(f"Initialized alignment output prior: type={alignment_output_prior_type}, weight={alignment_output_prior_weight}")


# In[19]:


train_dls = [train_dl[f'subj0{s}'] for s in subj_list]

model, optimizer, *train_dls, lr_scheduler = accelerator.prepare(model, optimizer, *train_dls, lr_scheduler)
# leaving out test_dl since we will only have local_rank 0 device do evals
if adapter_prior_state is not None:
    adapter_prior_state = {n: p.to(device) for n, p in adapter_prior_state.items()}
if adapter_prior_ridge is not None:
    adapter_prior_ridge = adapter_prior_ridge.to(device)
if alignment_output_prior_ridge is not None:
    alignment_output_prior_ridge = alignment_output_prior_ridge.to(device)


# In[20]:


print(f"{model_name} starting with epoch {epoch} / {num_epochs}")
progress_bar = tqdm(range(epoch,num_epochs), ncols=1200, disable=(local_rank!=0))
test_image, test_voxel = None, None
mse = nn.MSELoss()
l1 = nn.L1Loss()
soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))

def alignment_output_prior_loss(current_outputs, reference_outputs):
    if alignment_output_prior_type == "distill":
        return mse(current_outputs, reference_outputs)
    if alignment_output_prior_type == "moments":
        current_flat = current_outputs.flatten(1).float()
        reference_flat = reference_outputs.flatten(1).float()
        current_norm = torch.norm(current_flat, dim=1).mean()
        reference_norm = torch.norm(reference_flat, dim=1).mean()
        return (
            mse(current_flat.mean(dim=0), reference_flat.mean(dim=0)) +
            mse(current_flat.var(dim=0, unbiased=False), reference_flat.var(dim=0, unbiased=False)) +
            mse(current_norm, reference_norm)
        )
    raise ValueError(f"Unknown alignment_output_prior_type: {alignment_output_prior_type}")

def align_ridge_output(ridge_output):
    return model.alignment_lora(ridge_output)

def feature_summary(tensor):
    with torch.cuda.amp.autocast(enabled=False):
        flat = tensor.detach().float().flatten(1)
        centered = flat - flat.mean(dim=0, keepdim=True)
        gram = centered @ centered.t()
        denom = max(1, flat.shape[1] - 1)
        eigvals = torch.linalg.eigvalsh(gram / denom).clamp_min(0)
        probs = eigvals / eigvals.sum().clamp_min(1e-12)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
    return {
        "norm": torch.norm(flat, dim=1).mean().item(),
        "mean": flat.mean().item(),
        "std": flat.std(unbiased=False).item(),
        "effective_rank": torch.exp(entropy).item(),
    }

probe_initial_features = None

def collect_probe_features(cache):
    if cache is None:
        return None
    voxel = cache["voxel"].to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=data_type):
        ridge = model.ridge(voxel, 0)
        aligned = align_ridge_output(ridge)
        backbone, clip_voxels, _ = model.backbone(aligned)
    return {
        "ridge": ridge.detach().float().cpu(),
        "aligned": aligned.detach().float().cpu(),
        "clip_proj": clip_voxels.detach().float().cpu(),
    }

def probe_diagnostic_logs(cache):
    global probe_initial_features
    if cache is None:
        return {}
    current = collect_probe_features(cache)
    if current is None:
        return {}
    if probe_initial_features is None:
        probe_initial_features = {k: v.clone() for k, v in current.items()}
    metrics = {}
    for name, tensor in current.items():
        stats = feature_summary(tensor.to(device))
        for stat_name, value in stats.items():
            metrics[f"diag/{name}_{stat_name}"] = value
        initial = probe_initial_features[name].to(tensor.device, dtype=tensor.dtype)
        metrics[f"diag/{name}_drift_mse"] = torch.mean((tensor - initial) ** 2).item()
    return metrics

def eval_cached_pairs(cache, split_name):
    if cache is None:
        return {}
    voxel = cache["voxel"].to(device)
    image = cache["image"].to(device)
    clip_target = clip_img_embedder(image.float())
    voxel_ridge = align_ridge_output(model.ridge(voxel, 0))
    backbone, clip_voxels, blurry_image_enc_ = model.backbone(voxel_ridge)
    loss = 0.
    metrics = {}
    if use_prior:
        loss_prior, contaminated_prior_out = model.diffusion_prior(text_embed=backbone, image_embed=clip_target)
        loss = loss + loss_prior * prior_scale
        metrics[f"{split_name}/loss_prior"] = loss_prior.item()
    if clip_scale > 0:
        clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
        clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
        loss_clip = utils.soft_clip_loss(clip_voxels_norm, clip_target_norm, temp=.006)
        loss = loss + loss_clip * clip_scale
        labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device)
        metrics[f"{split_name}/fwd_pct_correct"] = utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
        metrics[f"{split_name}/bwd_pct_correct"] = utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()
        metrics[f"{split_name}/loss_clip_total"] = loss_clip.item()
    metrics[f"{split_name}/loss"] = loss.item()
    return metrics

probe_initial_features = collect_probe_features(val_cache)
if probe_initial_features is not None:
    print(f"Initialized fixed-probe diagnostics from {len(probe_initial_features['ridge'])} cached samples")

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
    loss_adapter_prior_total = 0.
    loss_alignment_output_prior_total = 0.
    alignment_lora_grad_norm_total = 0.
    alignment_lora_grad_nonfinite_total = 0.

    blurry_pixcorr = 0.
    test_blurry_pixcorr = 0. # needs >.456 to beat low-level subj01 results in mindeye v1

    # pre-load all batches for this epoch (it's MUCH faster to pre-load in bulk than to separate loading per batch)
    voxel_iters = {} # empty dict because diff subjects have differing # of voxels
    image_iters = torch.zeros(num_iterations_per_epoch, batch_size*len(subj_list), 3, 224, 224).float()
    annot_iters = {}
    perm_iters, betas_iters, select_iters = {}, {}, {}
    for s, train_dl in enumerate(train_dls):
        with torch.cuda.amp.autocast(dtype=data_type):
            iter = -1
            for behav0, past_behav0, future_behav0, old_behav0 in train_dl: 
                # Load images to cpu from hdf5 (requires sorted indexing)
                image_idx = behav0[:,0,0].cpu().long().numpy()
                image0, image_sorted_idx = np.unique(image_idx, return_index=True)                
                if len(image0) != len(image_idx): # hdf5 cant handle duplicate indexing
                    continue
                iter += 1
                image0 = torch.tensor(images[image0], dtype=data_type)
                image_iters[iter,s*batch_size:s*batch_size+batch_size] = image0

                # Load voxels for current batch, matching above indexing
                voxel_idx = behav0[:,0,5].cpu().long().numpy()
                voxel_sorted_idx = voxel_idx[image_sorted_idx]
                voxel0 = voxels[f'subj0{subj_list[s]}'][voxel_sorted_idx]
                voxel0 = torch.Tensor(voxel0).unsqueeze(1)
                voxel0 = apply_reliability(voxel0, f'subj0{subj_list[s]}')

                if epoch < int(mixup_pct * num_epochs):
                    voxel0, perm, betas, select = utils.mixco(voxel0)
                    perm_iters[f"subj0{subj_list[s]}_iter{iter}"] = perm
                    betas_iters[f"subj0{subj_list[s]}_iter{iter}"] = betas
                    select_iters[f"subj0{subj_list[s]}_iter{iter}"] = select

                voxel_iters[f"subj0{subj_list[s]}_iter{iter}"] = voxel0

                if iter >= num_iterations_per_epoch-1:
                    break

    # you now have voxel_iters and image_iters with num_iterations_per_epoch batches each
    for train_i in range(num_iterations_per_epoch):
        with torch.cuda.amp.autocast(dtype=data_type):
            optimizer.zero_grad()
            loss=0.

            voxel_list = [voxel_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
            image = image_iters[train_i].detach()
            image = image.to(device)

            if use_image_aug: 
                image = img_augment(image)

            clip_target = clip_img_embedder(image)
            assert not torch.any(torch.isnan(clip_target))

            if epoch < int(mixup_pct * num_epochs):
                perm_list = [perm_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
                perm = torch.cat(perm_list, dim=0)
                betas_list = [betas_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
                betas = torch.cat(betas_list, dim=0)
                select_list = [select_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
                select = torch.cat(select_list, dim=0)

            voxel_ridge_list = [model.ridge(voxel_list[si],si) for si,s in enumerate(subj_list)]
            voxel_ridge = torch.cat(voxel_ridge_list, dim=0)

            if adapter_prior_weight > 0:
                if adapter_prior_type == "weights":
                    ridge_module = accelerator.unwrap_model(model).ridge
                    loss_adapter_prior = 0.
                    adapter_prior_terms = 0
                    for n, p in ridge_module.named_parameters():
                        if n in adapter_prior_state:
                            loss_adapter_prior = loss_adapter_prior + torch.mean((p - adapter_prior_state[n].to(p.device, dtype=p.dtype)) ** 2)
                            adapter_prior_terms += 1
                    loss_adapter_prior = loss_adapter_prior / max(1, adapter_prior_terms)
                elif adapter_prior_type == "outputs":
                    with torch.no_grad():
                        prior_outputs = [adapter_prior_ridge(voxel_list[si], si) for si, s in enumerate(subj_list)]
                        prior_outputs = torch.cat(prior_outputs, dim=0)
                    loss_adapter_prior = mse(voxel_ridge, prior_outputs)
                else:
                    raise ValueError(f"Unknown adapter_prior_type: {adapter_prior_type}")
                loss_adapter_prior_total += loss_adapter_prior.item()
                loss += loss_adapter_prior * adapter_prior_weight

            if alignment_output_prior_weight > 0:
                with torch.no_grad():
                    alignment_reference_outputs = [alignment_output_prior_ridge(voxel_list[si], si) for si, s in enumerate(subj_list)]
                    alignment_reference_outputs = torch.cat(alignment_reference_outputs, dim=0)
                loss_alignment_output_prior = alignment_output_prior_loss(voxel_ridge, alignment_reference_outputs)
                loss_alignment_output_prior_total += loss_alignment_output_prior.item()
                loss += loss_alignment_output_prior * alignment_output_prior_weight

            voxel_ridge = align_ridge_output(voxel_ridge)
            backbone, clip_voxels, blurry_image_enc_ = model.backbone(voxel_ridge)

            if clip_scale>0:
                clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

            if use_prior:
                loss_prior, prior_out = model.diffusion_prior(text_embed=backbone, image_embed=clip_target)
                loss_prior_total += loss_prior.item()
                loss_prior *= prior_scale
                loss += loss_prior

                recon_cossim += nn.functional.cosine_similarity(prior_out, clip_target).mean().item()
                recon_mse += mse(prior_out, clip_target).item()

            if clip_scale>0:
                if epoch < int(mixup_pct * num_epochs):                
                    loss_clip = utils.mixco_nce(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=.006,
                        perm=perm, betas=betas, select=select)
                else:
                    epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                    loss_clip = utils.soft_clip_loss(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=epoch_temp)

                loss_clip_total += loss_clip.item()
                loss_clip *= clip_scale
                loss += loss_clip

            if blurry_recon:     
                image_enc_pred, transformer_feats = blurry_image_enc_

                image_enc = autoenc.encode(2*image-1).latent_dist.mode() * 0.18215
                loss_blurry = l1(image_enc_pred, image_enc)
                loss_blurry_total += loss_blurry.item()

                if epoch < int(mixup_pct * num_epochs):
                    image_enc_shuf = image_enc[perm]
                    betas_shape = [-1] + [1]*(len(image_enc.shape)-1)
                    image_enc[select] = image_enc[select] * betas[select].reshape(*betas_shape) + \
                        image_enc_shuf[select] * (1 - betas[select]).reshape(*betas_shape)

                image_norm = (image - mean)/std
                image_aug = (blur_augs(image) - mean)/std
                _, cnx_embeds = cnx(image_norm)
                _, cnx_aug_embeds = cnx(image_aug)

                cont_loss = utils.soft_cont_loss(
                    nn.functional.normalize(transformer_feats.reshape(-1, transformer_feats.shape[-1]), dim=-1),
                    nn.functional.normalize(cnx_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                    nn.functional.normalize(cnx_aug_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                    temp=0.2)
                loss_blurry_cont_total += cont_loss.item()

                loss += (loss_blurry + 0.1*cont_loss) * blur_scale #/.18215

            if clip_scale>0:
                # forward and backward top 1 accuracy        
                labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device) 
                fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()

            if blurry_recon:
                with torch.no_grad():
                    # only doing pixcorr eval on a subset of the samples per batch because its costly & slow to compute autoenc.decode()
                    random_samps = np.random.choice(np.arange(len(image)), size=len(image)//5, replace=False)
                    blurry_recon_images = (autoenc.decode(image_enc_pred[random_samps]/0.18215).sample/ 2 + 0.5).clamp(0,1)
                    pixcorr = utils.pixcorr(image[random_samps], blurry_recon_images)
                    blurry_pixcorr += pixcorr.item()

            utils.check_loss(loss)
            accelerator.backward(loss)
            if alignment_lora_rank > 0:
                lora_grad_norm = 0.
                lora_grad_terms = 0
                lora_grad_nonfinite = 0
                for p in accelerator.unwrap_model(model).alignment_lora.parameters():
                    if p.grad is not None:
                        grad_norm = p.grad.detach().float().norm()
                        if torch.isfinite(grad_norm):
                            lora_grad_norm += grad_norm.item()
                            lora_grad_terms += 1
                        else:
                            lora_grad_nonfinite += 1
                alignment_lora_grad_norm_total += lora_grad_norm / max(1, lora_grad_terms)
                alignment_lora_grad_nonfinite_total += lora_grad_nonfinite
            optimizer.step()

            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])

            if lr_scheduler_type is not None:
                lr_scheduler.step()

    model.eval()
    if local_rank==0:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=data_type): 
            for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):  
                # all test samples should be loaded per batch such that test_i should never exceed 0
                assert len(behav) == num_test

                ## Average same-image repeats ##
                if test_image is None:
                    voxel = voxels[f'subj0{subj}'][behav[:,0,5].cpu().long()].unsqueeze(1)
                    voxel = apply_reliability(voxel, f'subj0{subj}')

                    image = behav[:,0,0].cpu().long()

                    unique_image, sort_indices = torch.unique(image, return_inverse=True)
                    for im in unique_image:
                        locs = torch.where(im == image)[0]
                        if len(locs)==1:
                            locs = locs.repeat(3)
                        elif len(locs)==2:
                            locs = locs.repeat(2)[:3]
                        assert len(locs)==3
                        if test_image is None:
                            test_image = torch.Tensor(images[im][None])
                            test_voxel = voxel[locs][None]
                        else:
                            test_image = torch.vstack((test_image, torch.Tensor(images[im][None])))
                            test_voxel = torch.vstack((test_voxel, voxel[locs][None]))

                loss=0.

                test_indices = torch.arange(len(test_voxel))[:300]
                voxel = test_voxel[test_indices].to(device)
                image = test_image[test_indices].to(device)
                assert len(image) == 300

                clip_target = clip_img_embedder(image.float())

                for rep in range(3):
                    voxel_ridge = align_ridge_output(model.ridge(voxel[:,rep],0)) # 0th index of subj_list
                    backbone0, clip_voxels0, blurry_image_enc_ = model.backbone(voxel_ridge)
                    if rep==0:
                        clip_voxels = clip_voxels0
                        backbone = backbone0
                    else:
                        clip_voxels += clip_voxels0
                        backbone += backbone0
                clip_voxels /= 3
                backbone /= 3

                if clip_scale>0:
                    clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                    clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

                # for some evals, only doing a subset of the samples per batch because of computational cost
                random_samps = np.random.choice(np.arange(len(image)), size=len(image)//5, replace=False)

                if use_prior:
                    loss_prior, contaminated_prior_out = model.diffusion_prior(text_embed=backbone[random_samps], image_embed=clip_target[random_samps])
                    test_loss_prior_total += loss_prior.item()
                    loss_prior *= prior_scale
                    loss += loss_prior

                if clip_scale>0:
                    loss_clip = utils.soft_clip_loss(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=.006)

                    test_loss_clip_total += loss_clip.item()
                    loss_clip = loss_clip * clip_scale
                    loss += loss_clip

                if blurry_recon:
                    image_enc_pred, _ = blurry_image_enc_
                    blurry_recon_images = (autoenc.decode(image_enc_pred[random_samps]/0.18215).sample / 2 + 0.5).clamp(0,1)
                    pixcorr = utils.pixcorr(image[random_samps], blurry_recon_images)
                    test_blurry_pixcorr += pixcorr.item()

                if clip_scale>0:
                    # forward and backward top 1 accuracy        
                    labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device) 
                    test_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                    test_bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()

                utils.check_loss(loss)                
                test_losses.append(loss.item())

            assert (test_i+1) == 1
            logs = {"train/loss": np.mean(losses[-(train_i+1):]),
                "test/loss": np.mean(test_losses[-(test_i+1):]),
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
                "train/loss_adapter_prior": loss_adapter_prior_total / (train_i + 1),
                "train/loss_alignment_output_prior": loss_alignment_output_prior_total / (train_i + 1),
                "train/alignment_lora_grad_norm": alignment_lora_grad_norm_total / (train_i + 1),
                "train/alignment_lora_grad_nonfinite": alignment_lora_grad_nonfinite_total / (train_i + 1),
                "params/total": total_params,
                "params/trainable": trainable_params,
                }
            for group in optimizer.param_groups:
                if "name" in group:
                    logs[f"lr/{group['name']}"] = group["lr"]
            val_logs = eval_cached_pairs(val_cache, "val")
            logs.update(val_logs)
            logs.update(probe_diagnostic_logs(val_cache))
            if val_logs:
                val_source = val_cache.get("source", "unknown") if val_cache is not None else "unknown"
                print(
                    "val_metrics "
                    f"epoch={epoch + 1} source={val_source} "
                    f"val/loss={val_logs.get('val/loss', float('nan')):.6g} "
                    f"val/fwd_pct_correct={val_logs.get('val/fwd_pct_correct', float('nan')):.6g} "
                    f"val/bwd_pct_correct={val_logs.get('val/bwd_pct_correct', float('nan')):.6g}"
                )

            # if finished training, save jpg recons if they exist
            if (epoch == num_epochs-1) or (epoch % ckpt_interval == 0):
                if blurry_recon:    
                    image_enc = autoenc.encode(2*image[:4]-1).latent_dist.mode() * 0.18215
                    # transform blurry recon latents to images and plot it
                    fig, axes = plt.subplots(1, 8, figsize=(10, 4))
                    jj=-1
                    for j in [0,1,2,3]:
                        jj+=1
                        axes[jj].imshow(utils.torch_to_Image((autoenc.decode(image_enc[[j]]/0.18215).sample / 2 + 0.5).clamp(0,1)))
                        axes[jj].axis('off')
                        jj+=1
                        axes[jj].imshow(utils.torch_to_Image((autoenc.decode(image_enc_pred[[j]]/0.18215).sample / 2 + 0.5).clamp(0,1)))
                        axes[jj].axis('off')

                    if wandb_log:
                        logs[f"test/blur_recons"] = wandb.Image(fig, caption=f"epoch{epoch:03d}")
                        plt.close()
                    else:
                        plt.show()

            progress_bar.set_postfix(**logs)

            if wandb_log: wandb.log(logs)

            if ckpt_saving and "val/loss" in logs:
                if logs["val/loss"] < best_val_loss:
                    best_val_loss = logs["val/loss"]
                    epochs_since_best_val = 0
                    save_ckpt("best_val")
                else:
                    epochs_since_best_val += 1

    # Save model checkpoint and reconstruct
    if (ckpt_saving) and (epoch % ckpt_interval == 0):
        save_ckpt(f'last')

    # wait for other GPUs to catch up if needed
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    if early_stop_patience > 0 and val_cache is not None and epochs_since_best_val >= early_stop_patience:
        print(f"Early stopping after {epoch + 1} epochs; best_val_loss={best_val_loss}")
        break

print("\n===Finished!===\n")
if ckpt_saving:
    save_ckpt(f'last')


# In[ ]:


plt.plot(losses)
plt.show()
plt.plot(test_losses)
plt.show()
