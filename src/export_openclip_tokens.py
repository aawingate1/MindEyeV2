#!/usr/bin/env python

import argparse
import os

import h5py
import kornia
import numpy as np
import open_clip
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Export MindEye-compatible OpenCLIP image tokens.")
    parser.add_argument("--image_h5_path", type=str, required=True)
    parser.add_argument("--cocoidxs_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_dir", type=str, default=None)
    return parser.parse_args()


def resolve_hf_cache_dir(cache_dir: str | None) -> str | None:
    if cache_dir:
        if os.path.basename(cache_dir) == "hf_hub":
            return cache_dir
        hf_hub_subdir = os.path.join(cache_dir, "hf_hub")
        if os.path.isdir(hf_hub_subdir):
            return hf_hub_subdir
        return cache_dir

    env_cache = os.environ.get("HF_HUB_CACHE")
    if env_cache:
        return env_cache

    env_home = os.environ.get("HF_HOME")
    if env_home:
        return os.path.join(env_home, "hf_hub")
    return None


class MindEyeOpenCLIPTokenEmbedder(torch.nn.Module):
    def __init__(self, device="cuda", cache_dir=None):
        super().__init__()
        hf_cache_dir = resolve_hf_cache_dir(cache_dir)
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-bigG-14",
            device=torch.device("cpu"),
            pretrained="laion2b_s39b_b160k",
            cache_dir=hf_cache_dir,
        )
        del model.transformer
        model.visual.output_tokens = True
        self.model = model.eval().to(device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.device = device
        self.register_buffer(
            "mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32),
            persistent=False,
        )

    def preprocess(self, image):
        image = kornia.geometry.resize(
            image,
            (224, 224),
            interpolation="bicubic",
            align_corners=True,
            antialias=True,
        )
        image = (image + 1.0) / 2.0
        image = kornia.enhance.normalize(image, self.mean, self.std)
        return image

    @torch.no_grad()
    def forward(self, image):
        image = self.preprocess(image)
        _, tokens = self.model.visual(image)
        return tokens


def main():
    args = parse_args()
    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"

    hf_cache_dir = resolve_hf_cache_dir(args.cache_dir)
    if hf_cache_dir is not None:
        os.environ.setdefault("HF_HUB_CACHE", hf_cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", hf_cache_dir)
        os.environ.setdefault("HF_HOME", os.path.dirname(hf_cache_dir))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    cocoidxs = np.load(args.cocoidxs_path)
    embedder = MindEyeOpenCLIPTokenEmbedder(device=requested_device, cache_dir=hf_cache_dir)

    token_array = None
    with h5py.File(args.image_h5_path, "r") as image_h5:
        images = image_h5["images"]
        for start in range(0, len(cocoidxs), args.batch_size):
            end = min(start + args.batch_size, len(cocoidxs))
            image_batch = torch.tensor(images[cocoidxs[start:end]], dtype=torch.float32, device=requested_device)
            token_batch = embedder(image_batch).detach().cpu().numpy().astype(np.float16)
            if token_array is None:
                token_array = np.empty((len(cocoidxs),) + token_batch.shape[1:], dtype=np.float16)
            token_array[start:end] = token_batch
            print(f"embedded images {start}:{end} / {len(cocoidxs)}")

    np.save(args.output_path, token_array)
    print(f"saved token array to {args.output_path}")


if __name__ == "__main__":
    main()
