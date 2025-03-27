"""Evaluation script to run on single node of HFAI server
Currently having some problem with AutoAttack"""
import haienv
haienv.set_env('ns')

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
# from torchvision import transforms
from ImageNetDG import ImageNetDG
import argparse
from utils import *
from tqdm import tqdm
from train_hfai_node import validate, validate_corruption, prepare_loader
import models

import hfai
import hfai.distributed as dist
from ffrecord.torch import DataLoader
from ffrecord.torch.dataset import Subset
dist.set_nccl_opt_level(dist.HFAI_NCCL_OPT_LEVEL.AUTO)


def get_val_transform(config, split):
    normalize = transforms.Normalize(config['mean'], config['std'])

    # ImageNet-C and ImageNet-Stylized are already 224 x 224 images
    if split in ["corruption", "stylized"]:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    else:
        val_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])
    return val_transform


def main(args):
    val_batch_size = 32
    val_ratio = 1.
    save_path = Path("output/hfai")

    if args.debug:
        val_batch_size = 2
        val_ratio = 0.01

    model_name = 'vit_base_patch16_224'
    ckpt_path = '../pretrained/vit_base_patch16_224-dat.pth.tar'

    print(f"=== Evaluating model {model_name} ===")
    model, patch_size, img_size, model_config = get_model_and_config(model_name, ckpt_path, use_ema=True)
    model.cuda()
    # model = hfai.nn.to_hfai(model)
    model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()

    result = dict()
    # Evaluate on val and OOD datasets except imagenet-c
    for split in SPLITS:
        if split != "train" and not split.startswith("c-"):
            # Use a logits mask to evaluate on 200-class validation sets
            if split == "adversarial":
                mask = imagenet_a_mask
            elif split == "rendition":
                mask = imagenet_r_mask
            else:
                mask = None
            val_transform = get_val_transform(model_config, split)
            if split == "val":
                dts = hfai.datasets.ImageNet('val', transform=val_transform)
                val_loader = prepare_loader(dts, val_batch_size)
            else:
                val_loader = prepare_loader(split, val_batch_size, val_transform)
            acc, _ = validate(val_loader, model, criterion, val_ratio, mask=mask)
            result[split] = acc
            # Evaluate adversarial robustness on the validation set.
            if split == "val":
                result["fgsm"] = validate(val_loader, model, criterion, val_ratio, adv="FGSM")[0]
    # Evaluate on imagenet-c
    c_transform = get_val_transform(model_config, "corruption")
    corruption_rs = validate_corruption(model, c_transform, criterion, val_batch_size, 1.)
    result["corruption"] = corruption_rs["mce"]
    print(result)
    total = get_mean([100 - v if k == "corruption" else v for k, v in result.items()])
    print(f"Avg performance: {total}\n", result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-db', '--debug', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default='none', help='checkpoint path')
    args = parser.parse_args()
    main(args)