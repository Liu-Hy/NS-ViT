"""Script to run on the HAL server"""

import os
from pathlib import Path
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
# from torchvision import transforms
from custom_dataset import ImageFolder
import argparse
from utils import *
from tqdm import tqdm
from train_single import validate, validate_corruption, prepare_loader


def main(args):
    val_batch_size = 256
    val_ratio = 1.
    save_path = Path("output/hal")
    data_path = Path("/var/lib/data/imagenet")
    info_path = Path("info")

    if args.debug:
        val_batch_size = 128
        val_ratio = 0.01

    model_name = 'vit_base_patch16_224-dat'
    model, patch_size, img_size, model_config = get_model_and_config(model_name, pretrained=(args.ckpt_path=="none"), model_dir="pretrained")
    if args.ckpt_path != "none":
        checkpoint = torch.load(save_path.joinpath(args.ckpt_path))
        model.load_state_dict(checkpoint["model_state_dict"])
    model.cuda()

    if "-" in model_name:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize(model_config['mean'], model_config['std'])

    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])

    criterion = nn.CrossEntropyLoss()

    result = dict()
    # Evaluate on val and OOD datasets except imagenet-c
    for split in SPLITS:
        if split != "train" and not split.startswith("c-"):
            typ_path = data_path.joinpath(split)
            val_loader = prepare_loader(typ_path, info_path, val_batch_size, val_transform)
            acc, _ = validate(val_loader, model, criterion, val_ratio)
            result[split] = acc
            if split == "val":
                result["fgsm"] = validate(val_loader, model, criterion, val_ratio, adv=True)[0]
    # Evaluate on imagenet-c
    corruption_rs = validate_corruption(data_path.joinpath("corruption"), info_path, model, val_transform, criterion, val_batch_size, val_ratio)
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
