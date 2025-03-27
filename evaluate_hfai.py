"""Script to run on the HFAI server"""
import haienv
haienv.set_env('ns')

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
from ImageNetDG import ImageNetDG
import argparse
from utils import *
from tqdm import tqdm
from train_hfai import validate, validate_corruption, prepare_loader

import hfai
import hfai.distributed as dist
from ffrecord.torch import DataLoader
from ffrecord.torch.dataset import Subset
dist.set_nccl_opt_level(dist.HFAI_NCCL_OPT_LEVEL.AUTO)

def main(local_rank, args):
    val_batch_size = 32
    val_ratio = 1.
    save_path = Path("output/hfai")

    if args.debug:
        val_batch_size = 2
        val_ratio = 0.01

    ip = os.environ['MASTER_ADDR']
    port = os.environ['MASTER_PORT']
    hosts = int(os.environ['WORLD_SIZE'])  # 机器个数
    rank = int(os.environ['RANK'])  # 当前机器编号
    gpus = torch.cuda.device_count()  # 每台机器的GPU个数

    # world_size是全局GPU个数，rank是当前GPU全局编号
    dist.init_process_group(backend='nccl',
                            init_method=f'tcp://{ip}:{port}',
                            world_size=hosts * gpus,
                            rank=rank * gpus + local_rank)
    torch.cuda.set_device(local_rank)

    model_name = 'vit_base_patch16_224'
    #ckpt_path = "base_ps16_epochs10_lr0.0001_bs16_adv_True_nlr0.1_rounds3_lim3_eps0.01_imgr0.1_trainr1.0_valr1.0/model/7"
    model, patch_size, img_size, model_config = get_model_and_config(model_name, variant='dat', offline=True)
    if args.ckpt_path != "none":
        checkpoint = torch.load(save_path.joinpath(args.ckpt_path))
        model.load_state_dict(checkpoint["model_state_dict"])
    model.cuda()
    # model = hfai.nn.to_hfai(model)
    model = DistributedDataParallel(model.cuda(), device_ids=[local_rank])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(model_config['mean'], model_config['std'])])

    criterion = nn.CrossEntropyLoss()

    result = dict()
    # Evaluate on val and OOD datasets except imagenet-c
    for split in SPLITS:
        if split != "train" and not split.startswith("c-"):
            val_loader = prepare_loader(split, val_batch_size, val_transform)
            acc, _ = validate(val_loader, model, criterion, val_ratio)
            result[split] = acc
    # Evaluate on imagenet-c
    corruption_rs = validate_corruption(model, val_transform, criterion, val_batch_size, val_ratio)
    result["corruption"] = corruption_rs["mce"]
    print(result)
    if rank == 0 and local_rank == 0:
        total = get_mean([100 - v if k == "corruption" else v for k, v in result.items()])
        print(f"Avg performance: {total}\n", result)


if __name__ == '__main__':
    ngpus = torch.cuda.device_count()
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    parser = argparse.ArgumentParser()
    parser.add_argument('-db', '--debug', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default='none', help='checkpoint path')
    args = parser.parse_args()
    hfai.multiprocessing.spawn(main, args=(args,), nprocs=ngpus)