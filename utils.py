import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from collections import OrderedDict

import timm
import torch
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
#import wandb
from dotenv import load_dotenv
from timm.data import resolve_data_config
from timm.models import create_model
from torch.utils.data import SubsetRandomSampler
from torch.utils import model_zoo

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from constants import *
_logger = logging.getLogger(__name__)
load_dotenv()


def parse_opts():
    parser = argparse.ArgumentParser(description='Nullspace robustness study of deep learning architectures')
    parser.add_argument('--arch', default=None, choices=['vit_base_patch32_224', 'vit_small_patch32_224',
                                                         'vit_large_patch32_224', 'swin_tiny_patch4_window7_224',
                                                         'resnet50', 'efficientnet_b0', 'convnext_tiny',
                                                         'mobilenetv3_small'
                                                         ], help='Neural network architecture')
    parser.add_argument('--output', default=None, help='Directory to save the output of a run!')
    parser.add_argument('--data', default=None, help='Path to the data files!')
    parser.add_argument('--type', default=None, choices=['input', 'encoder'], help='Experiment type')
    parser.add_argument('--img-size', default=224, type=int, help='Input image size for the network')
    parser.add_argument('--epochs', default=500, help='Number of epochs for the optimisation')
    parser.add_argument('--eps', default=0.01, help='learning rate for optimisation')
    parser.add_argument('--milestones', nargs='+', default=[150, 300, 400])
    parser.add_argument('--batch-size', default=256)
    parser.add_argument('--lims', nargs='+', default=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0],
                        help='Picks the pre-saved starting noise from the artifact')
    return parser.parse_args()


def empty_gpu():
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def init_dataset(args, model_config):
    """Create a Dataloader object for the training and validation set respectively. """
    base_dataset = datasets.ImageFolder(os.path.join(args.data, 'train'), transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(), transforms.Normalize(model_config['mean'], model_config['std'])]))
    loader = torch.utils.data.DataLoader(base_dataset, batch_size=args.batch_size, shuffle=True, num_workers=32,
                                         pin_memory=True)
    val_dataset = datasets.ImageFolder(os.path.join(args.data, 'val'), transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(), transforms.Normalize(model_config['mean'], model_config['std'])]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=32, pin_memory=True)
    return loader, val_loader


def get_mean(l):
    return sum(l) / len(l)


def load_state_dict(checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = ''
        if isinstance(checkpoint, dict):
            if use_ema and checkpoint.get('state_dict_ema', None) is not None:
                state_dict_key = 'state_dict_ema'
            elif use_ema and checkpoint.get('model_ema', None) is not None:
                state_dict_key = 'model_ema'
            elif 'state_dict' in checkpoint:
                state_dict_key = 'state_dict'
            elif 'model' in checkpoint:
                state_dict_key = 'model'
        if state_dict_key:
            state_dict = checkpoint[state_dict_key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        # Remove the 0th (normalization) layer from the checkpoint in the DAT model
        if '0.mean' in state_dict.keys() and '0.std' in state_dict.keys():
            n_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("1."):
                    n_state_dict[k[2:]] = v
            state_dict = n_state_dict
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_checkpoint(model, checkpoint_path, use_ema=False, strict=True):
    if os.path.splitext(checkpoint_path)[-1].lower() in ('.npz', '.npy'):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, 'load_pretrained'):
            model.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError('Model cannot load numpy checkpoint')
        return
    state_dict = load_state_dict(checkpoint_path, use_ema)
    model.load_state_dict(state_dict, strict=strict)


def get_model_and_config(model_name, ckpt_path=None, use_ema=False):
    print(f"Creating model: {model_name}")
    model = create_model(
        model_name,
        pretrained=False,
    )

    if ckpt_path is not None:
        load_checkpoint(model, ckpt_path, use_ema=use_ema, strict=True)

    config = resolve_data_config({}, model=model)
    # Of all ViT models concerned in this work, DAT is the only version that uses mean and std different from default_cfg of the timm model
    if ckpt_path is not None and "dat" in ckpt_path.lower():
        config["mean"] = [0.485, 0.456, 0.406]
        config["std"] = [0.229, 0.224, 0.225]
    print(config)
    try:
        patch_size = model.patch_embed.patch_size[0]
        img_size = model.patch_embed.img_size[0]
    except:
        patch_size = 32
        img_size = 224
    print(f'{model_name}, {img_size}x{img_size}, patch_size:{patch_size}')
    return model, patch_size, img_size, config


def validate_noise(data_path, model, transform, batch_size, delta_x, val_ratio, device):
    model.eval()
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        mdl = model.module
    elif isinstance(model, torch.nn.Module):
        mdl = model
    # for type_path in sorted(data_path.iterdir()):
    clean_path = data_path.joinpath("imagenet")
    if delta_x is not None:
        print("---- Validate noise effect (1st row learned noise, 2nd row permuted)")
        corr_res = validate_encoder_noise(mdl, clean_path, transform, batch_size, delta_x, val_ratio, device)
        idx = torch.randperm(delta_x.nelement())
        t = delta_x.reshape(-1)[idx].reshape(delta_x.size())
        incorr_res = validate_encoder_noise(mdl, clean_path, transform, batch_size, t, val_ratio, device)


def encoder_forward(model, x):
    # Concat CLS token to the patch embeddings,
    # Forward pass them through the model encoder to get the features of the CLS token.
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        mdl = model.module
    elif isinstance(model, torch.nn.Module):
        mdl = model
    cls_token = mdl.cls_token
    pos_embed = mdl.pos_embed
    pos_drop = mdl.pos_drop
    blocks = mdl.blocks
    norm = mdl.norm

    x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = pos_drop(x + pos_embed)
    x = blocks(x)
    x = norm(x)
    return mdl.pre_logits(x[:, 0])


def encoder_level_epsilon_noise(model, loader, img_size, rounds, nlr, lim, eps, img_ratio, disable=False):
    print(f"img size {img_size}")
    model.eval()
    model.zero_grad()
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        mdl = model.module
    elif isinstance(model, torch.nn.Module):
        mdl = model

    patch_embed = mdl.patch_embed

    with torch.no_grad():
        _ = patch_embed(torch.rand(1, 3, img_size, img_size).cuda(non_blocking=True))
        del_x_shape = _.shape

    delta_x = torch.empty(del_x_shape).uniform_(-lim, lim).type(torch.FloatTensor).cuda(non_blocking=True)
    if isinstance(model, DistributedDataParallel):
        dist.broadcast(delta_x, 0)
        # dist.barrier()
    delta_x.requires_grad = True
    print(f"Noise norm: {round(torch.norm(delta_x).item(), 4)}", flush=True)

    optimizer = AdamW([delta_x], lr=nlr)
    scheduler = CosineAnnealingLR(optimizer, len(loader) * rounds)

    for i in range(rounds):
        iterator = tqdm(loader, position=0, disable=disable)
        for st, (imgs, lab) in enumerate(iterator):
            assert delta_x.requires_grad == True
            if st > int(img_ratio * len(loader)) - 1:
                break
            imgs = imgs.cuda(non_blocking=True)

            with torch.no_grad():
                og_preds = mdl.head(mdl.forward_features(imgs))

            optimizer.zero_grad()

            x = patch_embed(imgs)
            x = x + delta_x

            preds = mdl.head(encoder_forward(model, x))

            p_og = torch.softmax(og_preds, dim=-1)
            p_alt = torch.softmax(preds, dim=-1)
            mse_probs = (((p_og - p_alt) ** 2).sum(dim=-1)).mean()
            if isinstance(model, DistributedDataParallel):
                dist.all_reduce(mse_probs)
                mse_probs /= dist.get_world_size()
            if mse_probs < eps:
                print(f"Image finished training at epoch {i} step {st}", flush=True)
                return delta_x

            error_mult = (((preds - og_preds) ** 2).sum(dim=-1) ** 0.5).mean()
            error_mult.backward()
            if isinstance(model, DistributedDataParallel):
                dist.barrier()
                dist.all_reduce(delta_x.grad)
                delta_x.grad /= dist.get_world_size()
            optimizer.step()
            scheduler.step()
            iterator.set_postfix({"error": round(error_mult.item(), 4)})
        if not (i + 1) % 1:
            print(f'Noise trained for {i + 1} epochs, error: {round(error_mult.item(), 4)}', flush=True)

    return delta_x


def validate_encoder_noise(model, data_path, transform, batch_size, delta_x, val_ratio, device):
    """Evaluate the influence of the encoder noise "delta+x" to the model's prediction on a dataset"""
    og_preds = {'feats': [], 'outs': []}
    alt_preds = {'feats': [], 'outs': []}
    val_dataset = datasets.ImageFolder(data_path, transform)

    val_size = len(val_dataset)
    indices = torch.randperm(val_size)[:int(val_ratio * val_size)]
    val_sampler = SubsetRandomSampler(indices)

    loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=16,
                                         pin_memory=True)
    model.eval()
    with torch.no_grad():
        for _, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device)
            og_feats = model.module.forward_features(imgs)
            og_outs = model.module.head(og_feats)
            og_preds['feats'].append(og_feats.cpu())
            og_preds['outs'].append(og_outs.cpu())

            x = model.module.patch_embed(imgs)
            x += delta_x
            alt_feats = encoder_forward(model, x)
            alt_outs = model.module.head(alt_feats)
            alt_preds['feats'].append(alt_feats.cpu())
            alt_preds['outs'].append(alt_outs.cpu())

    og_preds['feats'] = torch.cat(og_preds['feats'], dim=0)
    og_preds['outs'] = torch.cat(og_preds['outs'], dim=0)
    alt_preds['feats'] = torch.cat(alt_preds['feats'], dim=0)
    alt_preds['outs'] = torch.cat(alt_preds['outs'], dim=0)

    mse_feats = (((og_preds['feats'] - alt_preds['feats']) ** 2).sum(dim=-1)).mean()
    mse_logits = (((og_preds['outs'] - alt_preds['outs']) ** 2).sum(dim=-1)).mean()
    p_og = torch.softmax(og_preds['outs'], dim=-1)
    p_alt = torch.softmax(alt_preds['outs'], dim=-1)
    mse_probs = (((p_og - p_alt) ** 2).sum(dim=-1)).mean()

    mx_probs, mx_cls = torch.max(p_og, dim=-1)
    alt_probs = []
    for i, j in enumerate(mx_cls):
        alt_probs.append(p_alt[i, j])
    # alt_max_probs = ((((p_og-p_alt)**2)*mult).sum(dim=-1)).mean()
    # print('ALT MSE MAX PROBS', alt_max_probs.item())
    alt_probs = torch.tensor(alt_probs)
    assert alt_probs[0] == p_alt[0][mx_cls[0]] and alt_probs[-1] == p_alt[-1][mx_cls[-1]]

    abs_conf = torch.abs(mx_probs - alt_probs).mean()
    mse_conf = ((mx_probs - alt_probs) ** 2).mean()

    uneq = ((mx_cls == torch.max(p_alt, dim=-1)[1]).sum()) / p_og.shape[0]  # rate of agreement

    print(
        f'MSE FEATS: {mse_feats.item():.4f}\t MSE LOGITS: {mse_logits.item():.4f}\t MSE PROBS: {mse_probs.item():.4f}\t ABS MAX PROB: {abs_conf.item():.4f}\t MSE MAX PROB: {mse_conf.item():.4f}\t EQ CLS: {uneq:.4f}')
    return dict(mse_feats=mse_feats.item(), mse_logits=mse_logits.item(), mse_probs=mse_probs.item(),
                abs_conf=abs_conf.item(), mse_conf=mse_conf.item(), eq=uneq)


def validate_complete(model, loader, delta_x, device):
    """Evaluate the influence of the input noise "delta+x" to the model's prediction on a dataset"""
    with torch.no_grad():
        ogs, alts = [], []
        for _, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device)
            ogs.append(model(imgs).cpu())

            imgs = imgs + delta_x
            alts.append(model(imgs).cpu())

    ogs = torch.cat(ogs, dim=0)
    alts = torch.cat(alts, dim=0)

    mse_logits = (((ogs - alts) ** 2).sum(dim=-1)).mean()
    p_ogs = torch.softmax(ogs, dim=-1)
    p_alts = torch.softmax(alts, dim=-1)
    mse_probs = (((p_ogs - p_alts) ** 2).sum(dim=-1)).mean()

    mx_probs, mx_cls = torch.max(p_ogs, dim=-1)
    eq_cls_pred = ((mx_cls == torch.max(p_alts, dim=-1)[1]).sum()) / p_ogs.shape[0]

    print(f'MSE LOGITS: {mse_logits.item():.4f}\t MSE PROBS: {mse_probs.item():.4f}\t EQ CLS: {eq_cls_pred:.4f}')
    return dict(mse_logits=mse_logits.item(), mse_probs=mse_probs.item(), eq=eq_cls_pred.item())