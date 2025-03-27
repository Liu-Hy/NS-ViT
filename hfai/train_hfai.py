"""Multi-node training script on the HFAI server"""
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
import torchattacks
# from torchvision import transforms
from ImageNetDG import ImageNetDG
import argparse
from utils import *
from tqdm import tqdm

import hfai
import hfai.distributed as dist
from ffrecord.torch import DataLoader
from ffrecord.torch.dataset import Subset
from easyrobust.attacks import AutoAttack

dist.set_nccl_opt_level(dist.HFAI_NCCL_OPT_LEVEL.AUTO)


# hfai.nn.functional.set_replace_torch()

def adv_train(dataloader, model, criterion, optimizer, scheduler, adv, delta_x, train_ratio, epoch, local_rank,
              start_step, best_acc, disable):
    model.train()
    iterator = tqdm(dataloader, position=0, disable=disable)
    epoch_loss = 0.
    for step, batch in enumerate(iterator):
        step += start_step
        if step > int(train_ratio * len(dataloader)):
            break
        imgs, labels = [x.cuda(non_blocking=True) for x in batch]
        optimizer.zero_grad()
        outputs = model(imgs)
        # optimizer.zero_grad()
        loss = criterion(outputs, labels)
        if adv:
            x = model.module.patch_embed(imgs)
            x = x + delta_x.cuda(non_blocking=True)
            adv_outputs = model.module.head(encoder_forward(model, x))
            adv_loss = criterion(adv_outputs, labels)
            # adv_loss.backward()
            # consistency = ((adv_outputs - outputs) ** 2).sum(dim=-1).mean()
            loss = loss + adv_loss  # + consistency
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()
        iterator.set_postfix({"loss": round((epoch_loss / (step + 1)), 3)})
        if step % 100 == 0:
            model.try_save(epoch, step + 1, others=(best_acc, delta_x), force=True)


def validate(dataloader, model, criterion, val_ratio, mask=None, adv='none', eps=8/225):
    loss, correct1, correct5, total = torch.zeros(4).cuda()
    model.eval()
    assert adv in ['none', 'FGSM', 'Linf', 'L2'], '{} is not supported!'.format(adv)
    if adv == "FGSM":
        attack = torchattacks.FGSM(model, eps=8 / 225)
    elif adv != "none":
        attack = AutoAttack(model, norm=adv, eps=eps, version='standard')
    for step, batch in enumerate(dataloader):
        if step > int(val_ratio * len(dataloader)):
            break
        samples, labels = [x.cuda(non_blocking=True) for x in batch]
        if adv == "FGSM":
            samples = attack(samples, labels)
        elif adv != "none":
            samples = attack.run_standard_evaluation(samples, labels, bs=samples.shape[0])
        with torch.no_grad():
            outputs = model(samples)
            if mask is not None:
                outputs[:, mask] = -float('inf')
            loss += criterion(outputs, labels)
            _, preds = outputs.topk(5, -1, True, True)
            correct1 += torch.eq(preds[:, :1], labels.unsqueeze(1)).sum()
            correct5 += torch.eq(preds, labels.unsqueeze(1)).sum()
            total += samples.size(0)

    for x in [loss, correct1, correct5, total]:
        dist.reduce(x, 0)

    loss_val = loss.item() / dist.get_world_size() / len(dataloader)
    acc1 = 100 * correct1.item() / total.item()
    acc5 = 100 * correct5.item() / total.item()

    return acc1, loss_val


def validate_corruption(model, transform, criterion, batch_size, val_ratio):
    result = dict()
    type_errors = []
    for typ in CORRUPTIONS:
        errors = []
        for s in range(1, 6):
            split = "c-" + typ + "-" + str(s)
            loader = prepare_loader(split, batch_size, transform)
            acc, _ = validate(loader, model, criterion, val_ratio)
            errors.append(100 - acc)
        type_errors.append(get_mean(errors))
    me = get_mean(type_errors)
    relative_es = [(e / al) for (e, al) in zip(type_errors, ALEX)]
    mce = 100 * get_mean(relative_es)
    result["es"] = type_errors
    result["ces"] = relative_es
    result["me"] = me
    result["mce"] = mce
    print(f"mCE: {mce:.2f}%, mean_err: {me}%", flush=True)
    return result


def prepare_loader(split_data, batch_size, transform=None):
    if isinstance(split_data, str):
        split_data = ImageNetDG(split_data, transform=transform)
    data_sampler = DistributedSampler(split_data)
    data_loader = DataLoader(split_data, batch_size=batch_size, sampler=data_sampler, num_workers=4, pin_memory=True)
    return data_loader


def main(local_rank, args):
    # 超参数设置
    epochs = 10
    train_batch_size = 16  # 16  # 256 for base model
    val_batch_size = 16  # 16
    rounds = 3
    lr = args.lr  # When using SGD and StepLR, set to 0.001
    lim = args.lim
    nlr = args.nlr
    eps = args.eps
    adv = True
    img_ratio = 0.1
    train_ratio = 1.
    val_ratio = 1.
    save_path = Path("../output/hfai")
    data_path = Path("/var/lib/data")
    # save_path.mkdir(exist_ok=True, parents=True)

    if args.debug:
        train_batch_size, val_batch_size = 2, 2
        img_ratio, train_ratio, val_ratio = 0.001, 0.001, 0.1

    # 多机通信
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

    if rank == 0 and local_rank == 0:
        print(args)
    disable = (local_rank != 0)

    # 模型、数据、优化器
    model_name = 'vit_base_patch16_224'
    ckpt_path = "../pretrained/vit_base_patch16_224-dat.pth.tar"
    model, patch_size, img_size, model_config = get_model_and_config(model_name, ckpt_path=ckpt_path)
    model.cuda()
    # model = hfai.nn.to_hfai(model)
    model = DistributedDataParallel(model.cuda(), device_ids=[local_rank])

    m = model_name.split('_')[1]
    setting = f'{m}_ps{patch_size}_epochs{epochs}_lr{lr}_bs{train_batch_size}_adv_{adv}_nlr{nlr}_rounds{rounds}' + \
              f'_lim{lim}_eps{eps}_imgr{img_ratio}_trainr{train_ratio}_valr{val_ratio}'
    setting_path = save_path.joinpath(setting)
    noise_path = setting_path.joinpath("noise")
    model_path = setting_path.joinpath("model")
    if local_rank == 0:
        setting_path.mkdir(exist_ok=True, parents=True)
        noise_path.mkdir(exist_ok=True, parents=True)
        model_path.mkdir(exist_ok=True, parents=True)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(model_config['mean'], model_config['std'])])

    dataset = hfai.datasets.ImageNet('train', transform=train_transform)
    data_size = len(dataset)
    print(f"data set size: {data_size}")
    cutoff = int(0.9 * data_size)
    rand_idx = torch.randperm(data_size)
    train_indices, dev_indices = rand_idx[:cutoff], rand_idx[cutoff:]
    train_set = Subset(dataset, train_indices)
    dev_set = Subset(dataset, dev_indices)
    train_sampler = DistributedSampler(train_set)
    img_sampler = DistributedSampler(train_set)
    dev_sampler = DistributedSampler(dev_set)

    train_loader = DataLoader(train_set, train_batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    img_loader = DataLoader(train_set, train_batch_size, sampler=img_sampler, num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_set, val_batch_size, sampler=dev_sampler, num_workers=4, pin_memory=True)

    print(f"splitting length: train set: {len(train_set)}, dev set: {len(dev_set)}")

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(model_config['mean'], model_config['std'])])
    val_set = hfai.datasets.ImageNet('val', transform=val_transform)
    val_sampler = DistributedSampler(val_set)
    val_loader = DataLoader(val_set, val_batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, len(train_loader) * epochs)

    ckpt_path = save_path.joinpath(setting + '_' + 'latest.pt')
    try:
        start_epoch, start_step, others = hfai.checkpoint.init(model, optimizer, scheduler=scheduler,
                                                               ckpt_path=ckpt_path)
    except RuntimeError:
        start_epoch, start_step, others = 0, 0, None
        print("Failed to load checkpoint, start from scratch instead.")
    best_acc, delta_x = 0., None
    if others is not None:
        best_acc, delta_x = others

    # 训练、验证
    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        train_loader.set_step(start_step)
        if adv:
            if delta_x is None:
                print("---- Learning noise")
                delta_x = encoder_level_epsilon_noise(model, img_loader, img_size, rounds, nlr, lim, eps, img_ratio,
                                                      disable)
                torch.save({"delta_x": delta_x}, noise_path.joinpath(str(epoch)))
            print(f"Noise norm: {round(torch.norm(delta_x).item(), 4)}")

        if local_rank == 0:
            print("---- Training model")
        adv_train(train_loader, model, criterion, optimizer, scheduler, adv, delta_x, train_ratio, epoch, local_rank,
                  start_step, best_acc, disable)
        start_step = 0
        delta_x = None
        if local_rank == 0:
            print("---- Validating model")
        result = dict()
        # Evaluate on held-out set
        dev_acc, _ = validate(dev_loader, model, criterion, val_ratio)
        # Evaluate on val set
        val_acc, _ = validate(val_loader, model, criterion, val_ratio)
        result["val"] = val_acc
        if local_rank == 0:
            try:
                torch.save({"model_name": model_name, "epoch": epoch,
                            "model_state_dict": model.module.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "result": result}, model_path.joinpath(str(epoch)))
            except FileExistsError:
                print("File exists")
            # 保存
            if rank == 0:
                print(f"Dev acc: {dev_acc}")
                total = get_mean([100 - v if k == "corruption" else v for k, v in result.items()])
                print(f"Avg performance: {total}\n", result)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    print(f'New Best Acc: {best_acc:.2f}%')
                    try:
                        torch.save(
                            {"model_state_dict": model.module.state_dict(), "best_epoch": epoch, "best_acc": best_acc},
                            setting_path.joinpath("best_epoch"))
                    except FileExistsError:
                        print("File exists")


if __name__ == '__main__':
    ngpus = torch.cuda.device_count()
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    parser = argparse.ArgumentParser()
    parser.add_argument('-db', '--debug', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for updating network')
    parser.add_argument('--lim', type=float, default=3, help='sampling limit of the noise')
    parser.add_argument('--nlr', type=float, default=0.1, help='learning rate for the noise')
    parser.add_argument('--eps', type=float, default=0.01, help='threshold to stop training the noise')

    args = parser.parse_args()
    hfai.multiprocessing.spawn(main, args=(args,), nprocs=ngpus)