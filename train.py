"""Distributed version to run on the server"""
import hfai_env
hfai_env.set_env('nullspace')

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

from utils import *
from tqdm import tqdm

import hfai
import hfai.distributed as dist
dist.set_nccl_opt_level(dist.HFAI_NCCL_OPT_LEVEL.AUTO)
hfai.nn.functional.set_replace_torch()

def adv_train(dataloader, model, criterion, optimizer, scheduler, adv, delta_x, epoch, local_rank, start_step, best_acc):
    model.train()
    for step, batch in enumerate(dataloader):
        step += start_step

        imgs, labels = [x.cuda(non_blocking=True) for x in batch]
        optimizer.zero_grad()
        outputs = model(imgs)
        # optimizer.zero_grad()
        loss = criterion(outputs, labels)
        if adv:
            x = model.patch_embed(imgs)
            x = x + delta_x
            adv_outputs = model.head(encoder_forward(model, x))
            adv_loss = criterion(adv_outputs, labels)
            # adv_loss.backward()
            consistency = ((adv_outputs - outputs) ** 2).sum(dim=-1).mean()
            loss = loss + adv_loss  # + consistency
        loss.backward()
        optimizer.step()
        scheduler.step()
        if local_rank == 0 and step % 20 == 0:
            if adv:
                print(
                    f'Epoch: {epoch}, Step {step}, Loss: {round(loss.item(), 4)}, Consistency_ratio: {round((consistency / (loss + adv_loss)).item(), 4)}',
                    flush=True)
            else:
                print(
                    f'Epoch: {epoch}, Step {step}, Loss: {round(loss.item(), 4)}', flush=True)
        if step % 100 == 0:
            model.try_save(epoch, step + 1, others=(best_acc, delta_x), force=True)


def validate(dataloader, model, criterion, val_ratio, epoch, local_rank):
    loss, correct1, correct5, total = torch.zeros(4).cuda()
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if step > int(val_ratio * len(dataloader)) - 1:
                print(f"break evaluation at step {step}")
                break
            samples, labels = [x.cuda(non_blocking=True) for x in batch]
            # print(labels.shape, labels.max(), labels.min())
            outputs = model(samples)
            # print(f'output shape: {outputs.shape}')
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
    """if local_rank == 0:
        if is_clean:
            print(f'Validation loss: {loss_val:.4f}, Acc1: {acc1:.2f}%, Acc5: {acc5:.2f}%', flush=True)"""

    return acc1, loss_val


def validate_corruption(model, transform, criterion, batch_size, val_ratio, epoch, local_rank):
    result = dict()
    type_errors = []
    for typ in CORRUPTIONS:
        errors = []
        for s in range(1, 6):
            split = "c-" + typ + "-" + str(s)
            loader = prepare_loader(split, batch_size, transform)
            acc, _ = validate(loader, model, criterion, val_ratio, epoch, local_rank)
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
    if isinstance(split_data, "str"):
        split_data = ImageNetDG(split_data, transform=transform)
    else:
        assert isinstance(split_data, ImageNetDG)
    data_sampler = DistributedSampler(split_data)
    data_loader = split_data.loader(batch_size, sampler=data_sampler, num_workers=4, pin_memory=True)
    return data_loader

def main(local_rank):
    #empty_gpu()
    # 超参数设置
    epochs = 5
    train_batch_size = 128  # 256 for base model
    val_batch_size = 128
    lr = 3e-4  # When using SGD and StepLR, set to 0.001
    rounds, nlr, lim = 10, 0.1, 3  # lim=1.0, nlr=0.02
    eps = 0.01  # 0.001
    adv = True
    img_ratio = 0.1
    train_ratio = 1
    val_ratio = 0.1
    save_path = "output"
    data_path = Path("/var/lib/data")
    #save_path.mkdir(exist_ok=True, parents=True)

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

    # 模型、数据、优化器
    model_name = 'vit_base_patch32_224'
    model, patch_size, img_size, model_config = get_model_and_config(model_name, pretrained=True)
    model.cuda()
    model = hfai.nn.to_hfai(model)
    model = DistributedDataParallel(model.cuda(), device_ids=[local_rank])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(model_config['mean'], model_config['std'])])

    held_out = 0.1
    data_set = ImageNetDG('train', transform=train_transform)
    train_set, dev_set = torch.utils.data.random_split(data_set, (1 - held_out, held_out))
    print(f"splitting length: train set: {len(train_set)}, dev set: {len(dev_set)}")
    train_sampler = DistributedSampler(train_set)
    train_loader = train_set.loader(train_batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)

    dev_loader = prepare_loader(dev_set, val_batch_size)
    img_loader = train_loader

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(model_config['mean'], model_config['std'])])

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, len(train_loader) * epochs)

    ckpt_path = os.path.join(save_path, 'latest.pt')
    start_epoch, start_step, others = hfai.checkpoint.init(model, optimizer, scheduler=scheduler, ckpt_path=ckpt_path)
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
                delta_x = encoder_level_epsilon_noise(model, img_loader, img_size, rounds, nlr, lim, eps)
            print(f"Noise norm: {round(torch.norm(delta_x).item(), 4)}")

        print("---- Training model")
        adv_train(train_loader, model, criterion, optimizer, scheduler, adv, delta_x, epoch, local_rank, start_step, best_acc)
        start_step = 0
        delta_x = None
        print("---- Validating model")
        result = dict()
        # Evaluate on held-out set
        dev_acc, _ = validate(dev_loader, model, criterion, 1, epoch, local_rank)
        # TODO: incorporate dev_acc
        # Evaluate on val and OOD datasets except imagenet-c
        for split in SPLITS:
            if split != "train" and not split.startswith("c-"):
                dev_loader = prepare_loader(split, val_batch_size, val_transform)
                acc, _ = validate(dev_loader, model, criterion, 1, epoch, local_rank)
                result[split] = acc
        # Evaluate on imagenet-c
        corruption_rs = validate_corruption(model, val_transform, criterion, val_batch_size, val_ratio, epoch, local_rank)
        result["corruption"] = corruption_rs["mce"]
        # 保存
        if rank == 0 and local_rank == 0:
            print(f"Dev acc: {dev_acc}")
            total = get_mean([100 - v if k == "corruption" else v for k, v in result.items()])
            print(f"Avg performance: {total}\n", result)
            if dev_acc > best_acc:
                best_acc = dev_acc
                print(f'New Best Acc: {best_acc:.2f}%')
                torch.save(model.module.state_dict(),
                           os.path.join(save_path, 'best.pt'))


if __name__ == '__main__':
    ngpus = torch.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus)
