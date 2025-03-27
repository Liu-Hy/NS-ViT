"""Single-gpu version to debug and run on local machine"""

import os
from pathlib import Path
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
#from torchvision import transforms
from custom_dataset import ImageFolder
from utils import *
from tqdm import tqdm
from torch.utils.data import DataLoader


def adv_train(dataloader, model, criterion, optimizer, scheduler, adv, delta_x, train_ratio):
    model.train()
    iterator = tqdm(dataloader, position=0, leave=True)
    epoch_loss = 0.
    for step, batch in enumerate(iterator):
        if step > int(train_ratio * len(dataloader)):
            break
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
        """if step % 20 == 0:
            if adv:
                print(
                    f'Step {step}, Loss: {round(loss.item(), 4)}, Consistency_ratio: {round((consistency / (loss + adv_loss)).item(), 4)}',
                    flush=True)
            else:
                print(
                    f'Step {step}, Loss: {round(loss.item(), 4)}', flush=True)"""
        epoch_loss += loss.item()
        iterator.set_postfix({"loss": round((epoch_loss / (step + 1)), 3)})


def validate(dataloader, model, criterion, val_ratio):
    loss, correct1, correct5, total = torch.zeros(4).cuda()
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if step > int(val_ratio * len(dataloader)):
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

    loss_val = loss.item() / len(dataloader)
    acc1 = 100 * correct1.item() / total.item()
    acc5 = 100 * correct5.item() / total.item()
    #if is_clean:
        #print(f'Validation loss: {loss_val:.4f}, Acc1: {acc1:.2f}%, Acc5: {acc5:.2f}%', flush=True)

    return acc1, loss_val



def validate_corruption(data_path, info_path, model, transform, criterion, batch_size, val_ratio):
    result = dict()
    type_errors = []
    for typ in tqdm(CORRUPTIONS, position=0, leave=True):
        type_path = data_path.joinpath(typ)
        assert type_path in list(data_path.iterdir())
        errors = []
        for s in range(1, 6):
            s_path = type_path.joinpath(str(s))
            assert s_path in list(type_path.iterdir())
            loader = prepare_loader(s_path, info_path, batch_size, transform)
            acc, _ = validate(loader, model, criterion, 0.01)
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


def prepare_loader(split_data, info_path, batch_size, transform=None):
    if isinstance(split_data, (str, Path)):
        split_data = ImageFolder(split_data, info_path, transform=transform)
    data_loader = DataLoader(split_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    return data_loader

def main():
    empty_gpu()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    # 超参数设置
    epochs = 5
    train_batch_size = 64  # 256 for base model
    val_batch_size = 128
    lr = 1e-4  # When using SGD and StepLR, set to 0.001 # when AdamW and bachsize=256, 3e-4
    rounds, nlr, lim = 3, 0.02, 1  # lim=3, nlr=0.1 # round=3
    eps = 0.001  # 0.001
    adv = True
    img_ratio = 0.02 # 0.02
    train_ratio = 0.1
    val_ratio = 0.05
    task = "imagenet"  # "imagenette"
    save_path = Path("./output").joinpath(task)
    data_path = Path("/var/lib/data")
    save_path.mkdir(exist_ok=True, parents=True)

    # 模型、数据、优化器
    model_name = 'vit_base_patch16_224'
    model, patch_size, img_size, model_config = get_model_and_config(model_name, variant="DAT")
    model.cuda()

    m = model_name.split('_')[1]
    setting = f'{m}_ps{patch_size}_epochs{epochs}_lr{lr}_bs{train_batch_size}_adv_{adv}_nlr{nlr}_rounds{rounds}' + \
              f'_lim{lim}_eps{eps}_imgr{img_ratio}_trainr{train_ratio}_valr{val_ratio}'
    setting_path = save_path.joinpath(setting)
    noise_path = setting_path.joinpath("noise")
    model_path = setting_path.joinpath("model")
    setting_path.mkdir(exist_ok=True, parents=True)
    noise_path.mkdir(exist_ok=True, parents=True)
    model_path.mkdir(exist_ok=True, parents=True)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(model_config['mean'], model_config['std'])])

    info_path = Path("info")

    held_out = 0.1
    data_set = ImageFolder(data_path.joinpath('imagenet/train'), info_path, train_transform)
    len_dev = int(held_out * len(data_set))
    len_train = len(data_set) - len_dev
    train_set, dev_set = torch.utils.data.random_split(data_set, (len_train, len_dev))
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    img_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=4,
                              pin_memory=True)
    dev_loader = prepare_loader(dev_set, info_path, val_batch_size)

    """base_dataset = datasets.ImageFolder(data_path.joinpath('imagenet/train'), train_transform)
    img_indices = torch.randperm(train_size)[:int(img_ratio * train_size)]
    img_sampler = SubsetRandomSampler(img_indices)from utils import *
    train_indices = torch.randperm(train_size)[:int(train_ratio * train_size)]
    train_sampler = SubsetRandomSampler(train_indices)
    img_loader = torch.utils.data.DataLoader(base_dataset, batch_size=train_batch_size, sampler=img_sampler,
                                             num_workers=16, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(base_dataset, batch_size=train_batch_size, sampler=train_sampler,
                                               num_workers=16, pin_memory=True)"""

    """
    val_dataset = hfai.datasets.ImageNet('val', transform=val_transform)
    val_datasampler = DistributedSampler(val_dataset)
    val_dataloader = val_dataset.loader(batch_size, sampler=val_datasampler, num_workers=4, pin_memory=True)"""
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(model_config['mean'], model_config['std'])])
    # val_dataset = datasets.ImageFolder(data_path.joinpath('val'), val_transform)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=16,
    # pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, len(train_loader) * epochs)

    best_acc = 0.

    # 训练、验证
    start_epoch = len(list(model_path.iterdir()))
    if start_epoch > 0:
        print(f"Restore training from epoch {start_epoch}")
        checkpoint = torch.load(model_path.joinpath(str(start_epoch - 1)))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_acc = torch.load(setting_path.joinpath("best_epoch"))["best_acc"]
        print(f"Previous best acc: {best_acc}")

    delta_x = None
    for epoch in range(start_epoch, epochs):
        print("\n" + "-" * 32 + f"\nEnter epoch {epoch}")
        if adv:
            # delta_x = encoder_level_noise(model, img_loader, rounds, nlr, lim=lim, device=device)
            if Path.exists(noise_path.joinpath(str(epoch))):
                print(f"Loading learned noise at epoch {epoch}")
                delta_x = torch.load(noise_path.joinpath(str(epoch)))['delta_x'].to(device)
            else:
                print("---- Learning noise")
                delta_x = encoder_level_epsilon_noise(model, img_loader, img_size, rounds, nlr, lim, eps, img_ratio)
                torch.save({"delta_x": delta_x}, noise_path.joinpath(str(epoch)))
            print(f"Noise norm: {round(torch.norm(delta_x).item(), 4)}")

        print("---- Training model")
        adv_train(train_loader, model, criterion, optimizer, scheduler, adv, delta_x, train_ratio)
        print("---- Validating model")
        result = dict()
        # Evaluate on held-out set
        dev_acc, _ = validate(dev_loader, model, criterion, val_ratio)
        # Evaluate on val and OOD datasets except imagenet-c
        for split in SPLITS:
            if split != "train" and not split.startswith("c-"):
                if split == "val":
                    typ_path = data_path.joinpath("imagenet", split)
                else:
                    typ_path = data_path.joinpath(split)
                val_loader = prepare_loader(typ_path, info_path, val_batch_size, val_transform)
                acc, _ = validate(val_loader, model, criterion, val_ratio)
                result[split] = acc
        # Evaluate on imagenet-c
        corruption_rs = validate_corruption(data_path.joinpath("corruption"), info_path, model, val_transform, criterion, val_batch_size, val_ratio)
        result["corruption"] = corruption_rs["mce"]
        # 保存
        print(f"Dev acc: {dev_acc}")
        total = get_mean([100 - v if k == "corruption" else v for k, v in result.items()])
        print(f"Avg performance: {total}\n", result)

        torch.save({"model_name": model_name, "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "result": result}, model_path.joinpath(str(epoch)))
        # 保存
        if dev_acc > best_acc:
            best_acc = dev_acc
            print(f'New Best Acc: {best_acc:.2f}%')
            torch.save({"model_state_dict": model.state_dict(), "best_epoch": epoch, "best_acc": best_acc}, setting_path.joinpath("best_epoch"))


if __name__ == '__main__':
    main()