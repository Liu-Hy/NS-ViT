import os
from pathlib import Path
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torchvision import transforms, models, datasets
from utils import get_model_and_config, validate_encoder_noise, encoder_forward, empty_gpu
from methods import encoder_level_noise, encoder_level_epsilon_noise
from tqdm import tqdm

corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
               'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate',
               'jpeg_compression']

alex = [88.6, 89.4, 92.3, 82.0, 82.6, 78.6, 79.8, 86.7, 82.7, 81.9, 56.5, 85.3, 64.6, 71.8, 60.7]


def train(dataloader, model, criterion, optimizer, scheduler, epoch):
    model.train()
    for step, batch in enumerate(dataloader):
        imgs, labels = [x.cuda(non_blocking=True) for x in batch]
        outputs = model(imgs)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 20 == 0:
            print(f'Training Step: {step}, Loss: {loss.item()}', flush=True)


def adv_train(dataloader, model, criterion, optimizer, scheduler, adv, delta_x):
    model.train()
    iterator = tqdm(dataloader)
    for step, batch in enumerate(iterator):
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
            tot_loss = loss + adv_loss  # + consistency
            tot_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 20 == 0:
            if adv:
                print(
                    f'Step {step}, Loss: {round(loss.item(), 4)}, consistency_ratio: {round((consistency / (loss + adv_loss)).item(), 4)}',
                    flush=True)
            else:
                print(
                    f'Step {step}, Loss: {round(loss.item(), 4)}', flush=True)


def validate(data_path, model, criterion, transform, batch_size, val_ratio, is_clean=False):
    loss, correct1, correct5, total = torch.zeros(4).cuda()
    model.eval()
    val_dataset = datasets.ImageFolder(data_path, transform)

    val_size = len(val_dataset)
    indices = torch.randperm(val_size)[:int(val_ratio * val_size)]
    val_sampler = SubsetRandomSampler(indices)

    dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=16,
                                             pin_memory=True)
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
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
    if is_clean:
        print(f'Validation loss: {loss_val:.4f}, Acc1: {acc1:.2f}%, Acc5: {acc5:.2f}%', flush=True)

    return 100 - acc1, loss_val


def mean(l):
    return sum(l) / len(l)


def validate_all(data_path, model, criterion, transform, batch_size, delta_x, val_ratio, device):
    model.eval()
    es = []
    # result = dict()
    result = {"mce": 1.}
    # for type_path in sorted(data_path.iterdir()):
    clean_path = data_path.joinpath("imagenet/val")
    if delta_x is not None:
        print("---- Validate noise effect (1st row learned noise, 2nd row permuted)")
        corr_res = validate_encoder_noise(model, clean_path, transform, batch_size, delta_x, val_ratio, device)
        idx = torch.randperm(delta_x.nelement())
        t = delta_x.reshape(-1)[idx].reshape(delta_x.size())
        incorr_res = validate_encoder_noise(model, clean_path, transform, batch_size, t, val_ratio, device)
    clean_e, _ = validate(clean_path, model, criterion, transform, batch_size, val_ratio, is_clean=True)
    for typ in tqdm(corruptions):
        type_path = data_path.joinpath(f"imagenet-c/{typ}")
        assert type_path in list(data_path.joinpath("imagenet-c").iterdir())
        type_e = []
        for s in range(1, 6):
            s_path = type_path.joinpath(str(s))
            assert s_path in list(type_path.iterdir())
            s_e, _ = validate(s_path, model, criterion, transform, batch_size, val_ratio)
            type_e.append(s_e)
        es.append(mean(type_e))
    me = mean(es)
    relative_es = [(e / al) for (e, al) in zip(es, alex)]
    mce = 100 * mean(relative_es)
    result["clean_e"] = clean_e
    result["es"] = es
    result["ces"] = relative_es
    result["me"] = me
    result["mce"] = mce
    print(f"mCE: {mce:.2f}%, mean_err: {me}%", flush=True)
    return result


def main():
    empty_gpu()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    # 超参数设置
    epochs = 10
    train_batch_size = 128  # 256 for base model
    val_batch_size = 128
    lr = 3e-4  # When using SGD and StepLR, set to 0.001 # when AdamW and bachsize=256, 3e-4
    rounds, nlr, lim = 30, 0.01, 4  # lim=1.0, nlr=0.02
    eps = 0.01  # 0.001
    adv = False
    img_ratio = 0.05
    train_ratio = 0.1
    val_ratio = 0.01
    task = "imagenet"  # "imagenette"
    save_path = Path("./output").joinpath(task)
    data_path = Path("/var/lib/data")
    save_path.mkdir(exist_ok=True, parents=True)

    # 模型、数据、优化器
    # model = models.resnet50().cuda()
    model_name = 'vit_base_patch16_224'
    model, patch_size, img_size, model_config = get_model_and_config(model_name, pretrained=True)
    model.cuda()

    m = model_name.split('_')[1]
    setting = f'{m}_ps{patch_size}_epochs{epochs}_lr{lr}_bs{train_batch_size}_adv_{adv}_nlr{nlr}_rounds{rounds}' + \
              f'_lim{lim}_eps{eps}_imgr{img_ratio}_trainr{train_ratio}_valr{val_ratio}'
    setting_path = save_path.joinpath(setting)
    setting_path.mkdir(exist_ok=True, parents=True)
    noise_path = setting_path.joinpath("noise")
    noise_path.mkdir(exist_ok=True, parents=True)
    model_path = setting_path.joinpath("model")
    model_path.mkdir(exist_ok=True, parents=True)
    """train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])  # 定义训练集变换 
    train_dataset = hfai.datasets.ImageNet('train', transform=train_transform)
    train_datasampler = DistributedSampler(train_dataset)
    train_dataloader = train_dataset.loader(batch_size, sampler=train_datasampler, num_workers=4, pin_memory=True)"""

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(model_config['mean'], model_config['std'])])

    base_dataset = datasets.ImageFolder(data_path.joinpath('imagenet/train'), train_transform)
    print(data_path.joinpath('imagenet/train'))
    train_size = len(base_dataset)
    img_indices = torch.randperm(train_size)[:int(img_ratio * train_size)]
    img_sampler = SubsetRandomSampler(img_indices)
    train_indices = torch.randperm(train_size)[:int(train_ratio * train_size)]
    train_sampler = SubsetRandomSampler(train_indices)
    img_loader = torch.utils.data.DataLoader(base_dataset, batch_size=train_batch_size, sampler=img_sampler,
                                             num_workers=16, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(base_dataset, batch_size=train_batch_size, sampler=train_sampler,
                                               num_workers=16, pin_memory=True)

    """val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])  # 定义测试集变换
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
    # optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.9)

    best_rbn = 0.

    # 训练、验证
    start_epoch = len(list(model_path.iterdir()))
    if start_epoch > 0:
        print(f"Restore training from epoch {start_epoch}")
        checkpoint = torch.load(model_path.joinpath(str(start_epoch - 1)))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_rbn = torch.load(setting_path.joinpath("best_epoch"))["best_rbn"]
        print(f"Previous best rbn: {best_rbn}")

    delta_x = None
    for epoch in range(start_epoch, epochs):
        print("\n" + "-" * 32 + f"\nEnter epoch {epoch}")
        if adv:
            # delta_x = encoder_level_noise(model, img_loader, rounds, nlr, lim=lim, device=device)
            if Path.exists(noise_path.joinpath(str(epoch))):
                print(f"Loading learned noise at epoch {epoch}")
                delta_x = torch.load(noise_path.joinpath(str(epoch)))['delta_x'].to(device)
            else:
                delta_x = encoder_level_epsilon_noise(model, img_loader, img_size, rounds, nlr, lim, eps, device)
                torch.save({"delta_x": delta_x}, noise_path.joinpath(str(epoch)))
            print(f"Noise norm: {round(torch.norm(delta_x).item(), 4)}")

        """print("---- Validate noise effect (1st row learned noise, 2nd row permuted)")
        corr_res = validate_by_parts(model, val_loader, delta_x, device)
        idx = torch.randperm(delta_x.nelement())
        t = delta_x.reshape(-1)[idx].reshape(delta_x.size())
        incorr_res = validate_by_parts(model, val_loader, t, device)"""

        print("---- Training model")
        # adv_train(train_loader, model, criterion, optimizer, scheduler, adv, delta_x)
        # acc = validate(val_loader, model, criterion, epoch)
        print("---- Validating model")
        rs = validate_all(data_path, model, criterion, val_transform, val_batch_size, delta_x, val_ratio, device)
        torch.save({"model_name": model_name, "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "result": rs}, model_path.joinpath(str(epoch)))
        rbn = rs["mce"]
        # 保存
        if rbn > best_rbn:
            best_rbn = rbn
            print(f'New Best Robustness: {rbn:.2f}%')
            torch.save({"best_epoch": epoch, "best_rbn": rbn},
                       setting_path.joinpath("best_epoch"))


if __name__ == '__main__':
    main()
