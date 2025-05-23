"""For each sampling limit, generate the nullspace noise, cache it to be used next time, and test it on a
validation set"""
import os

import fire
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler

from methods import encoder_level_noise
from utils import get_model_and_config, validate_encoder_noise
import time

def main(model_name='vit_base_patch32_224', data_dir='data/', output_dir='./outputs/'):
    # print(torch.cuda.get_device_name(0))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    m = model_name.split('_')[1]
    results = {}
    for lim in [2.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0]:
        model, patch_size, img_size, model_config = get_model_and_config(model_name, pretrained=True)
        model = model.to(device)
        model.eval()
        rounds, nlr = 30, 0.02  # rounds=500
        del_name = f'enc_{m}_p{patch_size}_im{img_size}_nlr_{nlr}_rounds_{rounds}_lim_{lim}.v1.pth'

        base_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(), transforms.Normalize(model_config['mean'], model_config['std'])]))
        train_size = len(base_dataset)
        indices = torch.randperm(train_size)[:int(0.1 * train_size)]
        train_sampler = SubsetRandomSampler(indices)
        loader = torch.utils.data.DataLoader(base_dataset, batch_size=256, sampler=train_sampler,
                                             num_workers=16, pin_memory=True)
        if os.path.exists(os.path.join(output_dir, del_name)):
            delta_y = torch.load(os.path.join(output_dir, del_name))['delta_y'].to(device)
        else:
            # 0.1 loader
            delta_y = encoder_level_noise(model, loader, rounds, nlr, lim=lim, device=device)
            torch.save({'delta_y': delta_y.cpu()}, os.path.join(output_dir, del_name))

        # validation
        val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(), transforms.Normalize(model_config['mean'], model_config['std'])]))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=16,
                                                 pin_memory=True)
        corr_res = validate_encoder_noise(model, val_loader, delta_y, device)
        idx = torch.randperm(delta_y.nelement())
        t = delta_y.reshape(-1)[idx].reshape(delta_y.size())
        incorr_res = validate_encoder_noise(model, val_loader, t, device)
        results[del_name] = {'corr': corr_res, 'shuffle': incorr_res}
    torch.save(results, os.path.join(output_dir, model_name + '.ns'))


if __name__ == '__main__':
    fire.Fire(main)