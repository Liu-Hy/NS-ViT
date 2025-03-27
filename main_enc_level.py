import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import get_model_and_config, validate_by_parts
from methods import encoder_level_noise
import fire


def main(model_name='vit_base_patch32_224', data_dir='data/', output_dir='./outputs/'):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    m = model_name.split('_')[1]
    results = {}
    for lim in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0]:
        model, patch_size, img_size, model_config = get_model_and_config(model_name, pretrained=True)
        model = model.to(device)
        model.eval()
        rounds, eps, milestones, err_fac, mag_fac = 500, 0.01, [150, 300, 400], 1.0, 0.0
        del_name = f'enc_{m}_p{patch_size}_im{img_size}_eps_{eps}_rounds_{rounds}_lim_{lim}.v1.pth'

        base_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(), transforms.Normalize(model_config['mean'], model_config['std'])]))
        loader = torch.utils.data.DataLoader(base_dataset, batch_size=256, shuffle=True, num_workers=32, pin_memory=True)
        if os.path.exists(os.path.join(output_dir, del_name)):
            delta_y = torch.load(os.path.join(output_dir, del_name))['delta_y'].to(device)
        else:
            delta_y = encoder_level_noise(model, loader, rounds, eps, milestones, lim=lim, device=device)
            torch.save({'delta_y': delta_y.cpu()}, os.path.join(output_dir, del_name))

        # validation
        val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(), transforms.Normalize(model_config['mean'], model_config['std'])]))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=32, pin_memory=True)
        corr_res = validate_by_parts(model_name, model, val_loader, delta_y, device)
        idx = torch.randperm(delta_y.nelement())
        t = delta_y.reshape(-1)[idx].reshape(delta_y.size())
        incorr_res = validate_by_parts(model_name, model, val_loader, t, device)
        results[del_name] = {'corr': corr_res, 'shuffle': incorr_res}
    torch.save(results, os.path.join(output_dir,model_name+'.ns'))

if __name__ == '__main__':
  fire.Fire(main)
