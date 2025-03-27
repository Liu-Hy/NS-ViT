"""For each sampling limit, load the ns noise trained on vit_small network, and test it on the validation set using
different networks, to see how the ns noise transfer across architectures. """

import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import fire
import timm
from timm.data import resolve_data_config


def main():
    data_dir = 'data'

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)

    results = {}
    unnorm = lambda x, c, i: c['mean'][i] + x * c['std'][i]

    models = ['vit_small_patch32_224', 'vit_base_patch32_224', 'resnet50', 'efficientnet_b0', 'mobilenetv3_small_050', 'swin_tiny_patch4_window7_224', 'convnext_tiny']
    nets = [timm.create_model(m, pretrained=True) for m in models]
    sc = resolve_data_config({}, model=nets[0])

    for lim in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0]:
        nullnoise = torch.load(f'outputs/del_x_vit_small_patch32_224_eps_0.01_rounds_500_lim_{lim}.v1.pth')['delta_y'][0].detach().cpu()
        results[lim] = {}
        nullnoise = torch.cat([unnorm(nullnoise[0], sc, 0).unsqueeze(0), unnorm(nullnoise[1], sc, 1).unsqueeze(0),
                               unnorm(nullnoise[2], sc, 2).unsqueeze(0)], dim=0)
        print('noise in img domain', torch.min(nullnoise), torch.max(nullnoise))

    # validation
        for idx, model in enumerate(nets):
            model_config = resolve_data_config({}, model=model)
            model = model.to(device)
            model.eval()

            norm = transforms.Normalize(model_config['mean'], model_config['std'])
            tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
            apply_norm = lambda x: norm(x).unsqueeze(0)
            mnoise = apply_norm(nullnoise)  # 和cross-main里的不一样了。那里是先加噪音再normalize, 这里是分别normalize再相加

            val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), tf)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=32,
                                                     pin_memory=True)
            c, inc, t = 0, 0, 0

            for _, (imgs, _) in enumerate(val_loader):
                with torch.no_grad():
                    ns_imgs = imgs + mnoise
                    ns_imgs = ns_imgs.to(device)
                    imgs = imgs.to(device)
                    _, clss = torch.max(model(imgs), dim=-1)
                    _, n_clss = torch.max(model(ns_imgs), dim=-1)

                    c += (clss == n_clss).sum()
                    t += imgs.shape[0]
            results[lim][models[idx]] = c / t
            print(models[idx], lim, results[lim][models[idx]])
        torch.save(results, 'outputs/cross_robustness.omega.nullspace')


if __name__ == '__main__':
    fire.Fire(main)

