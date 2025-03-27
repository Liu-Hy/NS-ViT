import os
import torch
from utils import validate_complete
from methods import image_level_nullnoise
import timm
from timm.data import resolve_data_config 
import wandb
from utils import init_wandb, init_dataset
from utils import parse_opts


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f'using device: {device}')
    wandb_logger = init_wandb(args)
    path = wandb_logger.use_artifact(f'{args.type}:latest').download()
    starting_delta_x = torch.load(os.path.join(path, f'init.pth'))

    start_lim = 0
    try:
        checkpoint = torch.load(wandb.restore('checkpoint.pth').name, map_location="cpu")
        for start_lim, lim in enumerate(args.lim):
            if lim not in checkpoint['results'].key():
                break
        args.lims = args.lims[start_lim:]
    except:
        checkpoint = {'results': {}, 'delta_x': {}}

    for lim in args.lims:
        model = timm.create_model(args.arch, pretrained=True)
        model_config = resolve_data_config({}, model=model)
        print(f'Model config {model_config} for {args.arch} running exp. type {args.type}')
        model = model.to(device)
        model.eval()

        loader, val_loader = init_dataset(args, model_config)

        delta_x = image_level_nullnoise(model, loader, args, delta_x=starting_delta_x[lim].to(device), device=device)
        checkpoint['delta_x'][lim] = delta_x.detach().cpu()

        # validation
        corr_res = validate_complete(model, val_loader, delta_x, device)
        idx = torch.randperm(delta_x.nelement())
        t = delta_x.reshape(-1)[idx].reshape(delta_x.size())
        incorr_res = validate_complete(model, val_loader, t, device)
        checkpoint['results'][lim] = {'corr': corr_res, 'shuffle': incorr_res}

        torch.save(checkpoint, os.path.join(args.output, 'checkpoint.pth'))
        wandb.save(args.output)


if __name__ == '__main__':
  main(parse_opts())
