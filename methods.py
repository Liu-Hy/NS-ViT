"""Functions for finding image-level and """

import torch
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from utils import encoder_forward
import time
from tqdm import tqdm


def encoder_level_noise(model, loader, img_size, rounds, nlr, lim, device):
    model = model.to(device)
    model.eval()
    model.zero_grad()

    #for param in model.parameters():
        #param.requires_grad = False

    model = model.to(device)
    patch_embed = model.patch_embed

    with torch.no_grad():
        _ = patch_embed(torch.rand(1, 3, img_size, img_size).to(device))
        del_x_shape = _.shape
        #print(del_x_shape)

    assert isinstance(lim, (float, int))
    delta_x = torch.empty(del_x_shape).uniform_(-lim, lim).type(torch.FloatTensor).to(device)
    delta_x.requires_grad = True

    optimizer = AdamW([delta_x], lr=nlr)
    scheduler = CosineAnnealingLR(optimizer, len(loader) * rounds)
    #optimizer = SGD([delta_x], lr=eps, momentum=0.9, weight_decay=1e-4)
    #scheduler = StepLR(optimizer, step_size=200, gamma=0.9)
    #print('Starting magnitude', delta_x.shape, (((delta_x.squeeze(0)) ** 2).sum(dim=0) ** 0.5).mean())

    iterator = tqdm(range(rounds), position=0, leave=True)
    for i in iterator:
        for _, (imgs, _) in enumerate(loader):
            assert delta_x.requires_grad == True
            imgs = imgs.to(device)

            with torch.no_grad():
                og_preds = model.head(model.forward_features(imgs))

            #model.zero_grad()
            optimizer.zero_grad()

            x = patch_embed(imgs)
            x = x + delta_x

            preds = model.head(encoder_forward(model, x))
            error_mult = (((preds - og_preds) ** 2).sum(dim=-1) ** 0.5).mean()
            # error_mult = ((preds - og_preds) ** 2).sum(dim=-1).mean()
            error_mult.backward()
            optimizer.step()
            scheduler.step()
            """grad = delta_x.grad.data
            with torch.no_grad():
                delta_x -= eps * grad.detach()"""
                # delta_x = torch.max(torch.min(delta_x, delta_x_max), delta_x_min)
                # delta_x = torch.clamp(delta_x, min=delta_x_min, max=delta_x_max)

            # delta_x.grad.zero_()

        iterator.set_postfix({"error": round(error_mult.item(), 4)})
        #if not (i + 1) % 2:
            #print(f'Noise trained for {i+1} epochs, error: {round(error_mult.item(), 4)}')

    return delta_x

def encoder_level_epsilon_noise(model, loader, img_size, rounds, nlr, lim, eps, device):
    print(f"img size {img_size}")
    model = model.to(device)
    model.eval()
    model.zero_grad()

    #for param in model.parameters():
        #param.requires_grad = False

    model = model.to(device)
    patch_embed = model.patch_embed

    with torch.no_grad():
        _ = patch_embed(torch.rand(1, 3, img_size, img_size).to(device))
        del_x_shape = _.shape
        #print(del_x_shape)

    assert isinstance(lim, (float, int))
    delta_x = torch.empty(del_x_shape).uniform_(-lim, lim).type(torch.FloatTensor).to(device)
    delta_x.requires_grad = True
    print(f"Noise norm: {round(torch.norm(delta_x).item(), 4)}")

    optimizer = AdamW([delta_x], lr=nlr)
    scheduler = CosineAnnealingLR(optimizer, len(loader) * rounds)
    #optimizer = SGD([delta_x], lr=eps, momentum=0.9, weight_decay=1e-4)
    #scheduler = StepLR(optimizer, step_size=200, gamma=0.9)
    #print('Starting magnitude', delta_x.shape, (((delta_x.squeeze(0)) ** 2).sum(dim=0) ** 0.5).mean())

    #iterator = tqdm(range(rounds))
    for i in range(rounds):
        iterator = tqdm(loader, position=0, leave=True)
        for st, (imgs, lab) in enumerate(iterator):
            assert delta_x.requires_grad == True
            imgs = imgs.to(device)

            with torch.no_grad():
                og_preds = model.head(model.forward_features(imgs))

            #model.zero_grad()
            optimizer.zero_grad()

            x = patch_embed(imgs)
            x = x + delta_x

            preds = model.head(encoder_forward(model, x))

            p_og = torch.softmax(og_preds, dim=-1)
            p_alt = torch.softmax(preds, dim=-1)
            mse_probs = (((p_og - p_alt) ** 2).sum(dim=-1)).mean()
            if mse_probs < eps:
                print(f"Image finished training at epoch {i} step {st}")
                return delta_x

            error_mult = (((preds - og_preds) ** 2).sum(dim=-1) ** 0.5).mean()

            # error_mult = ((preds - og_preds) ** 2).sum(dim=-1).mean()
            # hinge = torch.max(torch.stack([error_mult - eps, torch.tensor(0)]))
            error_mult.backward()
            optimizer.step()
            scheduler.step()
            """grad = delta_x.grad.data
            with torch.no_grad():
                delta_x -= eps * grad.detach()"""
                # delta_x = torch.max(torch.min(delta_x, delta_x_max), delta_x_min)
                # delta_x = torch.clamp(delta_x, min=delta_x_min, max=delta_x_max)

            # delta_x.grad.zero_()
            iterator.set_postfix({"error": round(error_mult.item(), 4)})
        if not (i + 1) % 1:
            print(f'Noise trained for {i+1} epochs, error: {round(error_mult.item(), 4)}')

    return delta_x

def image_level_nullnoise(model, loader, args, logger, lim, delta_x, device):
    """问题： 这里并没有用lim来initialize，而是直接用传递的delta_x了"""
    model = model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    # del_x_shape = (1, 3, img_size, img_size)
    print('Starting magnitude', delta_x.shape, (((delta_x.squeeze(0)) ** 2).sum(dim=0) ** 0.5).mean())

    eps = args.eps
    step = 0
    for i in range(args.epochs):
        for _, (imgs, _) in enumerate(loader):
            delta_x.requires_grad = True
            imgs = imgs.to(device)

            with torch.no_grad():
                og_preds = model(imgs)

            model.zero_grad()

            imgs = imgs + delta_x

            preds = model(imgs)
            error_mult = (((preds - og_preds) ** 2).sum(dim=-1) ** 0.5).mean()
            error_mult.backward()
            grad = delta_x.grad.data
            with torch.no_grad():
                delta_x -= eps * grad.detach()

        if i % 100 == 0:
            print(i, error_mult.item())
            logger.log({f'loss/lim_{lim}': error_mult.item()}, step=step)
        if i in args.milestones:
            eps /= 10.0
        step += 1

    return delta_x
