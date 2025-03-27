import torch
from utils import encoder_forward


def encoder_level_noise(model, loader, rounds, eps, milestones, lim, device):
    # Todo: Some issue with making it work on a GPU
    model = model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    model = model.to(device)
    patch_embed = model.patch_embed

    with torch.no_grad():
        _ = patch_embed(torch.rand(1, 3, 224, 224).to(device))
        del_x_shape = _.shape
        print(del_x_shape)

    print('lim type', lim)
    if not isinstance(lim, str):
        delta_x = torch.empty(del_x_shape).uniform_(-lim, lim).type(torch.FloatTensor).to(device)
    elif lim == 'max':
        with torch.no_grad():
            delta_x = patch_embed(torch.ones(1, 3, 224, 224).to(device))
    elif lim == 'range':
        with torch.no_grad():
            delta_x_range = []
            for i in range(100):
                sample = torch.empty(100, 3, 224, 224).uniform_(-2, 2).type(torch.FloatTensor).to(device)
                delta_x_range.append(patch_embed(sample))
            delta_x_range = torch.cat(delta_x_range, dim=0)
            delta_x_max = delta_x_range.max(dim=0)[0]
            delta_x_min = delta_x_range.min(dim=0)[0]
            delta_x = delta_x_range.mean(dim=0).unsqueeze(0)
        del delta_x_range

    print('Starting magnitude', delta_x.shape, (((delta_x.squeeze(0)) ** 2).sum(dim=0) ** 0.5).mean())

    for i in range(rounds):
        for _, (imgs, _) in enumerate(loader):
            delta_x.requires_grad = True
            imgs = imgs.to(device)

            with torch.no_grad():
                og_preds = model.head(model.forward_features(imgs))

            model.zero_grad()

            x = patch_embed(imgs)
            x = x + delta_x

            preds = model.head(encoder_forward(model, x))
            error_mult = (((preds-og_preds)**2).sum(dim=-1)**0.5).mean()
            error_mult.backward()
            grad = delta_x.grad.data
            with torch.no_grad():
                delta_x -= eps * grad.detach()
                # delta_x = torch.max(torch.min(delta_x, delta_x_max), delta_x_min)
                # delta_x = torch.clamp(delta_x, min=delta_x_min, max=delta_x_max)

            # delta_x.grad.zero_()

        if not i % 20:
            print(i, error_mult.item())
        if i in milestones:
            eps /= 10.0

    return delta_x


def image_level_nullnoise(model, loader, args, delta_x, device):
    model = model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    # del_x_shape = (1, 3, img_size, img_size)
    print('Starting magnitude', delta_x.shape, (((delta_x.squeeze(0)) ** 2).sum(dim=0) ** 0.5).mean())

    eps = args.eps
    for i in range(args.epochs):
        for _, (imgs, _) in enumerate(loader):
            delta_x.requires_grad = True
            imgs = imgs.to(device)

            with torch.no_grad():
                og_preds = model(imgs)

            model.zero_grad()

            imgs = imgs + delta_x

            preds = model(imgs)
            error_mult = (((preds-og_preds)**2).sum(dim=-1)**0.5).mean()
            error_mult.backward()
            grad = delta_x.grad.data
            with torch.no_grad():
                delta_x -= eps * grad.detach()

        if not i % 50:
            print(i, error_mult.item())
        if i in args.milestones:
            eps /= 10.0

    return delta_x
