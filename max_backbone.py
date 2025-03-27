import numpy as np
from io import BytesIO
from multiprocessing import Pool
from urllib.request import urlopen

import fire
import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
import timm
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy import optimize
from scipy.linalg import null_space
from scipy.optimize import LinearConstraint
from scipy.optimize import lsq_linear
from timm.data import resolve_data_config

sns.set()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)


def get_model_and_config(model_name, pretrained=True):
    print(f'{model_name} is pretrained? {pretrained}')
    model = timm.create_model(model_name, pretrained=pretrained)
    config = resolve_data_config({}, model=model)
    print(config)
    try:
        patch_size = model.patch_embed.patch_size[0]
        img_size = model.patch_embed.img_size[0]
    except:
        patch_size = 32
        img_size = 224
    print(f'{model_name}, {img_size}x{img_size}, patch_size:{patch_size}')
    return model, patch_size, img_size, config


def calculateNullSpace(matrix):
    ns = []
    for i in range(3):
        # What's the dimension of matrix? Seems to be computed for each channel separately.
        ns.append(null_space(matrix[:, i].view(matrix.shape[0], -1).numpy()))
        print('NullSpace MAX, MIN, SHAPE: ', np.max(ns[-1]), np.min(ns[-1]), ns[-1].shape)
    return ns


def empty_gpu():
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def load_images():
    links = [
        "https://github.com/hila-chefer/Transformer-Explainability/blob/main/samples/catdog.png?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n02364673_guinea_pig.JPEG?raw=true",
        "https://github.com/hila-chefer/Transformer-Explainability/blob/main/samples/dogbird.png?raw=true",
        "https://github.com/hila-chefer/Transformer-Explainability/blob/main/samples/dogcat2.png?raw=true",
        "https://github.com/hila-chefer/Transformer-Explainability/blob/main/samples/el1.png?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n03891332_parking_meter.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n03095699_container_ship.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n02002556_white_stork.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n07747607_orange.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n03630383_lab_coat.JPEG?raw=true",
        "https://github.com/ajschumacher/imagen/blob/master/imagen/n02924116_88878_bus.jpg?raw=true",
        "https://github.com/ajschumacher/imagen/blob/master/imagen/n01910747_17159_jellyfish.jpg?raw=true",
        "https://github.com/ajschumacher/imagen/blob/master/imagen/n01882714_11334_koala_bear.jpg?raw=true",
        "https://github.com/ajschumacher/imagen/blob/master/imagen/n01784675_11489_centipede.jpg?raw=true",
        "https://github.com/ajschumacher/imagen/blob/master/imagen/n01639765_51193_frog.jpg?raw=true",
        "https://github.com/ajschumacher/imagen/blob/master/imagen/n03372029_39768_flute.jpg?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01828970_bee_eater.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01980166_fiddler_crab.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n02177972_weevil.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n02514041_barracouta.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n02494079_squirrel_monkey.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n03690938_lotion.JPEG?raw=true",
        "https://github.com/ajschumacher/imagen/blob/master/imagen/n03676483_24567_lipstick.jpg?raw=true",
        "https://github.com/ajschumacher/imagen/blob/master/imagen/n03720891_10274_maraca.jpg?raw=true",
        "https://github.com/ajschumacher/imagen/blob/master/imagen/n02958343_8827_car.jpg?raw=true",
        "https://github.com/ajschumacher/imagen/blob/master/imagen/n02970849_4696_cart.jpg?raw=true",
        "https://github.com/ajschumacher/imagen/blob/master/imagen/n03062245_872_cocktail_shaker.jpg?raw=true",
        "https://github.com/ajschumacher/imagen/blob/master/imagen/n03141823_6920_crutch.jpg?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n02256656_cicada.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01443537_goldfish.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01514859_hen.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01558993_robin.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01491361_tiger_shark.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n02100583_vizsla.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n02114548_white_wolf.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n02422699_impala.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n03394916_French_horn.JPEG?raw=true",
        "https://dl.fbaipublicfiles.com/dino/img.png",
        "https://github.com/ajschumacher/imagen/blob/master/imagen/n01662784_244_turtle.jpg?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n02099267_flat-coated_retriever.JPEG?raw=true",
        "https://github.com/ajschumacher/imagen/blob/master/imagen/n02129165_16177_lion.jpg?raw=true",
        "https://github.com/ajschumacher/imagen/blob/master/imagen/n02411705_3709_sheep.jpg?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01855672_goose.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01806143_peacock.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01807496_partridge.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01795545_black_grouse.JPEG?raw=true"
    ]
    images = []

    for link in links:
        response = requests.get(link)
        img = Image.open(BytesIO(response.content))
        images.append(img.convert('RGB'))
    return images


cls_map_link = 'https://github.com/Waikato/wekaDeeplearning4j/blob/master/src/main/resources/class-maps/IMAGENET.txt?raw=True'
data = urlopen(cls_map_link).read().decode('utf-8').split('\n')
cls_map = {}
for i, line in enumerate(data):
    cls_map[i] = line.strip().split(',')[0]
print(cls_map[0], cls_map[1])


def inference_og_and_delx(net, og_img, img):
    transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    og_img = transform(og_img)

    preds = net(torch.cat([og_img.unsqueeze(0), (img + og_img).unsqueeze(0)], dim=0).to(device)).cpu()
    mse_logits = ((preds[0] - preds[1]) ** 2).sum().item()

    probs = torch.softmax(preds, dim=-1)
    mse_probs = ((probs[0] - probs[1]) ** 2).sum().item()

    p, c = torch.max(probs, dim=-1)

    errors = f"MSE LOGITS: {mse_logits:.4f} -- MSE PROBS: {mse_probs:.4f}"

    fig, axs = plt.subplots(1, 3, figsize=(11, 5))

    title_0 = f"CLS: {cls_map[c[0].item()]} -- PROB: {p[0].item():.4f}"
    title_1 = f"CLS: {cls_map[c[1].item()]} -- PROB: {p[1].item():.4f}"
    axs[0].imshow(og_img.permute(1, 2, 0).numpy())
    axs[0].axis('off')
    axs[0].set_title(title_0)

    axs[1].imshow(torch.clip(og_img + img, 0, 1).permute(1, 2, 0).numpy())
    axs[1].axis('off')
    axs[1].set_title(title_1)

    axs[2].imshow(torch.clip(img, 0, 1).permute(1, 2, 0).numpy())
    axs[2].axis('off')
    axs[2].set_title('Del x')

    plt.suptitle(errors)
    plt.show()
    plt.close()


def checkLegal(imgResult):
    a = np.min(imgResult)
    b = np.max(imgResult)
    return a >= -1 and b <= 1


def greedy_max(idx, img_patches, patchWeights):
    for c in range(3):
        current_channel_patch = patchWeights[c]
        og_patch = np.copy(img_patches[c])
        patch_size = og_patch.shape[0]
        bestDist = 0
        for k in range(current_channel_patch.shape[-1]):
            bestWeights = np.zeros([patch_size, patch_size])
            patch = np.copy(img_patches[c])
            s = 50.0
            while s >= 1:
                pwk = (current_channel_patch[:, k].reshape((patch_size, patch_size))) * s
                if checkLegal(patch + pwk):
                    dist = np.sum(np.abs(og_patch - patch - pwk))
                    if dist > bestDist:
                        bestDist = dist
                        bestWeights = np.copy(pwk)
                pwk = -(current_channel_patch[:, k].reshape((patch_size, patch_size))) * s
                if checkLegal(patch + pwk):
                    dist = np.sum(np.abs(og_patch - patch - pwk))
                    if dist > bestDist:
                        bestDist = dist
                        bestWeights = np.copy(pwk)
                s -= 0.05
            img_patches[c] += bestWeights
    return (idx, img_patches)


def patch_parallel_impose(src_patch, n_space, idx):
    func = lambda x, A, y: np.sum(y - A.dot(x))
    # func = lambda x, A, y: -1.0*np.sum(np.abs(y-A@x))
    tar_r = src_patch[0].view(-1)
    tar_g = src_patch[1].view(-1)
    tar_b = src_patch[2].view(-1)
    _, patch_size, _ = src_patch.shape

    bounds_r_u = (1 - src_patch[0]).numpy().reshape(-1)
    bounds_r_l = (-1 - src_patch[0]).numpy().reshape(-1)
    sol_r = optimize.minimize(func,
                              x0=torch.zeros(n_space[0].shape[-1]).numpy().astype(np.double),
                              method='SLSQP',
                              constraints=LinearConstraint(n_space[0], bounds_r_l, bounds_r_u),
                              options={'maxiter': 1000},
                              args=(n_space[0], tar_r.numpy()))
    f_r = torch.tensor(n_space[0]) @ sol_r.x

    bounds_g_u = (1 - src_patch[1]).numpy().reshape(-1)
    bounds_g_l = (-1 - src_patch[1]).numpy().reshape(-1)
    sol_g = optimize.minimize(func,
                              x0=torch.zeros(n_space[1].shape[-1]).numpy().astype(np.double),
                              method='SLSQP',
                              constraints=LinearConstraint(n_space[1], bounds_g_l, bounds_g_u),
                              options={'maxiter': 1000},
                              args=(n_space[1], tar_g.numpy()))
    f_g = torch.tensor(n_space[1]) @ sol_g.x

    bounds_b_u = (1 - src_patch[2]).numpy().reshape(-1)
    bounds_b_l = (-1 - src_patch[2]).numpy().reshape(-1)
    sol_b = optimize.minimize(func,
                              x0=torch.zeros(n_space[2].shape[-1]).numpy().astype(np.double),
                              method='SLSQP',
                              constraints=LinearConstraint(n_space[2], bounds_b_l, bounds_b_u),
                              options={'maxiter': 1000},
                              args=(n_space[2], tar_b.numpy()))

    f_b = torch.tensor(n_space[2]) @ sol_b.x
    f = torch.cat([f_r.view(1, patch_size, patch_size),
                   f_g.view(1, patch_size, patch_size), f_b.view(1, patch_size, patch_size)], dim=0)
    return (idx, f + src_patch.view(-1, patch_size, patch_size))


def backbone_parallel_impose(src_patch, del_y, weights, idx):
    tars = del_y.view(-1)

    bounds_r_u = (1 - src_patch[0]).numpy().reshape(-1)
    bounds_r_l = (-1 - src_patch[0]).numpy().reshape(-1)
    bounds_g_u = (1 - src_patch[1]).numpy().reshape(-1)
    bounds_g_l = (-1 - src_patch[1]).numpy().reshape(-1)
    bounds_b_u = (1 - src_patch[2]).numpy().reshape(-1)
    bounds_b_l = (-1 - src_patch[2]).numpy().reshape(-1)
    bounds_l = np.concatenate([bounds_r_l, bounds_g_l, bounds_b_l])
    bounds_u = np.concatenate([bounds_r_u, bounds_g_u, bounds_b_u])
    sols = lsq_linear(weights, tars.numpy().astype(np.double), bounds=(bounds_l, bounds_u), verbose=0, max_iter=1000).x
    return (idx, src_patch + torch.tensor(sols).view(src_patch.shape))


def create_batches(simg, weights, img_size, patch_size=32, lp=2):
    batches = []
    simg = torch.clip(simg, -1, 1)

    for i in range(0, img_size, patch_size):
        for j in range(0, img_size, patch_size):
            sol_idx = int(i // patch_size * (img_size // patch_size) + j // patch_size)
            i_r = simg[0][i:i + patch_size, j:j + patch_size].unsqueeze(0)
            i_g = simg[1][i:i + patch_size, j:j + patch_size].unsqueeze(0)
            i_b = simg[2][i:i + patch_size, j:j + patch_size].unsqueeze(0)
            batches.append((
                torch.cat([i_r, i_g, i_b], dim=0),
                weights,
                sol_idx,
            ))
    return batches


def create_greedy_batches(simg, weights, img_size, patch_size=32):
    batches = []
    simg = torch.clip(simg, -1, 1)

    for i in range(0, img_size, patch_size):
        for j in range(0, img_size, patch_size):
            sol_idx = int(i // patch_size * (img_size // patch_size) + j // patch_size)
            i_r = simg[0][i:i + patch_size, j:j + patch_size].unsqueeze(0)
            i_g = simg[1][i:i + patch_size, j:j + patch_size].unsqueeze(0)
            i_b = simg[2][i:i + patch_size, j:j + patch_size].unsqueeze(0)
            batches.append((
                sol_idx,
                torch.cat([i_r, i_g, i_b], dim=0),
                weights,
            ))
    return batches


def create_backbone_batches(simg, del_y, weights, img_size, patch_size=32):
    batches = []
    simg = torch.clip(simg, -1, 1)

    for i in range(0, img_size, patch_size):
        for j in range(0, img_size, patch_size):
            sol_idx = int(i // patch_size * (img_size // patch_size) + j // patch_size)
            src_r = simg[0][i:i + patch_size, j:j + patch_size].unsqueeze(0)
            src_g = simg[1][i:i + patch_size, j:j + patch_size].unsqueeze(0)
            src_b = simg[2][i:i + patch_size, j:j + patch_size].unsqueeze(0)
            batches.append((torch.cat([src_r, src_g, src_b], dim=0),
                            del_y[sol_idx],
                            weights,
                            sol_idx
                            ))
    return batches


def create_mod(outputs, img_size=224, patch_size=32):
    mod_img = torch.zeros((3, img_size, img_size))
    num_per_row = img_size // patch_size
    for output in outputs:
        idx, vals = output
        r, c = idx // num_per_row, idx % num_per_row
        r, c = r * patch_size, c * patch_size
        mod_img[0][r:r + patch_size, c:c + patch_size] = vals[0]  # .view(patch_size, patch_size)
        mod_img[1][r:r + patch_size, c:c + patch_size] = vals[1]  # .view(patch_size, patch_size)
        mod_img[2][r:r + patch_size, c:c + patch_size] = vals[2]  # .view(patch_size, patch_size)
    return mod_img


def predict(net, mod_img, img, mean, std):
    fn = lambda x, i: mean[i] + x * std[i]
    vimg = torch.cat([fn(img[0], 0).unsqueeze(0), fn(img[1], 1).unsqueeze(0), fn(img[2], 2).unsqueeze(0)], dim=0)
    vmod = torch.cat([fn(mod_img[0], 0).unsqueeze(0), fn(mod_img[1], 1).unsqueeze(0), fn(mod_img[2], 2).unsqueeze(0)],
                     dim=0)
    print('mod_img', torch.max(mod_img), torch.min(mod_img))
    print('view img', torch.max(vimg), torch.min(vimg))
    print('view mod', torch.max(vmod), torch.min(vmod))
    print(vimg.shape, vmod.shape)

    d_img = img - mod_img
    idx = torch.randperm(mod_img.nelement())
    pimg = d_img.reshape(-1)[idx].reshape(mod_img.size()) + img

    imgs = torch.cat([img.unsqueeze(0), mod_img.unsqueeze(0), pimg.unsqueeze(0)], dim=0)
    with torch.no_grad():
        preds = torch.softmax(net(imgs), dim=-1)
    ps, cs = torch.max(preds, dim=-1)
    print(f'SRC: {ps[0]:.4f} {cls_map[cs[0].item()]}')
    print(f'MOD: {ps[1]:.4f} {cls_map[cs[1].item()]}')
    print(f'PER: {ps[2]:.4f} {cls_map[cs[2].item()]}')

    return vmod, vimg


def main(model_name, img_size=224, lp=2):
    images = load_images()

    rnet, _, _, rconf = get_model_and_config('resnet50', pretrained=True)
    rnorm = transforms.Normalize(rconf['mean'], rconf['std'])
    print('resnet', rconf)
    rnet.eval()

    net, patch_size, img_size, model_config = get_model_and_config(model_name, pretrained=True)
    net.eval()
    normalize = transforms.Normalize(model_config['mean'], model_config['std'])
    transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(), normalize])
    weights = net.patch_embed.proj.weight.detach().cpu()
    n_space = calculateNullSpace(weights)

    for i in range(3):
        print(np.max(n_space[i]), np.min(n_space[i]))

    weights = weights.view(weights.shape[0], -1).numpy()
    result = {}
    for lim in [1.0, 2.0]:
        m = model_name.split('_')[1]
        del_name = f'./outputs/enc_{m}_p32_im224_eps_0.01_rounds_500_lim_{lim}.v1.pth'
        delta_y = torch.load(del_name)['delta_y'].detach()[0]
        result[lim] = []
        for img_id, image in enumerate(images):
            img = transform(image)
            batches = create_backbone_batches(img, delta_y, weights, img_size)
            with Pool(8) as p:
                outputs = p.starmap(backbone_parallel_impose, batches)
            mod_img = create_mod(outputs)
            result[lim].append(mod_img)
            predict(net, mod_img, img, model_config['mean'], model_config['std'])
            '''
            batches = create_greedy_batches(img, n_space, img_size)
            with Pool(32) as p:
              outputs_p = p.starmap(greedy_max, batches)
            mod_img_p = create_mod(outputs_p)
            #result['patch'][lim].append(mod_img_p)
            rm_img, r_img = predict(net, mod_img_p, img, model_config['mean'], model_config['std'])
            predict(rnet, rnorm(rm_img), rnorm(r_img), rconf['mean'], rconf['std'])
            result[img_id] = {'greedy': mod_img_p}
            batches = create_batches(img, n_space, img_size)
            print('--- opt ---')
            with Pool(32) as p:
              outputs_p = p.starmap(patch_parallel_impose, batches)
            mod_img_m= create_mod(outputs_p)
            rm_img, r_img = predict(net, mod_img_m, img, model_config['mean'], model_config['std'])
            predict(rnet, rnorm(rm_img), rnorm(r_img), rconf['mean'], rconf['std'])
            result[img_id]['max'] = mod_img_m
            '''
            print('*' * 10, img_id)
            torch.save(result, f'./outputs/{model_name}.robust.backbone')
            if img_id >= 4:
                break


if __name__ == '__main__':
    fire.Fire(main)
