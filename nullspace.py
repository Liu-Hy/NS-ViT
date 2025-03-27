import numpy as np
from io import BytesIO
from urllib.request import urlopen

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


def patch_parallel_impose(src_patch, tar_patch, idx):
    tar_r = (tar_patch[0] - src_patch[0]).view(-1)
    tar_g = (tar_patch[1] - src_patch[1]).view(-1)
    tar_b = (tar_patch[2] - src_patch[2]).view(-1)

    # func = lambda x, A, y: np.linalg.norm(y-A@x)
    func = lambda x, A, y: np.sum(np.abs(y - A @ x))
    bounds_r_u = (1 - src_patch[0]).numpy().reshape(-1)
    bounds_r_l = (-1 - src_patch[0]).numpy().reshape(-1)
    sol_r = optimize.minimize(func,
                              x0=torch.ones(n_space[0].shape[-1]).numpy().astype(np.double),
                              method='SLSQP',
                              constraints=LinearConstraint(n_space[0], bounds_r_l, bounds_r_u),
                              options={'maxiter': 5000},
                              args=(n_space[0], tar_r.numpy())).x

    bounds_g_u = (1 - src_patch[1]).numpy().reshape(-1)
    bounds_g_l = (-1 - src_patch[1]).numpy().reshape(-1)
    sol_g = optimize.minimize(func,
                              x0=torch.ones(n_space[1].shape[-1]).numpy().astype(np.double),
                              method='SLSQP',
                              constraints=LinearConstraint(n_space[1], bounds_g_l, bounds_g_u),
                              options={'maxiter': 5000},
                              args=(n_space[1], tar_g.numpy())).x

    bounds_b_u = (1 - src_patch[2]).numpy().reshape(-1)
    bounds_b_l = (-1 - src_patch[2]).numpy().reshape(-1)
    sol_b = optimize.minimize(func,
                              x0=torch.ones(n_space[2].shape[-1]).numpy().astype(np.double),
                              method='SLSQP',
                              constraints=LinearConstraint(n_space[2], bounds_b_l, bounds_b_u),
                              options={'maxiter': 5000},
                              args=(n_space[2], tar_b.numpy())).x

    f_r = torch.tensor(n_space[0]) @ sol_r
    f_g = torch.tensor(n_space[1]) @ sol_g
    f_b = torch.tensor(n_space[2]) @ sol_b

    f = torch.cat([f_r.view(1, -1), f_g.view(1, -1), f_b.view(1, -1)], dim=0)
    return (f, idx)


def backbone_parallel_impose(src_patch, tar_patch, idx):
    tars = delta_y[idx].view(-1)

    bounds_r_u = (1 - src_patch[0]).numpy().reshape(-1)
    bounds_r_l = (-1 - src_patch[0]).numpy().reshape(-1)
    bounds_g_u = (1 - src_patch[1]).numpy().reshape(-1)
    bounds_g_l = (-1 - src_patch[1]).numpy().reshape(-1)
    bounds_b_u = (1 - src_patch[2]).numpy().reshape(-1)
    bounds_b_l = (-1 - src_patch[2]).numpy().reshape(-1)
    bounds_l = np.concatenate([bounds_r_l, bounds_g_l, bounds_b_l])
    bounds_u = np.concatenate([bounds_r_u, bounds_g_u, bounds_b_u])
    sols = lsq_linear(weights, tars.numpy().astype(np.double), bounds=(bounds_l, bounds_u), verbose=0, max_iter=10000).x
    return (torch.tensor(sols.reshape(3, -1)), idx)


def create_batches(simg, timg):
    batches = []
    simg = torch.clip(simg, -1, 1)
    timg = torch.clip(timg, -1, 1)

    for i in range(0, img_size, patch_size):
        for j in range(0, img_size, patch_size):
            sol_idx = int(i // patch_size * (img_size // patch_size) + j // patch_size)
            tar_r = timg[0][i:i + patch_size, j:j + patch_size].reshape(1, -1)
            src_r = simg[0][i:i + patch_size, j:j + patch_size].reshape(1, -1)
            tar_g = timg[1][i:i + patch_size, j:j + patch_size].reshape(1, -1)
            src_g = simg[1][i:i + patch_size, j:j + patch_size].reshape(1, -1)
            tar_b = timg[2][i:i + patch_size, j:j + patch_size].reshape(1, -1)
            src_b = simg[2][i:i + patch_size, j:j + patch_size].reshape(1, -1)
            batches.append((torch.cat([src_r, src_g, src_b], dim=0),
                            torch.cat([tar_r, tar_g, tar_b], dim=0),
                            sol_idx
                            ))
    return batches


def create_mod(outputs, src_img):
    mod_img = torch.zeros((3, img_size, img_size)) + src_img
    num_per_row = img_size // patch_size
    for output in outputs:
        vals, idx = output
        r, c = idx // num_per_row, idx % num_per_row
        r, c = r * patch_size, c * patch_size
        mod_img[0][r:r + patch_size, c:c + patch_size] += vals[0].view(patch_size, patch_size)
        mod_img[1][r:r + patch_size, c:c + patch_size] += vals[1].view(patch_size, patch_size)
        mod_img[2][r:r + patch_size, c:c + patch_size] += vals[2].view(patch_size, patch_size)
    return mod_img


def predict_save(mod_img, fname='sample'):
    fn = lambda x, i: model_config['mean'][i] + x * model_config['std'][i]
    view_mod_img = torch.cat(
        [fn(mod_img[0], 0).unsqueeze(0), fn(mod_img[1], 1).unsqueeze(0), fn(mod_img[2], 2).unsqueeze(0)])
    print(torch.max(mod_img), torch.min(mod_img))
    print(torch.max(view_mod_img), torch.min(view_mod_img))
    imgs = torch.cat([img.unsqueeze(0), mod_img.unsqueeze(0)], dim=0).to(device)
    with torch.no_grad():
        preds = torch.softmax(net(imgs), dim=-1)
    ps, cs = torch.max(preds, dim=-1)
    # fig, axs = plt.subplots(1, 1, figsize=(6, 6))
    print(f'MOD: {ps[1]:.4f} {cls_map[cs[1].item()]}')
    print(f'SRC: {ps[0]:.4f} {cls_map[cs[0].item()]}')
    # axs.imshow(view_mod_img.permute(1,2,0))
    # axs.axis('off')
    # plt.show()
    # plt.savefig(f'{fname}.png')
    # plt.close()
    return abs(ps[1] - ps[0]), cs[1] == cs[0]


model_name = 'vit_small_patch32_224'  # 'vit_small_patch16_224'
del_name = './outputs/enc_small_p32_im224_eps_0.01_rounds_500_lim_0.1.v1.pth'
images = load_images()

net, patch_size, img_size, model_config = get_model_and_config(model_name, pretrained=True)
net = net.to(device)
net.eval()
normalize = transforms.Normalize(model_config['mean'], model_config['std'])
transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(), normalize])
weights = net.patch_embed.proj.weight.detach().cpu()
n_space = calculateNullSpace(weights)
weights = weights.view(weights.shape[0], -1).numpy()
delta_y = torch.load(del_name)['delta_y'].detach()[0]
print(weights.shape, n_space[0].shape, delta_y.shape)

response = requests.get('https://miro.medium.com/max/3150/2*mXZKj2OVMaWGW97xXA00aQ.png')
iclr_img = normalize(
    transforms.ToTensor()(Image.open(BytesIO(response.content)).convert('RGB').resize((img_size, img_size))))
# zero_img = torch.zeros((3, img_size, img_size))
# for i in range(0, img_size, patch_size):
#  for j in range(0, img_size, patch_size):
#    zero_img[:, i:i+patch_size, j:j+patch_size] = iclr_img[:]
# iclr_img = zero_img
tp, tc = 0, 0
for idx, image in enumerate(images):
    img = transform(image)

    # batches = create_batches(img, iclr_img)
    # with Pool(32) as p:
    #      outputs_p = p.starmap(backbone_parallel_impose, batches)
    #  mod_img = create_mod(outputs_p, img)
    #  torch.save(mod_img, './outputs/patches_and_backbone.backbone.pth')
    # else:
    #  mod_img = torch.load('./outputs/patches_and_backbone.backbone.pth')

    # predict_save(mod_img, 'patches_and_backbone.backbone')

    # batches = create_batches(img, iclr_img)
    # with Pool(32) as p:
    #    outputs_b = p.starmap(patch_parallel_impose, batches)
    # mod_img = create_mod(outputs_b, img)
    # torch.save(mod_img, f'./outputs/images/l1/{idx}.patches.pth')
    # predict_save(mod_img)

    mod_img = torch.load(f'./outputs/images/{idx}.patches.pth')
    ps, cs = predict_save(mod_img)
    tp += ps
    tc += cs
    print(tp / (idx + 1), tc / (idx + 1))
