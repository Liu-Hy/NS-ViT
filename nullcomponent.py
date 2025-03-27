from io import BytesIO
from multiprocessing import Pool
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
import timm
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from scipy.linalg import null_space
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


def backbone_parallel_impose(_, idx):
    tars = delta_y[idx].view(-1)

    bounds_l = -1 * torch.ones(3 * 1024)
    bounds_u = torch.ones(3 * 1024)
    sols = lsq_linear(weights, tars.numpy().astype(np.double), bounds=(bounds_l, bounds_u), verbose=0, max_iter=10000).x
    return (torch.tensor(sols.reshape(3, -1)), idx)


def create_mod(outputs):
    mod_img = torch.zeros((3, img_size, img_size))
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


model_name = 'vit_base_patch32_224'  # 'vit_small_patch16_224'
net, patch_size, img_size, model_config = get_model_and_config(model_name, pretrained=True)
m = model_name.split('_')[1]
net, patch_size, img_size, model_config = get_model_and_config(model_name, pretrained=True)
net.eval()
rounds, eps, milestones, err_fac, mag_fac = 500, 0.01, [150, 300, 400], 1.0, 0.0

normalize = transforms.Normalize(model_config['mean'], model_config['std'])
transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(), normalize])
weights = net.patch_embed.proj.weight.detach().cpu()
n_space = calculateNullSpace(weights)
weights = weights.view(weights.shape[0], -1).numpy()

val_dataset = datasets.ImageFolder('../data/imagenette2/val', transforms.Compose([
    transforms.Resize((img_size, img_size)), transforms.ToTensor(),
    transforms.Normalize(model_config['mean'], model_config['std'])]))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=32, pin_memory=True)

results = {}

net = net.to(device)

for lim in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0]:
    del_name = f'outputs/enc_{m}_p{patch_size}_im{img_size}_eps_{eps}_rounds_{rounds}_lim_{lim}.v1.pth'
    delta_y = torch.load(del_name)['delta_y'].detach()[0]
    print(weights.shape, n_space[0].shape, delta_y.shape)
    batches = [(None, i) for i in range(delta_y.shape[0])]
    with Pool(32) as p:
        outputs_p = p.starmap(backbone_parallel_impose, batches)
    mod_img = create_mod(outputs_p).to(device)
    idx = torch.randperm(mod_img.nelement())
    pimg = mod_img.reshape(-1)[idx].reshape(mod_img.size())

    print(torch.min(mod_img), torch.max(mod_img), mod_img.shape)
    mod_img = mod_img.unsqueeze(0)
    pimg = pimg.unsqueeze(0)

    correct = 0
    shuffle = 0
    total = 0

    for _, (imgs, _) in enumerate(val_loader):
        imgs = imgs.to(device)
        with torch.no_grad():
            preds = net(imgs)
            _, indices = torch.max(preds, dim=-1)
            _, mind = torch.max(net(imgs + mod_img), dim=-1)
            _, pind = torch.max(net(imgs + pimg), dim=-1)
            correct += (mind == indices).sum()
            shuffle += (indices == pind).sum()
            total += imgs.shape[0]
        print(correct / total, shuffle / total, total)

    results[lim] = {'corr': {'eq': correct / total}, 'shuffle': {'eq': shuffle / total}}
    torch.save(results, f'{model_name}.val.nu_theta')
