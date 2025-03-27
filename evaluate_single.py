"""Evaluation script on the single machine.
Note that it evaluates models on 10-superclass classification rather than 1k classification. """

from pathlib import Path

from torch import nn

from train_single import validate, validate_corruption, prepare_loader
# from torchvision import transforms
from utils import *


def get_val_transform(config, split):
    normalize = transforms.Normalize(config['mean'], config['std'])

    # ImageNet-C and ImageNet-Stylized are already 224 x 224 images
    if split in ["corruption", "stylized"]:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    else:
        val_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])
    return val_transform


def main(args):
    val_batch_size = 256
    val_ratio = 1.
    save_path = Path("output/hal")
    data_path = Path("/var/lib/data/imagenet")
    info_path = Path("info")

    if args.debug:
        val_batch_size = 2
        val_ratio = 0.01

    model_name = 'vit_base_patch16_224'
    ckpt_path = 'pretrained/vit_base_patch16_224-dat.pth.tar'
    model, patch_size, img_size, model_config = get_model_and_config(model_name, ckpt_path, use_ema=True)
    model.cuda()

    criterion = nn.CrossEntropyLoss()

    result = dict()
    # Evaluate on val and OOD datasets except imagenet-c
    for split in SPLITS:
        if split != "train" and not split.startswith("c-"):
            val_transform = get_val_transform(model_config, split)
            typ_path = data_path.joinpath(split)
            val_loader = prepare_loader(typ_path, info_path, val_batch_size, val_transform)
            acc, _ = validate(val_loader, model, criterion, val_ratio)
            result[split] = acc
            print(result)
            if split == "val":
                result["fgsm"] = validate(val_loader, model, criterion, val_ratio, adv=True)[0]
    # Evaluate on imagenet-c
    corruption_rs = validate_corruption(data_path.joinpath("corruption"), info_path, model, val_transform, criterion,
                                        val_batch_size, val_ratio)
    result["corruption"] = corruption_rs["mce"]
    print(result)
    total = get_mean([100 - v if k == "corruption" else v for k, v in result.items()])
    print(f"Avg performance: {total}\n", result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-db', '--debug', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default='none', help='checkpoint path')
    args = parser.parse_args()
    main(args)