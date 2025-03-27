"""Fetch an image with ns noise, process it to get the ns noise, and test its agreement with the predictions on the
original images by 4 models. As comparison, shuffle the noise to see the agreement.

"""

import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import get_model_and_config, validate_by_parts
from methods import compute_encoder_ns_v1
import fire
import requests
from PIL import Image
from io import BytesIO


def main():
    data_dir = '../data/imagenette2/'

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    m = model_name.split('_')[1]
    results = {}
    
    #sample noise
    response = requests.get("https://github.com/hila-chefer/Transformer-Explainability/blob/main/samples/catdog.png?raw=true")
    base_img = Image.open(BytesIO(response.content)).convert('RGB')
    nullnoise = torch.load('vit_small_patch32_224.robust.patch')[0]['max'] 
    unnorm = lambda x, c, i: c['mean'][i]+x*c['std'][i]
    _, _, _, config = get_model_and_config('vit_small_patch32_224', pretrained=True)
    # where is the "sc" in code?
    nullnoise = torch.cat([unnorm(nullnoise[0], sc, 0).unsqueeze(0), unnorm(nullnoise, sc, 1).unsqueeze(0),
                            unnorm(nullnoise[2], sc, 2).unsqueeze(0)], dim=0) - tf(base_img) 
    idx = torch.randperm(nullnoise.nelement())
    perm_noise = nullnoise.reshape(-1)[idx].reshape(nullnoise.size())

    # validation
    for model_name in ['vit_small_patch32_224', 'vit_base_patch32_224', 'resnet50', 'efficientnet_b0']:
        model = model.to(device)
        model.eval()
        
        tf = tf.Compose([tf.Resize((224, 224)), tf.ToTensor()])
        norm = transforms.Normalize(model_config['mean'], model_config['std'])
        apply_norm = lambda x: norm(x)
        val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), tf)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=32, pin_memory=True)
        
        c, inc, t = 0, 0, 0
        for _, (imgs, _) in enumerate(val_loader):
            with torch.no_grad():
                ns_imgs = map(apply_norm)(imgs + nullnoise)
                ns_imgs = og_imgs.to(device)  # ns_imgs?
                perm_imgs = map(apply_norm)(imgs + perm_noise)
                perm_imgs = perm_imgs.to(device)
                imgs = map(apply_norm)(imgs)
                imgs = imgs.to(device)

                _, clss = torch.max(model(imgs), dim=-1)
                _, n_clss = torch.max(model(ns_imgs), dim=-1)
                _, p_clss = torch.max(model(perm_imgs), dim=-1)
                 
                
                c += (clss == n_clss).sum()
                inc += (clss == p_clss).sum()
                t += imgs.shape[0]
        results[model_name] = {'corr': c/t, 'shuffle': inc/t}
        print(model_name, results[model_name])
    torch.save('outputs/cross_robustness.nullspace')  # What is saved to this path?

                
               

if __name__ == '__main__':
  fire.Fire(main)
