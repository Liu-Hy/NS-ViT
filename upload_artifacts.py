import os
import wandb
import torch
from dotenv import load_dotenv
load_dotenv()
os.makedirs('artifact/', exist_ok=True)


wandb.login(key=os.getenv('KEY'))
artifact = wandb.Artifact('input', type='dataset')
os.makedirs('artifacts/input', exist_ok=False)

input_shape = (1, 3, 224, 224)
lims = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0]

results = {
    'img_shape': input_shape[-1],
    'lims': lims,
    'delta_x': {}
}

for lim in lims:
    results['delta_x'][lim] = torch.empty(input_shape).uniform_(-lim, lim).type(torch.FloatTensor)

torch.save(results, f'.artifacts/input/init.pth')
artifact.add_dir('.artifacts/input/')
wandb.log_artifact(artifact)
