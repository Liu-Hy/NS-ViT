from torch import nn
import torch.nn.functional as F
from torch import Tensor
class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input)
