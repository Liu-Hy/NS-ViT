from typing import Callable, Optional
from utils import SPLITS
import pickle
import torch
from pathlib import Path
from ffrecord import FileReader
from hfai.datasets.base import (
    BaseDataset,
    get_data_dir,
    register_dataset
)
from ffrecord.torch import Dataset, DataLoader

"""
Expected file organization:

    [data_dir]
        train.ffr
            meta.pkl
            PART_00000.ffr
            PART_00001.ffr
            ...
        val.ffr
            meta.pkl
            PART_00000.ffr
            PART_00001.ffr
            ...
"""

@register_dataset
class ImageNetDG(BaseDataset):
    """
    这是一个图像识别数据集的OOD版本集合

    更多信息参考：https://image-net.org

    Args:
        split (str): 数据集划分形式，包括：训练集（``train``）或者验证集（``val``）
        transform (Callable): transform 函数，对图片进行 transfrom，接受一张图片作为输入，输出 transform 之后的图片
        check_data (bool): 是否对每一条样本检验校验和（默认为 ``True``）
        miniset (bool): 是否使用 mini 集合（默认为 ``False``）

    Returns:
        pic, label (PIL.Image.Image, int): 返回的每个样本是一个元组，包括一个RGB格式的图片，以及代表这张图片类别的标签

    Examples:

    .. code-block:: python

        from hfai.datasets import ImageNet
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        dataset = ImageNet(split, transform)
        loader = dataset.loader(batch_size=64, num_workers=4)

        for pic, label in loader:
            # training model

    """

    def __init__(
        self,
        split: str,
        transform: Optional[Callable] = None,
        check_data: bool = True,
    ) -> None:
        super(ImageNetDG, self).__init__()

        assert split in SPLITS
        self.split = split
        self.transform = transform
        self.data_dir = Path("/private_dataset/ImageNet_DG/")
        self.fname = self.data_dir / f"{split}" / "ffrdata"
        self.reader = FileReader(self.fname, check_data)

        with open(self.data_dir / f"{split}" / "meta.pkl", "rb") as fp:
            self.meta = pickle.load(fp)

    def __len__(self):
        return self.reader.n

    def __getitem__(self, indices):

        imgs_bytes = self.reader.read(indices)
        samples = []
        for i, bytes_ in enumerate(imgs_bytes):
            img = pickle.loads(bytes_).convert("RGB")
            label = self.meta["targets"][indices[i]]
            samples.append((img, int(label)))

        transformed_samples = []

        for img, label in samples:
            if self.transform:
                img = self.transform(img)
            transformed_samples.append((img, label))

        return transformed_samples

