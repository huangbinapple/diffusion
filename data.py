from typing import Optional
import torch
import torchvision
import torchvision.transforms as transforms


def load_mnist_dataset(train: Optional[bool] = True):
    ds = torchvision.datasets.MNIST(
        '~/dataset', train=train, transform=transforms.ToTensor()
    )
    return ds
  

def get_loader(train:Optional[bool]=True):
    """
    Returns a DataLoader for the MNIST dataset.
    """
    return torch.utils.data.DataLoader(
        load_mnist_dataset(train=train),
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )