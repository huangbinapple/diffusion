import torch
from torch import nn
from einops.layers.torch import Rearrange


class UNet(nn.Module):
    def __init__(self, noise_dim:int=128):
        super().__init__()
        # Downsample
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.downsample1 = nn.Conv2d(8, 8, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=3, padding=2)
        self.downsample2 = nn.Conv2d(32, 32, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.downsample3 = nn.Conv2d(64, 128, kernel_size=2, stride=2)
        # Upsample
        self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.upsample2 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(32, 8, kernel_size=3)
        self.upsample3 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(8, 1, kernel_size=3, padding=1)
        # Relu
        self.relu = nn.ReLU()
        # Noise related
        self.noise_dim = noise_dim
        self.noise_injections = nn.ModuleDict()
        for i in (28, 16, 8, 4):
            self.noise_injections[f'{i}'] = nn.Sequential(
                nn.Linear(noise_dim, i * i),
                Rearrange('B (i j) -> B 1 i j', i=i, j=i)
            )
        
    def encode_noise_level(
            self, noise_level:float, device='cpu') -> None:
        """Encode noise level into a one-hot representation"""
        batch_size = noise_level.size(0)
        result = torch.zeros(batch_size, self.noise_dim, device=device)
        result[range(batch_size), (noise_level * self.noise_dim).long()] = 1
        return result
    
    def forward(self, x, noise_level:torch.Tensor):
        noise_onehot = self.encode_noise_level(noise_level, device=x.device)
        noise_embedding = {
            i: self.noise_injections[f'{i}'](noise_onehot)
            for i in (28, 16, 8, 4)
        }
        # Downsample
        x_28 = x  # (B, 1, 28, 28)
        x = self.conv1(x) + noise_embedding[28]  # (B, 8, 28, 28)
        x = self.relu(x)
        x = self.downsample1(x) # (B, 8, 14, 14)
        x = self.relu(x)
        x_16 = x = self.conv2(x) + noise_embedding[16]  # (B, 32, 16, 16)
        x = self.relu(x)
        x = self.downsample2(x) # (B, 32, 8, 8)
        x = self.relu(x)
        x_8 = x = self.conv3(x) + noise_embedding[8] # (B, 64, 8, 8)
        x = self.relu(x)
        x = self.downsample3(x) + noise_embedding[4] # (B, 128, 4, 4)
        x = self.relu(x)
        # Upsample
        x = self.upsample1(x) + x_8 + noise_embedding[8] # (B, 64, 8, 8)
        x = self.relu(x)
        x = self.conv4(x) # (B, 32, 8, 8)
        x = self.relu(x)
        x = self.upsample2(x) + x_16 + noise_embedding[16] # (B, 32, 16, 16)
        x = self.relu(x)
        x = self.conv5(x) # (B, 8, 14, 14)
        x = self.relu(x)
        x = self.upsample3(x) + noise_embedding[28] # (B, 8, 28, 28)
        x = self.relu(x)
        x = self.conv6(x) + x_28 # (B, 1, 28, 28)
        return x
    
    
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Downsample
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.downsample1 = nn.Conv2d(8, 8, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=3, padding=2)
        self.downsample2 = nn.Conv2d(32, 32, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.downsample3 = nn.Conv2d(64, 128, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4)
        self.output = nn.Linear(256, 10)
        # Relu
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)  # (B, 8, 28, 28)
        x = self.relu(x)
        x = self.downsample1(x)  # (B, 8, 14, 14)
        x = self.relu(x)
        x = self.conv2(x)  # (B, 32, 16, 16)
        x = self.relu(x)
        x = self.downsample2(x)  # (B, 32, 8, 8)
        x = self.relu(x)
        x = self.conv3(x)  # (B, 64, 8, 8)
        x = self.relu(x)
        x = self.downsample3(x)  # (B, 128, 4, 4)
        x = self.relu(x)
        x = self.conv4(x)  # (B, 256, 1, 1)
        x = self.relu(x)
        x = x.squeeze(2, 3)  # (B, 256)
        logits = self.output(x)  # (B, 10)
        return logits