import torch
from torch import nn


class DiffusionRunner:
    def __init__(self, model):
        self.model = model
        # Init a adam optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # Init a MSE loss function
        self.loss = nn.MSELoss(reduction='none')
    
    def train_step(self, x):
        self.model.train()
        self.optimizer.zero_grad()
        noise_level = torch.rand(x.size(0), device=x.device)
        original_x = x
        noised_x = self.noise(x, noise_level)
        x = self.model(noised_x, noise_level)
        cost = self.compute_cost(x, original_x)  # (B, C, H, W)
        cost_mean_of_sample = cost.mean(dim=(1, 2, 3))
        cost = cost_mean_of_sample.sum()
        cost.backward()
        self.optimizer.step()
        return cost
        
    def noise(self, x, noise_level:torch.Tensor) -> torch.Tensor:
        """x has shape (B, C, H, W)"""
        assert x.size(0) == noise_level.size(0)
        return x +\
            torch.randn_like(x) * noise_level.reshape(x.size(0), 1, 1, 1)
            
    def evaluate(self, x, noise_level):
        """x has shape of (B, C, H, W)"""
        self.model.eval()
        with torch.no_grad():
            noise_level = torch.ones(x.size(0), device=x.device) * noise_level
            original_x = x
            noised_x = self.noise(x, noise_level)
            x = self.model(noised_x, noise_level)
            cost = self.compute_cost(x, original_x)  # (B, C, H, W)
            cost_mean_of_sample = cost.mean(dim=(1, 2, 3))
            cost = cost_mean_of_sample.sum()
        return cost
        
    
    def compute_cost(self, x, gt):
        return self.loss(x, gt)