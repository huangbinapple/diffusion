import torch
from torch import nn
import utils


class DiffusionRunner:
    def __init__(self, model):
        self.model = model
        self.checkpoint_manager = utils.CheckPointManager(
            'checkpoint5', high_is_better=False)
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
        noise = torch.randn_like(x)
        noise_level = noise_level.reshape(x.size(0), 1, 1, 1)
        return torch.sqrt(noise_level) * noise + torch.sqrt(1 - noise_level) * x
            
    def evaluate_on_samples(self, x, noise_level):
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

    def evaluate(self, test_loader, n_iter) -> int:
        noise_level_to_test = (0.1, 0.3, 0.5, 0.7, 0.9)
        costs = torch.zeros(len(noise_level_to_test))
        for batch in test_loader:
            image_tensor, label = batch
            for i, noise_level in enumerate(noise_level_to_test):
                costs[i] += self.evaluate_on_samples(
                    image_tensor.to('cuda:0'), noise_level).cpu()
        costs = costs / len(test_loader)
        score = costs.mean().item()
        self.checkpoint_manager.store_checkpoint(self, n_iter, score)
        return {'score': score, 'costs': costs}
    
    def compute_cost(self, x, gt):
        return self.loss(x, gt)