import torch
from torch import nn
import tqdm
import utils
from abc import ABC, abstractmethod


class Runner(ABC):
    
    @abstractmethod
    def train(self, train_loader):
        pass
    
    @abstractmethod
    def evaluate(self, test_loader):
        pass


class DiffusionRunner(Runner):
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
        loss = self.compute_loss(x, original_x)  # (B, C, H, W)
        loss_mean_of_sample = loss.mean(dim=(1, 2, 3))
        loss = loss_mean_of_sample.sum()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def train(self, train_loader, n_iter):
        loss = 0
        for batch in tqdm.tqdm(train_loader):
            image_tensor, label = batch
            loss += self.train_step(
                image_tensor.to('cuda:0'))
        print(f'After {n_iter + 1: 5} epoch, average loss: '
              f'{loss / len(train_loader.dataset):.3f}')
        
    def noise(self, x, noise_level:torch.Tensor) -> torch.Tensor:
        """x has shape (B, C, H, W)"""
        assert x.size(0) == noise_level.size(0)
        noise = torch.randn_like(x)
        noise_level = noise_level.reshape(x.size(0), 1, 1, 1)
        return torch.sqrt(noise_level) * noise + torch.sqrt(1 - noise_level) * x
            
    def evaluate_step(self, x, noise_level):
        """x has shape of (B, C, H, W)"""
        self.model.eval()
        with torch.no_grad():
            noise_level = torch.ones(x.size(0), device=x.device) * noise_level
            original_x = x
            noised_x = self.noise(x, noise_level)
            x = self.model(noised_x, noise_level)
            loss = self.compute_loss(x, original_x)  # (B, C, H, W)
            loss_mean_of_sample = loss.mean(dim=(1, 2, 3))
            loss = loss_mean_of_sample.sum()
        return loss

    def evaluate(self, test_loader, n_iter):
        noise_level_to_test = (0.1, 0.3, 0.5, 0.7, 0.9)
        losss = torch.zeros(len(noise_level_to_test))
        for batch in test_loader:
            image_tensor, label = batch
            for i, noise_level in enumerate(noise_level_to_test):
                losss[i] += self.evaluate_step(
                    image_tensor.to('cuda:0'), noise_level).cpu()
        losss = losss / len(test_loader.dataset)
        score = losss.mean().item()
        self.checkpoint_manager.store_checkpoint(self, n_iter, score)
        print(f'After {n_iter + 1: 5} epoch, average validation loss: ' +\
            ' '.join([f'{loss:.3f}' for loss in losss]) +\
            f', score: {score:.3f}')
    
    def compute_loss(self, x, gt):
        return self.loss(x, gt)
    

class ClassifierRunner(Runner):
    
    def __init__(self, model) -> None:
        self.model = model
        self.checkpoint_manager = utils.CheckPointManager('checkpoint5')
        # Init a adam optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # Init a MSE loss function
        self.loss = nn.CrossEntropyLoss(reduction='sum')
    
    def train_step(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()
        y_predict = self.model(x)
        loss = self.compute_loss(y_predict, y)  # ()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def train(self, dataloader, n_iter):
        loss = 0
        for batch in tqdm.tqdm(dataloader):
            image_tensor, label = batch
            loss += self.train_step(
                image_tensor.to('cuda:0'), label.to('cuda:0'))
        print(f'After {n_iter + 1: 5} epoch, average loss: '
              f'{loss / len(dataloader.dataset):.3f}')
    
    def evaluate_step(self, x, y):
        """x has shape of (B, C, H, W)"""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)  # (B, 10)
            loss = self.compute_loss(logits, y)  # ()
            y_predict = logits.argmax(dim=1)  # (B,)
            n_correct = (y_predict == y).sum() # ()
        return {'loss': loss, 'n_correct': n_correct}
    
    def evaluate(self, test_loader, n_iter) -> dict:
        loss, n_correct = 0, 0
        for batch in test_loader:
            image_tensor, label = batch
            eval_result = self.evaluate_step(
                image_tensor.to('cuda:0'), label.to('cuda:0'))
            loss += eval_result['loss']
            n_correct += eval_result['n_correct']
        loss = loss / len(test_loader.dataset)  
        acc = n_correct / len(test_loader.dataset)
        score = acc * 100
        self.checkpoint_manager.store_checkpoint(self, n_iter, score)
        print(f'After {n_iter + 1: 5} epoch, '
              f'average validation loss: {loss:.3f}, acc: {score:.3f}')

    def compute_loss(self, x, gt):
        return self.loss(x, gt)
