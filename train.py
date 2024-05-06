from typing import Optional
import tqdm

import torch

import data
import module
import model
import utils

    
def main():
    unet = module.UNet(noise_dim=32).to('cuda:0')
    print(unet)
    checkpoint_manager = utils.CheckPointManager(
        'checkpoint3', high_is_better=False)
    runner = model.DiffusionRunner(unet)
    n_params =\
        sum(p.numel() for p in unet.parameters() if p.requires_grad)
        
    print(f'{n_params = }')
    
    train_loader, test_loader =\
        data.get_loader(train=True), data.get_loader(train=False)
    eval_interval = 10
    for n_iter in range(1000):
        cost = 0
        for batch in tqdm.tqdm(train_loader):
            image_tensor, label = batch
            cost += runner.train_step(image_tensor.to('cuda:0'))
        print(f'After {n_iter + 1: 5} epoch, average cost: '
              f'{cost / len(train_loader):.3f}')
        
        if n_iter % eval_interval == 0:
            print("Evaluating...")
            noise_level_to_test = (0.1, 0.3, 0.5, 0.7, 0.9)
            costs = torch.zeros(len(noise_level_to_test))
            for batch in tqdm.tqdm(test_loader):
                image_tensor, label = batch
                for i, noise_level in enumerate(noise_level_to_test):
                    costs[i] += runner.evaluate(
                        image_tensor.to('cuda:0'), noise_level).cpu()
            costs = costs / len(test_loader)
            score = costs.mean().item()
            checkpoint_manager.store_checkpoint(runner, n_iter, score)
            print(f'After {n_iter + 1: 5} epoch, average validation cost: ' +\
                ' '.join([f'{cost:.3f}' for cost in costs]) +\
                f', score: {score:.3f}')


if __name__ == '__main__':
    main()