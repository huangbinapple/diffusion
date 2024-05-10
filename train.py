from typing import Optional
import tqdm

import data
import module
import model

    
def main():
    unet = module.UNet(noise_dim=32).to('cuda:0')
    print(unet)
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
            evaluate_result = runner.evaluate(test_loader, n_iter)
            score = evaluate_result['score']
            costs = evaluate_result['costs']
            print(f'After {n_iter + 1: 5} epoch, average validation cost: ' +\
                ' '.join([f'{cost:.3f}' for cost in costs]) +\
                f', score: {score:.3f}')


if __name__ == '__main__':
    main()