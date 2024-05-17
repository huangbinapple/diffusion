from typing import Optional


import data
import module
import model

    
def main():
    model_ = module.UNet(noise_dim=32).to('cuda:0')
    runner = model.DiffusionRunner(model_)
    # model_ = module.CNN().to('cuda:0')
    # runner = model.ClassifierRunner(model_)
    print(model_)
    n_params =\
        sum(p.numel() for p in model_.parameters() if p.requires_grad)
        
    print(f'{n_params = }')
    
    train_loader, test_loader =\
        data.get_loader(train=True), data.get_loader(train=False)
    eval_interval = 10
    for n_iter in range(1000):
        runner.train(train_loader, n_iter)
        if n_iter % eval_interval == 0:
            print("Evaluating...")
            runner.evaluate(test_loader, n_iter)


if __name__ == '__main__':
    main()