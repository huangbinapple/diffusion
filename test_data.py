import unittest
import torch
from data import get_loader


class TestGetLoader(unittest.TestCase):

    def test_get_loader_train(self):
        loader = get_loader(train=True)
        print(f'{type(loader) = }')
        self.assertTrue(isinstance(loader, torch.utils.data.DataLoader))

    def test_get_loader_test(self):
        loader = get_loader(train=False)
        print(f'{type(loader) = }')
        self.assertTrue(isinstance(loader, torch.utils.data.DataLoader))

if __name__ == '__main__':
    unittest.main()