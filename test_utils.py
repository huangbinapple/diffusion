import unittest
import os
import torch
from utils import CheckPointManager

# Fake Runner
class Runner:
    def __init__(self):
        self.model = torch.nn.Linear(10, 10)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)


class TestCheckPointManager(unittest.TestCase):
    
    def setUp(self):
        self.workspace = 'test'
        self.manager = CheckPointManager(self.workspace, max_store=5)
        self.runner = Runner()  # Assume a class Runner is defined elsewhere

    def test_get_file_name(self):
        # Test with no epoch and no score
        file_name = CheckPointManager.get_file_name()
        self.assertEqual(file_name, 'checkpoint.pt')

        # Test with epoch and no score
        file_name = CheckPointManager.get_file_name(nepoch=10)
        self.assertEqual(file_name, 'checkpoint_epoch_10.pt')

        # Test with no epoch and score
        file_name = CheckPointManager.get_file_name(score=10.5)
        self.assertEqual(file_name, 'checkpoint_score_10.50.pt')

        # Test with epoch and score
        file_name = CheckPointManager.get_file_name(nepoch=10, score=10.5)
        self.assertEqual(file_name, 'checkpoint_epoch_10_score_10.50.pt')
        
    def test_store_checkpoint(self):
        # Test when the queue is not full
        self.manager.store_checkpoint(self.runner, nepoch=1, score=0.5)
        self.assertTrue(os.path.exists(self.manager.get_path('checkpoint_epoch_1_score_0.50.pt')))
        self.assertTrue(os.path.exists(self.manager.get_path('latest.pt')))
        self.assertTrue(os.path.exists(self.manager.get_path('best.pt')))

        # Test when the queue is full
        for i in range(2, 6):
            self.manager.store_checkpoint(self.runner, nepoch=i, score=0.5)
        self.assertTrue(os.path.exists(self.manager.get_path('checkpoint_epoch_1_score_0.50.pt')))
        self.assertTrue(os.path.exists(self.manager.get_path('latest.pt')))
        self.assertTrue(os.path.exists(self.manager.get_path('best.pt')))

        # Test when the queue is full and delete the oldest one
        self.manager.store_checkpoint(self.runner, nepoch=6, score=0.3)
        self.assertFalse(os.path.exists(self.manager.get_path('checkpoint_epoch_1_score_0.50.pt')))
        self.assertTrue(os.path.exists(self.manager.get_path('checkpoint_epoch_6_score_0.30.pt')))
        self.assertTrue(os.path.exists(self.manager.get_path('latest.pt')))
        self.assertTrue(os.path.exists(self.manager.get_path('best.pt')))

    def test_load_checkpoint(self):
        # Test when the file exists
        self.manager.store_checkpoint(self.runner, nepoch=1, score=0.5)
        new_runner = Runner()
        self.manager.load_checkpoint(new_runner, 'checkpoint_epoch_1_score_0.50.pt')
        for key in new_runner.model.state_dict().keys():
            torch.testing.assert_close(
                self.runner.model.state_dict()[key],
                new_runner.model.state_dict()[key])

        with self.assertRaises(FileNotFoundError):
            self.manager.load_checkpoint(self.runner, 'nonexistent.pt')


if __name__ == '__main__':
    unittest.main()