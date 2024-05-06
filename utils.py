import os
import queue
import torch


class CheckPointManager:
    
    def __init__(self, workspace, max_store=5, high_is_better=True) -> None:
        self.workspace = workspace
        self.max_store = max_store
        self.high_is_better = high_is_better
        self.in_store = queue.PriorityQueue(maxsize=max_store)
        # If workspace is not exsit, create one.
        if not os.path.exists(self.workspace):
            os.makedirs(self.workspace)
            
    def store_managere_state(self, file_name):
        contents = {}
        contents['in_store'] = self.in_store.queue
        contents['max_store'] = self.max_store
        contents['high_is_better'] = self.high_is_better
        torch.save(contents, self.get_path(file_name))
        
    @staticmethod
    def get_file_name(nepoch=None, score=None):
        suffix = ''
        if nepoch is not None:
            suffix += f'_epoch_{nepoch}'
        if score is not None:
            suffix += f'_score_{score:.3f}'
        return f'checkpoint{suffix}.pt'
    
    def get_path(self, file_name):
        return os.path.join(self.workspace, file_name)
        
    def store_checkpoint(self, runner, nepoch=None, score=None):
        if self.in_store.full():
            _, file_to_delete = self.in_store.get()
            os.remove(self.get_path(file_to_delete))
        file_name = self.get_file_name(nepoch, score)
        content = {
            'epoch': nepoch,
            'model_state_dict': runner.model.state_dict(),
            'optimizer_state_dict': runner.optimizer.state_dict(),
            'score': score,
        }
        save_path = os.path.join(self.workspace, file_name)
        torch.save(content, save_path)
        score = 0 if score is None else score
        if not self.high_is_better:
            score = -score
        self.in_store.put((score, file_name))
        self.store_managere_state('manager.pt')
        # Create a softlink for the last stored filename
        self.delete_file('latest.pt')
        os.symlink(file_name, self.get_path('latest.pt'))
        # Create a softlink for the best stored filename
        file_best = self.get_best_file()
        self.delete_file('best.pt')
        os.symlink(file_best, self.get_path('best.pt'))
        
    def get_best_file(self):
        best_score = None
        for score, file in self.in_store.queue:
            if best_score is None or score > best_score:
                file_best = file
                best_score = score
        return file_best
        
    def load_checkpoint(self, runner, file_name):
        if not os.path.exists(self.get_path(file_name)):
            raise FileNotFoundError(self.get_path(file_name))
        contents = torch.load(self.get_path(file_name))
        runner.model.load_state_dict(contents['model_state_dict'])
        runner.optimizer.load_state_dict(contents['optimizer_state_dict'])
        
    def delete_file(self, file_name):
        file_path = self.get_path(file_name)
        if os.path.lexists(file_path):
            os.remove(file_path)