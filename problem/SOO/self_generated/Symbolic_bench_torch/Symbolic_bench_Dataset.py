# import pickle
import dill as pickle
from torch.utils.data import Dataset
from .basic_problem import GP_problem
import numpy as np
import torch


class Symbolic_bench_Dataset_torch(Dataset):
    def __init__(self,
                 data,
                 batch_size=1):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.N = len(self.data)
        self.ptr = [i for i in range(0, self.N, batch_size)]
        self.index = np.arange(self.N)
            
    @staticmethod
    def get_datasets(upperbound = 5,
                     train_batch_size=1,
                     test_batch_size=1,
                     instance_seed=3849,
                     train_test_split = 0.8,
                     get_all = False):
        # get problem instances
        if instance_seed > 0:
            rng = np.random.default_rng(instance_seed)
            
        train_set = []
        test_set = []
        all_set = []
        with open('./problem/datafiles/SOO/self_generated/Symbolic_bench_torch/256_programs.pickle','rb')as f:
            all_functions = pickle.load(f)
            f.close()
            
            
        # 需要读取eval_optimum
        eval_optimum = np.load('./optimum_dir/new_optimum.npy')
        for program in all_functions:
            # 172,173,174,220,221,237,48,105
            # 105 174 221 172 48 
            # 173,220,237
            if program.problemID not in [105 ,174, 221, 172 ,48 ]:
                all_set.append(GP_problem(program.execute,program.problemID,lb=-upperbound,ub=upperbound,dim = program.best_dim ,eval_optimum=eval_optimum))
        
        if get_all:
            return Symbolic_bench_Dataset_torch(all_set,train_batch_size)
        
        nums = len(all_set)
        train_len = int(nums*train_test_split)
        # train_test_split
        rand_idx = rng.permutation(nums)

        train_set = [all_set[i] for i in rand_idx[:train_len]]
        test_set = [all_set[i] for i in rand_idx[train_len:]]
    
        return Symbolic_bench_Dataset_torch(train_set, train_batch_size), Symbolic_bench_Dataset_torch(test_set, test_batch_size)
    
    
        
    def __getitem__(self, item):
        if self.batch_size < 2:
            return self.data[self.index[item]]
        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res

    def __len__(self):
        return self.N

    def __add__(self, other: 'Symbolic_bench_Dataset'):
        return Symbolic_bench_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)
        
    