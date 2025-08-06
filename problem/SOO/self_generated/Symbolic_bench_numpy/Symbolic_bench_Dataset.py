# import pickle
import dill as pickle
from torch.utils.data import Dataset
from .basic_problem import GP_problem

from problem import bbob
import numpy as np
train_idx = np.array([244,38,235,192,52,182,249,89,133,243,35,84,138,17,8,95,179,\
    40,165,27,229,39,198,77,211,21,107,2,105,5,230,79,22,42,82,102,177,146,108,148,110,\
        20,44,224,117,145,24,170,233,94,75,139,149,135,18,150,127,11,96,31,197,32,200,109,\
            56,61,72,41,132,120,160,184,51,49,19,196,162,136,143,48,57,125,13,214,210,180,116,\
                85,158,193,225,80,168,15,30,6,185,153,62,111,164,55,201,123,119,155,147,231,43,86,\
                    114,3,129,251,232,209,199,137,186,163,69,45,115,10,247,46,50,87,245,64,9,252,12,\
                        100,131,36,54,83,140,93,167,1,91,74,98,194,142,144,227,78,14,59,228,213,246,88,\
                            215,218,71,29],dtype=np.int64) 

test_idx = np.array([0,4,7,23,25,26,28,33,34,37,53,58,60,66,67,\
    68,70,73,76,81,90,92,97,99,101,103,106,112,113,118,121,122,124,126,\
        128,130,134,152,159,161,166,169,176,178,181,183,195,212,216,226,240,242,248,250,63],dtype=np.int64)

invalid_list = [142, 152, 158, 175 ,176, 188 ,191 ,192 ,203 ,205 ,207 ,208 ,218, 222 ,223, 224, 235 ,238 ,239, 240,  242,  254 ,255 ,256,172,173,174,220,221,237,48,105,155,157,190,204,206,17,66,189,209]            

class Symbolic_bench_Dataset(Dataset):
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
                     get_all = False):
        # get problem instances
        if instance_seed > 0:
            rng = np.random.default_rng(instance_seed)
        train_set = []
        test_set = []
        all_set = []
        with open('./problem/datafiles/SOO/self_generated/Symbolic_bench_numpy/256_programs.pickle','rb')as f:
            all_functions = pickle.load(f)
            f.close()
            
        # 需要读取eval_optimum
        eval_optimum = np.load('./optimum_dir/final_optimum.npy')
        for program in all_functions:
            all_set.append(GP_problem(program.execute,program.problemID,lb=-upperbound,ub=upperbound,dim = program.best_dim \
                ,eval_optimum=eval_optimum[program.problemID - 1]))
        
        if get_all:
            return Symbolic_bench_Dataset(all_set,train_batch_size),Symbolic_bench_Dataset(all_set,train_batch_size)
        
        nums = len(all_set)
        
        train_set = [all_set[i] for i in train_idx]
        test_set = [all_set[i] for i in test_idx]
        
        return Symbolic_bench_Dataset(train_set, train_batch_size), Symbolic_bench_Dataset(test_set, test_batch_size)
        
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
        
    