import os
import pickle
from problem import bbob, bbob_torch
from problem.SOO.self_generated.Symbolic_bench_numpy.Symbolic_bench_Dataset import Symbolic_bench_Dataset
from problem.SOO.self_generated.Symbolic_bench_torch.Symbolic_bench_Dataset import Symbolic_bench_Dataset_torch
from dataset.protein_docking import *
from dataset.UAV.uav_dataset import UAV_Dataset
from dataset.hpo_b.hpob_dataset import *
import pandas as pd
from dataset.MABBOB import ManyAffine
from dataset.GP_baseline import GP_baseline
import time
import torch
import numpy as np
def set_seed(seed=None):
    if seed is None:
        seed=int(time.time())
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
def construct_problem_set(config):
    problem = config.problem
    # if problem in ['bbob']:
    #     return bbob.BBOB_Dataset.get_datasets(suit=config.problem,
    #                                           dim=config.dim,
    #                                           upperbound=config.upperbound,
    #                                           train_batch_size=config.train_batch_size,
    #                                           test_batch_size=config.test_batch_size,
    #                                           difficulty=config.difficulty,
    #                                           get_all=True)

    # elif problem in ['protein', 'protein-torch']:
    #     return Protein_Docking_Dataset.get_datasets(version=problem,
    #                                                                 train_batch_size=config.train_batch_size,
    #                                                                 test_batch_size=config.test_batch_size,
    #                                                                 difficulty=config.difficulty,
    #                                                                 all_instances=config.all_instances,)
    if problem == 'Diverse-BBO':
        return Symbolic_bench_Dataset.get_datasets(upperbound=config.upperbound,    
                                                    train_batch_size=config.train_batch_size,
                                                        test_batch_size=config.test_batch_size,
                                                        )
    # elif problem in ['UAV']:
    #     return UAV_Dataset.get_datasets(difficulty='all')
    # elif problem in ['HPOB']:
    #     return HPOB_Dataset.get_datasets(get_all=config.all_instances)
    else:
        raise ValueError(problem + ' is not defined!')
    
    

    

def get_baseline_function(upperbound,func_dir,dim,random_state):
    train_func = []
    with open(os.path.join(func_dir,f'all_256_programs.pickle'),'rb') as f:
        programs = pickle.load(f)  
        f.close()
    for program in programs:
        train_func.append(GP_baseline(program.execute,program.problemID,lb=-upperbound,ub=upperbound,dim = dim , random_state=random_state ))
    return train_func
