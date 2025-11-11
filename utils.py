from problem.SOO.self_generated.Symbolic_bench_numpy.Symbolic_bench_Dataset import Diverse_BBO_Dataset
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
    if problem == 'Diverse-BBO':
        return Diverse_BBO_Dataset.get_datasets(upperbound=config.upperbound,    
                                                    train_batch_size=config.train_batch_size,
                                                        test_batch_size=config.test_batch_size,
                                                        )
    else:
        raise ValueError(problem + ' is not defined!')
    

