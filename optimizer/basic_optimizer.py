"""
This is the basic optimizer class. All traditional optimizers should inherit from this class.
Your own traditional should have the following functions:
    1. __init__(self, config) : to initialize the optimizer
    2. run_episode(self, problem) : to run the optimizer for an episode
"""
from problem.basic_problem import Basic_Problem
import numpy as np
import torch

class Basic_Optimizer:
    def __init__(self, config):
        self.__config = config

    def seed(self, seed = None):
        self.rng = np.random
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            self.rng = np.random.RandomState(seed)

    def run_episode(self, problem: Basic_Problem):
        raise NotImplementedError
