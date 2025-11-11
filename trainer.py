"""
This file is used to train the agent.(for the kind of optimizer that is learnable)
"""
import pickle
from tqdm import tqdm
from environment.basic_environment import PBO_Env
from VectorEnv import *
from logger import Logger
import copy
from utils import *
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from basic_agent.utils import save_class
from tensorboardX import SummaryWriter
from optimizer import (
    DE_DDQN_Optimizer,
    LDE_Optimizer,
    GLEET_Optimizer,
    SYMBOL_Optimizer,
    Random_search
)

from agents import (
    GLEET_Agent,
    DE_DDQN_Agent,
    LDE_Agent,
    SYMBOL_Agent,

)

matplotlib.use('Agg')


class Trainer(object):
    def __init__(self, config):
        self.config = config
        if config.resume_dir is None:
            self.agent = eval(config.train_agent)(config)
        else:
            file_path = config.resume_dir + config.train_agent + '.pkl'
            with open(file_path, 'rb') as f:
                self.agent = pickle.load(f)
            self.agent.update_setting(config)
        self.optimizer = eval(config.train_optimizer)(config)
        
        sym_config = copy.deepcopy(config)
        sym_config.problem = 'Diverse-BBO'
        train_set, self.test_set = construct_problem_set(sym_config)
        rng = np.random.default_rng(42)
        if config.problem == 'Diverse-BBO':
            self.train_set, self.test_set = construct_problem_set(config)
            
            
            
    def save_log(self, epochs, steps, cost, returns, normalizer):
        log_dir = self.config.log_dir + f'/train/{self.agent.__class__.__name__}/{self.config.run_time}/log/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return_save = np.stack((steps, returns),  0)
        np.save(log_dir+'return', return_save)
        for problem in self.train_set:
            name = problem.__str__()
            if len(cost[name]) == 0:
                continue
            while len(cost[name]) < len(epochs):
                cost[name].append(cost[name][-1])
                normalizer[name].append(normalizer[name][-1])
            cost_save = np.stack((epochs, cost[name], normalizer[name]),  0)
            np.save(log_dir+name+'_cost', cost_save)
            
    def draw_cost(self, Name=None, normalize=False):
        log_dir = self.config.log_dir + f'/train/{self.agent.__class__.__name__}/{self.config.run_time}/'
        if not os.path.exists(log_dir + 'pic/'):
            os.makedirs(log_dir + 'pic/')
        for problem in self.train_set:
            if Name is None:
                name = problem.__str__()
            elif (isinstance(Name, str) and problem.__str__() != Name) or (isinstance(Name, list) and problem.__str__() not in Name):
                continue
            else:
                name = Name
            plt.figure()
            plt.title(name + '_cost')
            values = np.load(log_dir + 'log/' + name+'_cost.npy')
            x, y, n = values
            if normalize:
                y /= n
            plt.plot(x, y)
            plt.savefig(log_dir+f'pic/{name}_cost.png')
            plt.close()
    
    def draw_average_cost(self, normalize=True):
        log_dir = self.config.log_dir + f'/train/{self.agent.__class__.__name__}/{self.config.run_time}/'
        if not os.path.exists(log_dir + 'pic/'):
            os.makedirs(log_dir + 'pic/')
        X = []
        Y = []
        for problem in self.train_set:
            name = problem.__str__()
            values = np.load(log_dir + 'log/' + name+'_cost.npy')
            x, y, n = values
            if normalize:
                y /= n
            X.append(x)
            Y.append(y)
        X = np.mean(X, 0)
        Y = np.mean(Y, 0)
        plt.figure()
        plt.title('all problem cost')
        plt.plot(X, Y)
        plt.savefig(log_dir+f'pic/all_problem_cost.png')
        plt.close()

    def draw_return(self):
        log_dir = self.config.log_dir + f'/train/{self.agent.__class__.__name__}/{self.config.run_time}/'
        if not os.path.exists(log_dir + 'pic/'):
            os.makedirs(log_dir + 'pic/')
        plt.figure()
        plt.title('return')
        values = np.load(log_dir + 'log/return.npy')
        plt.plot(values[0], values[1])
        plt.savefig(log_dir+f'pic/return.png')
        plt.close()


    def train_new(self):
        print(f'start training: {self.config.run_time}')
        is_end = False
        # todo tensorboard
        tb_logger = None
        if not self.config.no_tb:
            tb_logger = SummaryWriter(os.path.join('output/tensorboard', self.config.run_time))
            tb_logger.add_scalar("epoch-step", 0, 0)

        epoch = 0
        cost_record = {}
        normalizer_record = {}
        return_record = []
        learn_steps = []
        epoch_steps = []

        for problem in self.train_set.data:
            cost_record[problem.__str__()] = []
            normalizer_record[problem.__str__] = []

        bs = self.config.train_batch_size
        if self.config.train_mode == "single":
            self.train_set.batch_size = 1
        elif self.config.train_mode == "multi":
            self.train_set.batch_size = bs

        epoch_seed = self.config.epoch_seed
        id_seed = self.config.id_seed
        seed = self.config.seed

        if self.config.problem == 'Diverse-BBO':
            episode_optimum = np.load('./optimum_dir/final_optimum.npy')

        
        while not is_end:
            learn_step = 0
            self.train_set.shuffle()
            with tqdm(range(self.train_set.N), desc = f'Training {self.agent.__class__.__name__} Epoch {epoch}') as pbar:
                for problem_id, problem in enumerate(self.train_set):
                    # set seed
                    seed_list = (epoch * epoch_seed + id_seed * (np.arange(bs) + bs * problem_id) + seed).tolist()

                    if self.config.train_mode == "single":
                        env_list = [PBO_Env(copy.deepcopy(problem), copy.deepcopy(self.optimizer)) for _ in range(bs)] # bs
                    elif self.config.train_mode == "multi":
                        env_list = [PBO_Env(p, copy.deepcopy(self.optimizer)) for p in problem] # bs

                    # todo config add para
                    exceed_max_ls, train_meta_data = self.agent.train_episode(envs = env_list,
                                                                              seeds = seed_list,
                                                                            #   tb_logger = tb_logger,
                                                                              para_mode = "dummy",
                                                                              asynchronous = None,
                                                                              num_cpus = 1,
                                                                              num_gpus = 0,
                                                                              )
                    # exceed_max_ls, pbar_info_train = self.agent.train_episode(env)  # pbar_info -> dict
                    postfix_str = (
                        f"loss={train_meta_data['loss']:.2e}, "
                        f"learn_steps={train_meta_data['learn_steps']}, "
                        f"return={[f'{x:.2e}' for x in train_meta_data['return']]}"
                    )

                    pbar.set_postfix_str(postfix_str)
                    pbar.update(self.train_set.batch_size)
                    learn_step = train_meta_data['learn_steps']
                    # for id, p in enumerate(problem):
                    #     name = p.__str__()
                    #     cost_record[name].append(train_meta_data['gbest'][id])
                    #     normalizer_record[name].append(train_meta_data['normalizer'][id])
                    #     return_record.append(np.mean(train_meta_data['return']))
                    # learn_steps.append(learn_step)
                    if self.config.problem in ['Diverse-BBO']: 
                        for env in env_list:
                            if  env.problem.episode_optimum < env.problem.eval_optimum :
                                episode_optimum[env.problem.problemID - 1] = env.problem.episode_optimum
                    
                    if self.config.end_mode == "step" and exceed_max_ls:
                        is_end = True
                        break
                self.agent.train_epoch()
            epoch_steps.append(learn_step)
            epoch += 1

            if not self.config.no_tb:
                tb_logger.add_scalar("epoch-step", learn_step, epoch)

            # todo save
            # save_interval = 5
            # checkpoint0 0
            # checkpoint1 5
            if epoch >= (self.config.save_interval * self.agent.cur_checkpoint) and self.config.end_mode == "epoch":
                save_class(self.config.agent_save_dir, 'checkpoint' + str(self.agent.cur_checkpoint), self.agent)
                # 记录 checkpoint 和 total_step
                with open(self.config.agent_save_dir + "/checkpoint_log.txt", "a") as f:
                    f.write(f"Checkpoint {self.agent.cur_checkpoint}: {learn_step}\n")

                self.agent.cur_checkpoint += 1
            if self.config.end_mode == "epoch" and epoch >= self.config.max_epoch:
                is_end = True

            
        