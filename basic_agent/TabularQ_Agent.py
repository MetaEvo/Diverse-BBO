import math
from typing import Optional, Union, Literal, List

from VectorEnv.great_para_env import ParallelEnv
from basic_agent.basic_agent import Basic_Agent
from basic_agent.utils import *


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for idx, group in enumerate(param_groups)
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


class TabularQ_Agent(Basic_Agent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # define parameters
        self.gamma = self.config.gamma
        self.n_act = self.config.n_act
        self.n_state = self.config.n_state
        self.epsilon = self.config.epsilon
        self.lr_model = self.config.lr_model

        self.q_table = torch.zeros(self.n_state, self.n_act)
        
        # init learning time
        self.learning_time = 0
        self.cur_checkpoint = 0

        # save init agent
        save_class(self.config.agent_save_dir,'checkpoint'+str(self.cur_checkpoint),self)
        self.cur_checkpoint += 1

    def update_setting(self, config):
        self.config.max_learning_step = config.max_learning_step
        self.config.agent_save_dir = config.agent_save_dir
        self.learning_time = 0
        save_class(self.config.agent_save_dir, 'checkpoint0', self)
        self.config.save_interval = config.save_interval
        self.cur_checkpoint = 1
        
    # def get_action(self, state, epsilon_greedy=False):
    #     Q_list = self.q_table(state)
    #     if epsilon_greedy and np.random.rand() < self.epsilon:
    #         action = np.random.randint(low=0, high=self.n_act, size=len(state))
    #     else:
    #         action = torch.argmax(Q_list, -1).numpy()
    #     return action
    
    
    def get_action(self, state, epsilon_greedy=False):
        Q_list = torch.stack([self.q_table[st] for st in state ])
        if epsilon_greedy and np.random.rand() < self.epsilon:
            action = np.random.randint(low=0, high=self.n_act, size=len(state))
        else:
            action = torch.argmax(Q_list, -1).numpy()
        return action

    def train_episode(self, 
                      envs,
                      seeds: Optional[Union[int, List[int], np.ndarray]],
                      para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc']='dummy',
                      asynchronous: Literal[None, 'idle', 'restart', 'continue']=None,
                      num_cpus: Optional[Union[int, None]]=1,
                      num_gpus: int=0,
                      required_info={}):
        if self.device != 'cpu':
            num_gpus = max(num_gpus, 1)
        env = ParallelEnv(envs, para_mode, asynchronous, num_cpus, num_gpus)
        env.seed(seeds)
        # params for training
        gamma = self.gamma
        
        state = env.reset()
        state = torch.FloatTensor(state)
        
        _R = torch.zeros(len(env))
        # sample trajectory
        while not env.all_done():
            action = self.get_action(state=state, epsilon_greedy=True)
                        
            # state transient
            next_state, reward, is_end, info = env.step(action)
            _R += reward
            
            # error = reward + gamma * self.q_table[next_state].max() - self.q_table[state][action]
            
            error = [reward[i] + gamma * self.q_table[next_state[i]].max() - self.q_table[state[i]][action[i]]\
                for i in range(len(state)) ]
            
            for i in range(len(state)):
                self.q_table[state[i]][action[i]] += self.lr_model * error[i]
            
            # store info
            state = torch.FloatTensor(next_state)
            
            self.learning_time += 1
            if self.learning_time >= (self.config.save_interval * self.cur_checkpoint):
                save_class(self.config.agent_save_dir, 'checkpoint'+str(self.cur_checkpoint), self)
                self.cur_checkpoint += 1

            if self.learning_time >= self.config.max_learning_step:
                return_info = {'return': _R, 'learn_steps': self.learning_time, }
                for key in required_info.keys():
                    return_info[key] = env.get_env_attr(required_info[key])
                env.close()
                return self.learning_time >= self.config.max_learning_step, return_info
        
            
        is_train_ended = self.learning_time >= self.config.max_learning_step
        return_info = {'return': _R, 'learn_steps': self.learning_time, }
        env_cost = env.get_env_attr('cost')
        return_info['normalizer'] = env_cost[0]
        return_info['gbest'] = env_cost[-1]
        for key in required_info.keys():
            return_info[key] = env.get_env_attr(required_info[key])
        env.close()
        
        return is_train_ended, return_info
    
    def rollout_episode(self, 
                        env,
                        seed=None,
                      required_info={}):
        with torch.no_grad():
            if seed is not None:
                env.seed(seed)
            is_done = False
            state = env.reset()
            R = 0
            while not is_done:
                action = self.get_action([state])[0]
                state, reward, is_done = env.step(action)
                R += reward
            env_cost = env.get_env_attr('cost')
            env_fes = env.get_env_attr('fes')
            results = {'cost': env_cost, 'fes': env_fes, 'return': R}
            for key in required_info.keys():
                results[key] = getattr(env, required_info[key])
            return results
    
    def rollout_batch_episode(self, 
                              envs, 
                              seeds=None,
                              para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc']='dummy',
                              asynchronous: Literal[None, 'idle', 'restart', 'continue']=None,
                              num_cpus: Optional[Union[int, None]]=1,
                              num_gpus: int=0,
                      required_info={}):
        if self.device != 'cpu':
            num_gpus = max(num_gpus, 1)
        env = ParallelEnv(envs, para_mode, asynchronous, num_cpus, num_gpus)

        env.seed(seeds)
        state = env.reset()
        
        R = torch.zeros(len(env))
        # sample trajectory
        while not env.all_done():
            action = self.get_action(torch.FloatTensor(state))
            # state transient
            state, rewards, is_end, info = env.step(action)
            # print('step:{},max_reward:{}'.format(t,torch.max(rewards)))
            R += torch.FloatTensor(rewards).squeeze()
            state = torch.FloatTensor(state)
        _Rs = R.detach().numpy().tolist()
        env_cost = env.get_env_attr('cost')
        env_fes = env.get_env_attr('fes')
        results = {'cost': env_cost, 'fes': env_fes, 'return': _Rs}
        for key in required_info.keys():
            results[key] = env.get_env_attr(required_info[key])
        return results

# todo add metric
    def log_to_tb_train(self, tb_logger, mini_step,
                        extra_info = {}):
        # Iterate over the extra_info dictionary and log data to tb_logger
        # extra_info: Dict[str, Dict[str, Union[List[str], List[Union[int, float]]]]] = {
        #     "loss": {"name": [], "data": [0.5]},  # No "name", logs under "loss"
        #     "accuracy": {"name": ["top1", "top5"], "data": [85.2, 92.5]},  # Logs as "accuracy/top1" and "accuracy/top5"
        #     "learning_rate": {"name": ["adam", "sgd"], "data": [0.001, 0.01]}  # Logs as "learning_rate/adam" and "learning_rate/sgd"
        # }
        #
        # learning rate
        # for id, network_name in enumerate(self.network):
        #     tb_logger.add_scalar(f'learnrate/{network_name}', self.optimizer.param_groups[id]['lr'], mini_step)
        #
        # # grad and clipped grad
        # grad_norms, grad_norms_clipped = grad_norms
        # for id, network_name in enumerate(self.network):
        #     tb_logger.add_scalar(f'grad/{network_name}', grad_norms[id], mini_step)
        #     tb_logger.add_scalar(f'grad_clipped/{network_name}', grad_norms_clipped[id], mini_step)
        #
        #
        # # loss
        # tb_logger.add_scalar('loss/actor_loss', reinforce_loss.item(), mini_step)
        # tb_logger.add_scalar('loss/critic_loss', baseline_loss.item(), mini_step)
        # tb_logger.add_scalar('loss/total_loss', (reinforce_loss + baseline_loss).item(), mini_step)
        #
        # # train metric
        # avg_reward = torch.stack(memory_reward).mean().item()
        # max_reward = torch.stack(memory_reward).max().item()
        #
        # tb_logger.add_scalar('train/episode_avg_return', Return.mean().item(), mini_step)
        # tb_logger.add_scalar('train/target_avg_return_changed', Reward.mean().item(), mini_step)
        # tb_logger.add_scalar('train/critic_avg_output', critic_output.mean().item(), mini_step)
        # tb_logger.add_scalar('train/avg_entropy', entropy.mean().item(), mini_step)
        # tb_logger.add_scalar('train/-avg_logprobs', -logprobs.mean().item(), mini_step)
        # tb_logger.add_scalar('train/approx_kl', approx_kl_divergence.item(), mini_step)
        # tb_logger.add_scalar('train/avg_reward', avg_reward, mini_step)
        # tb_logger.add_scalar('train/max_reward', max_reward, mini_step)

        # extra info
        for key, value in extra_info.items():
            if not value['name']:
                tb_logger.add_scalar(f'{key}', value['data'][0], mini_step)
            else:
                name_list = value['name']
                data_list = value['data']
                for name, data in zip(name_list, data_list):
                    tb_logger.add_scalar(f'{key}/{name}', data, mini_step)
