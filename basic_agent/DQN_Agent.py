import copy
import math
from typing import Optional, Union, Literal, List
import torch.nn.functional as F

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


class DQN_Agent(Basic_Agent):
    def __init__(self, config, network: dict, learning_rates: Optional):
        super().__init__(config)
        self.config = config

        # define parameters
        self.gamma = self.config.gamma
        self.n_act = self.config.n_act
        self.epsilon = self.config.epsilon
        self.max_grad_norm = self.config.max_grad_norm
        self.memory_size = self.config.memory_size
        self.batch_size = self.config.batch_size
        self.warm_up_size = self.config.warm_up_size
        self.device = self.config.device

        self.replay_buffer = ReplayBuffer(self.memory_size)
        self.set_network(network, learning_rates)
        # figure out the actor network
        # self.model = None
        # assert hasattr(self, 'model')
        #
        # # figure out the optimizer
        # assert hasattr(torch.optim, self.config.optimizer)
        # self.optimizer = eval('torch.optim.' + self.config.optimizer)(
        #     [{'params': self.model.parameters(), 'lr': self.config.lr_model}])
        # # figure out the lr schedule
        # # assert hasattr(torch.optim.lr_scheduler, self.config.lr_scheduler)
        # # self.lr_scheduler = eval('torch.optim.lr_scheduler.' + self.config.lr_scheduler)(self.optimizer, self.config.lr_decay, last_epoch=-1,)
        #
        # assert hasattr(torch.nn, self.config.criterion)
        # self.criterion = eval('torch.nn.' + self.config.criterion)()
        #
        # self.replay_buffer = ReplayBuffer(self.memory_size)
        #
        # # move to device
        # self.model.to(self.device)

        # init learning time
        self.learning_time = 0
        self.cur_checkpoint = 0

        # save init agent
        save_class(self.config.agent_save_dir,'checkpoint'+str(self.cur_checkpoint),self)
        self.cur_checkpoint += 1

    def set_network(self, networks: dict, learning_rates: Optional):
        Network_name = []
        if networks:
            for name, network in networks.items():
                Network_name.append(name)
                setattr(self, name, network)  # Assign each network in the dictionary to the class instance
        self.networks = Network_name

        assert hasattr(self, 'model')  # Ensure that 'model' is set as an attribute of the class

        if isinstance(learning_rates, (int, float)):
            learning_rates = [learning_rates] * len(networks)
        elif len(learning_rates) != len(networks):
            raise ValueError("The length of the learning rates list must match the number of networks!")

        all_params = []
        for id, network_name in enumerate(networks):
            network = getattr(self, network_name)
            all_params.append({'params': network.parameters(), 'lr': learning_rates[id]})

        assert hasattr(torch.optim, self.config.optimizer)
        self.optimizer = eval('torch.optim.' + self.config.optimizer)(all_params)

        assert hasattr(torch.nn, self.config.criterion)
        self.criterion = eval('torch.nn.' + self.config.criterion)()

        for network_name in networks:
            getattr(self, network_name).to(self.device)

    def update_setting(self, config):
        self.config.max_learning_step = config.max_learning_step
        self.config.agent_save_dir = config.agent_save_dir
        self.learning_time = 0
        save_class(self.config.agent_save_dir, 'checkpoint0', self)
        self.config.save_interval = config.save_interval
        self.cur_checkpoint = 1
        
    def get_action(self, state, epsilon_greedy=False):
        state = torch.Tensor(state).to(self.device)
        with torch.no_grad():
            Q_list = self.model(state)
        if epsilon_greedy and np.random.rand() < self.epsilon:
            action = np.random.randint(low=0, high=self.n_act, size=len(state))
        else:
            action = torch.argmax(Q_list, -1).detach().cpu().numpy()
        return action

    def train_episode(self, 
                      envs,
                      seeds: Optional[Union[int, List[int], np.ndarray]],
                      para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc']='dummy',
                      asynchronous: Literal[None, 'idle', 'restart', 'continue']=None,
                      num_cpus: Optional[Union[int, None]]=1,
                      num_gpus: int=0,
                      required_info=['normalizer', 'gbest']):
        if self.device != 'cpu':
            num_gpus = max(num_gpus, 1)
        env = ParallelEnv(envs, para_mode, asynchronous, num_cpus, num_gpus)
        env.seed(seeds)
        # params for training
        gamma = self.gamma
        
        state = env.reset()
        try:
            state = torch.FloatTensor(state)
        except:
            pass
        
        _R = torch.zeros(len(env))
        _loss = []
        # sample trajectory
        while not env.all_done():
            action = self.get_action(state=state, epsilon_greedy=True)
                        
            # state transient
            next_state, reward, is_end, info = env.step(action)
            _R += reward
            # store info
            # convert next_state into tensor
            try:
                next_state = torch.FloatTensor(next_state).to(self.device)
            except:
                pass
            for s, a, r, ns, d in zip(state, action, reward, next_state, is_end):
                self.replay_buffer.append((s, a, r, ns, d))
            try:
                state = torch.FloatTensor(next_state).to(self.device)
            except:
                state = copy.deepcopy(next_state)
            
            # begin update
            if len(self.replay_buffer) >= self.warm_up_size:
                batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = self.replay_buffer.sample(self.batch_size)
                pred_Vs = self.model(batch_obs.to(self.device))  # [batch_size, n_act]
                action_onehot = torch.nn.functional.one_hot(batch_action.to(self.device), self.n_act)  # [batch_size, n_act]
                
                predict_Q = (pred_Vs * action_onehot).sum(1)  # [batch_size]
                target_Q = batch_reward.to(self.device) + (1 - batch_done.to(self.device)) * gamma * self.model(batch_next_obs.to(self.device)).max(1)[0].detach()
                
                self.optimizer.zero_grad()
                loss = self.criterion(predict_Q, target_Q)
                loss.backward()
                grad_norms = clip_grad_norms(self.optimizer.param_groups, self.config.max_grad_norm)
                self.optimizer.step()

                _loss.append(loss.item())

                self.learning_time += 1
                if self.learning_time >= (self.config.save_interval * self.cur_checkpoint):
                    save_class(self.config.agent_save_dir, 'checkpoint'+str(self.cur_checkpoint), self)
                    self.cur_checkpoint += 1

                if self.learning_time >= self.config.max_learning_step:
                    _Rs = _R.detach().numpy().tolist()
                    return_info = {'return': _Rs, 'loss': np.mean(_loss), 'learn_steps': self.learning_time, }
                    env_cost = env.get_env_attr('cost')
                    return_info['normalizer'] = env_cost[0]
                    return_info['gbest'] = env_cost[-1]
                    for key in required_info:
                        return_info[key] = env.get_env_attr(key)
                    env.close()
                    return self.learning_time >= self.config.max_learning_step, return_info
        

        is_train_ended = self.learning_time >= self.config.max_learning_step
        _Rs = _R.detach().numpy().tolist()
        return_info = {'return': _Rs, 'loss': np.mean(_loss), 'learn_steps': self.learning_time}
        env_cost = env.get_env_attr('cost')
        return_info['normalizer'] = env_cost[0]
        return_info['gbest'] = env_cost[-1]
        for key in required_info:
            return_info[key] = env.get_env_attr(key)
            # print(f"{key} : {return_info[key]}")
        env.close()
        
        return is_train_ended, return_info
    
    def rollout_episode(self, 
                        env,
                        seed=None,
                        required_info=['normalizer', 'gbest']):
        with torch.no_grad():
            if seed is not None:
                env.seed(seed)
            is_done = False
            state = env.reset()
            R = 0
            while not is_done:
                try:
                    state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                except:
                    st = [state]
                action = self.get_action(state)[0]
                action = action.cpu().numpy().squeeze()
                state, reward, is_done = env.step(action)
                R += reward
            _Rs = R.detach().numpy().tolist()
            env_cost = env.get_env_attr('cost')
            env_fes = env.get_env_attr('fes')
            results = {'cost': env_cost, 'fes': env_fes, 'return': _Rs}
            for key in required_info:
                results[key] = getattr(env, key)
            return results
    
    def rollout_batch_episode(self, 
                              envs, 
                              seeds=None,
                              para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc']='dummy',
                              asynchronous: Literal[None, 'idle', 'restart', 'continue']=None,
                              num_cpus: Optional[Union[int, None]]=1,
                              num_gpus: int=0,
                              required_info=['normalizer', 'gbest']):
        if self.device != 'cpu':
            num_gpus = max(num_gpus, 1)
        env = ParallelEnv(envs, para_mode, asynchronous, num_cpus, num_gpus)
        env.seed(seeds)
        state = env.reset()
        try:
            state = torch.FloatTensor(state).to(self.device)
        except:
            pass
        
        R = torch.zeros(len(env))
        # sample trajectory
        while not env.all_done():
            with torch.no_grad():
                action = self.get_action(state)
            

            # state transient
            state, rewards, is_end, info = env.step(action)
            # print('step:{},max_reward:{}'.format(t,torch.max(rewards)))
            R += torch.FloatTensor(rewards).squeeze()
            # store info
            try:
                state = torch.FloatTensor(state).to(self.device)
            except:
                pass
        _Rs = R.detach().numpy().tolist()
        env_cost = env.get_env_attr('cost')
        env_fes = env.get_env_attr('fes')
        results = {'cost': env_cost, 'fes': env_fes, 'return': _Rs}
        for key in required_info:
            results[key] = env.get_env_attr(key)
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