import copy
import math
from typing import Optional, Union, Literal, List

from VectorEnv.great_para_env import ParallelEnv
from basic_agent.basic_agent import Basic_Agent
from basic_agent.utils import *


# memory for recording transition during training process
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]


def clip_grad_norms(param_groups, max_norm = math.inf):
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
            norm_type = 2
        )
        for idx, group in enumerate(param_groups)
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


class PPO_Agent(Basic_Agent):
    def __init__(self, config, networks: dict, learning_rates: Optional):
        super().__init__(config)
        self.config = config

        # define parameters
        self.gamma = self.config.gamma
        self.n_step = self.config.n_step
        self.K_epochs = self.config.K_epochs
        self.eps_clip = self.config.eps_clip
        self.max_grad_norm = self.config.max_grad_norm
        self.device = self.config.device
        self.set_network(networks, learning_rates)
        # figure out the actor network
        # self.actor = None

        # figure out the critic network
        # self.critic = None
        # assert hasattr(self, 'actor') and hasattr(self, 'critic')
        #
        # # figure out the optimizer
        # assert hasattr(torch.optim, self.config.optimizer)
        # self.optimizer = eval('torch.optim.' + self.config.optimizer)(
        #     [{'params': self.actor.parameters(), 'lr': self.config.lr_actor}] +
        #     [{'params': self.critic.parameters(), 'lr': self.config.lr_critic}])
        # figure out the lr schedule
        # assert hasattr(torch.optim.lr_scheduler, self.config.lr_scheduler)

        # move to device
        # self.actor.to(self.device)
        # self.critic.to(self.device)

        # init learning time
        self.learning_time = 0
        self.cur_checkpoint = 0

        # save init agent
        save_class(self.config.agent_save_dir, 'checkpoint' + str(self.cur_checkpoint), self)
        self.cur_checkpoint += 1

    def set_network(self, networks: dict, learning_rates: Optional):
        Network_name = []
        if networks:
            for name, network in networks.items():
                Network_name.append(name)
                setattr(self, name, network)   # Assign each network in the dictionary to the class instance
        self.network = Network_name

        # make sure actor and critic network
        assert hasattr(self, 'actor') and hasattr(self, 'critic')

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

        for network_name in networks:
            getattr(self, network_name).to(self.device)

        # self.lr_scheduler = eval('torch.optim.lr_scheduler.' + self.config.lr_scheduler)(self.optimizer, self.config.lr_decay, last_epoch = -1, )
        
        
    def update_setting(self, config):
        self.config.max_learning_step = config.max_learning_step
        self.config.agent_save_dir = config.agent_save_dir
        self.learning_time = 0
        save_class(self.config.agent_save_dir, 'checkpoint0', self)
        self.config.save_interval = config.save_interval
        self.cur_checkpoint = 1

    def train_episode(self,
                      envs,
                      seeds: Optional[Union[int, List[int], np.ndarray]],
                      para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc'] = 'dummy',
                      asynchronous: Literal[None, 'idle', 'restart', 'continue'] = None,
                      num_cpus: Optional[Union[int, None]] = 1,
                      num_gpus: int = 0,
                      tb_logger = None,
                      required_info = {}):
        if self.device != 'cpu':
            num_gpus = max(num_gpus, 1)
        env = ParallelEnv(envs, para_mode, asynchronous, num_cpus, num_gpus)
        env.seed(seeds)
        memory = Memory()

        # params for training
        gamma = self.gamma
        n_step = self.n_step

        K_epochs = self.K_epochs
        eps_clip = self.eps_clip

        state = env.reset()
        try:
            state = torch.FloatTensor(state).to(self.device)
        except:
            pass

        t = 0
        # initial_cost = obj
        _R = torch.zeros(len(env))
        # sample trajectory
        while not env.all_done():
            t_s = t
            total_cost = 0
            entropy = []
            bl_val_detached = []
            bl_val = []

            # accumulate transition
            while t - t_s < n_step:

                memory.states.append(state.clone())
                action, log_lh, entro_p = self.actor(state)

                memory.actions.append(action.clone() if isinstance(action, torch.Tensor) else copy.deepcopy(action))
                memory.logprobs.append(log_lh)

                entropy.append(entro_p.detach().cpu())

                baseline_val = self.critic(state)
                baseline_val_detached = baseline_val.detach()

                bl_val_detached.append(baseline_val_detached)
                bl_val.append(baseline_val)

                # state transient
                state, rewards, is_end, info = env.step(action)
                memory.rewards.append(torch.FloatTensor(rewards).to(self.device))
                # print('step:{},max_reward:{}'.format(t,torch.max(rewards)))
                _R += rewards
                # store info

                # next
                t = t + 1

                try:
                    state = torch.FloatTensor(state).to(self.device)
                except:
                    pass

            # store info
            t_time = t - t_s
            total_cost = total_cost / t_time

            # begin update
            old_actions = torch.stack(memory.actions)
            try:
                old_states = torch.stack(memory.states).detach()  # .view(t_time, bs, ps, dim_f)
            except:
                pass
            # old_actions = all_actions.view(t_time, bs, ps, -1)
            old_logprobs = torch.stack(memory.logprobs).detach().view(-1)

            # Optimize PPO policy for K mini-epochs:
            old_value = None
            for _k in range(K_epochs):
                if _k == 0:
                    logprobs = memory.logprobs

                else:
                    # Evaluating old actions and values :
                    logprobs = []
                    entropy = []
                    bl_val_detached = []
                    bl_val = []

                    for tt in range(t_time):
                        # get new action_prob
                        _, log_p, entro_p = self.actor(old_states[tt], old_actions[tt])

                        logprobs.append(log_p)
                        entropy.append(entro_p.detach().cpu())

                        baseline_val = self.critic(old_states[tt])
                        baseline_val_detached = baseline_val.detach()

                        bl_val_detached.append(baseline_val_detached)
                        bl_val.append(baseline_val)

                logprobs = torch.stack(logprobs).view(-1)
                entropy = torch.stack(entropy).view(-1)
                bl_val_detached = torch.stack(bl_val_detached).view(-1)
                bl_val = torch.stack(bl_val).view(-1)

                # get traget value for critic
                Reward = []
                reward_reversed = memory.rewards[::-1]
                # get next value
                R = self.critic(self.actor(state))[0]

                for r in range(len(reward_reversed)):
                    R = R * gamma + reward_reversed[r]
                    Reward.append(R)
                # clip the target:
                Reward = torch.stack(Reward[::-1], 0)
                Reward = Reward.view(-1)

                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss:
                advantages = Reward - bl_val_detached

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
                reinforce_loss = -torch.min(surr1, surr2).mean()

                # define baseline loss
                if old_value is None:
                    baseline_loss = ((bl_val - Reward) ** 2).mean()
                    old_value = bl_val.detach()
                else:
                    vpredclipped = old_value + torch.clamp(bl_val - old_value, - eps_clip, eps_clip)
                    v_max = torch.max(((bl_val - Reward) ** 2), ((vpredclipped - Reward) ** 2))
                    baseline_loss = v_max.mean()

                # check K-L divergence (for logging only)
                approx_kl_divergence = (.5 * (old_logprobs.detach() - logprobs) ** 2).mean().detach()
                approx_kl_divergence[torch.isinf(approx_kl_divergence)] = 0
                # calculate loss
                loss = baseline_loss + reinforce_loss

                # update gradient step
                self.optimizer.zero_grad()
                loss.backward()

                # Clip gradient norm and get (clipped) gradient norms for logging
                # current_step = int(pre_step + t//n_step * K_epochs  + _k)
                grad_norms = clip_grad_norms(self.optimizer.param_groups, self.config.max_grad_norm)

                # perform gradient descent
                self.optimizer.step()
                self.learning_time += 1
                if self.learning_time >= (self.config.save_interval * self.cur_checkpoint) and self.config.end_mode == "step":
                    save_class(self.config.agent_save_dir, 'checkpoint' + str(self.cur_checkpoint), self)
                    self.cur_checkpoint += 1

                if self.learning_time >= self.config.max_learning_step:
                    memory.clear_memory()
                    return_info = {'return': _R, 'learn_steps': self.learning_time, }
                    env_cost = env.get_env_attr('cost')
                    return_info['normalizer'] = env_cost[0]
                    return_info['gbest'] = env_cost[-1]
                    for key in required_info.keys():
                        return_info[key] = env.get_env_attr(required_info[key])
                    env.close()
                    return self.learning_time >= self.config.max_learning_step, return_info

            memory.clear_memory()

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
                        seed = None,
                        required_info = {}):
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
                    state = [state]
                action = self.actor(state)[0]
                action = action.cpu().numpy().squeeze()
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
                              seeds = None,
                              para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc'] = 'dummy',
                              asynchronous: Literal[None, 'idle', 'restart', 'continue'] = None,
                              num_cpus: Optional[Union[int, None]] = 1,
                              num_gpus: int = 0,
                              required_info = {}):
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
        entropy = []
        # sample trajectory
        while not env.all_done():
            with torch.no_grad():
                action, log_lh, entro_p = self.actor(state)

            entropy.append(entro_p.detach().cpu())

            # state transient
            state, rewards, is_end, info = env.step(action)
            # print('step:{},max_reward:{}'.format(t,torch.max(rewards)))
            R += torch.FloatTensor(rewards).squeeze()
            # store info
            try:
                state = torch.FloatTensor(state).to(self.device)
            except:
                pass
        env_cost = env.get_env_attr('cost')
        env_fes = env.get_env_attr('fes')
        results = {'cost': env_cost, 'fes': env_fes, 'return': R}
        for key in required_info.keys():
            results[key] = env.get_env_attr(required_info[key])
        return results

    def log_to_tb_train(self, tb_logger, mini_step,
                        grad_norms, # network grad
                        reinforce_loss, baseline_loss, # actor loss critic loss
                        Return, Reward, memory_reward,
                        critic_output,
                        logprobs, entropy,
                        approx_kl_divergence,
                        extra_info = {}):
        # Iterate over the extra_info dictionary and log data to tb_logger
        # extra_info: Dict[str, Dict[str, Union[List[str], List[Union[int, float]]]]] = {
        #     "loss": {"name": [], "data": [0.5]},  # No "name", logs under "loss"
        #     "accuracy": {"name": ["top1", "top5"], "data": [85.2, 92.5]},  # Logs as "accuracy/top1" and "accuracy/top5"
        #     "learning_rate": {"name": ["adam", "sgd"], "data": [0.001, 0.01]}  # Logs as "learning_rate/adam" and "learning_rate/sgd"
        # }
        #
        # learning rate
        for id, network_name in enumerate(self.network):
            tb_logger.add_scalar(f'learnrate/{network_name}', self.optimizer.param_groups[id]['lr'], mini_step)

        # grad and clipped grad
        grad_norms, grad_norms_clipped = grad_norms
        for id, network_name in enumerate(self.network):
            tb_logger.add_scalar(f'grad/{network_name}', grad_norms[id], mini_step)
            tb_logger.add_scalar(f'grad_clipped/{network_name}', grad_norms_clipped[id], mini_step)


        # loss
        tb_logger.add_scalar('loss/actor_loss', reinforce_loss.item(), mini_step)
        tb_logger.add_scalar('loss/critic_loss', baseline_loss.item(), mini_step)
        tb_logger.add_scalar('loss/total_loss', (reinforce_loss + baseline_loss).item(), mini_step)

        # train metric
        # avg_reward = torch.stack(memory_reward).mean().item()
        avg_reward = torch.tensor(np.mean(memory_reward)).item()
        # max_reward = torch.stack(memory_reward).max().item()
        max_reward = torch.tensor(np.max(memory_reward)).item()

        tb_logger.add_scalar('train/episode_avg_return', Return.mean().item(), mini_step)
        tb_logger.add_scalar('train/target_avg_return_changed', Reward.mean().item(), mini_step)
        tb_logger.add_scalar('train/critic_avg_output', critic_output.mean().item(), mini_step)
        tb_logger.add_scalar('train/avg_entropy', torch.tensor(entropy).mean().item(), mini_step)
        tb_logger.add_scalar('train/-avg_logprobs', -logprobs.mean().item(), mini_step)
        tb_logger.add_scalar('train/approx_kl', approx_kl_divergence.item(), mini_step)
        tb_logger.add_scalar('train/avg_reward', avg_reward, mini_step)
        tb_logger.add_scalar('train/max_reward', max_reward, mini_step)

        # extra info
        for key, value in extra_info.items():
            if not value['name']:
                tb_logger.add_scalar(f'{key}', value['data'][0], mini_step)
            else:
                name_list = value['name']
                data_list = value['data']
                for name, data in zip(name_list, data_list):
                    tb_logger.add_scalar(f'{key}/{name}', data, mini_step)

