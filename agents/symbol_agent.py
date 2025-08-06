from basic_agent.PPO_Agent import *
from .symbol_related.critic import Critic
from .symbol_related.expression import *
from .symbol_related.lstm import LSTM
from .symbol_related.tokenizer import MyTokenizer


class Data_Memory():
    def __init__(self) -> None:
        self.teacher_cost = []
        self.stu_cost = []
        self.baseline_cost = []
        self.gap = []
        self.baseline_gap = []
        self.expr = []

    def clear(self):
        del self.teacher_cost[:]
        del self.stu_cost[:]
        del self.baseline_cost[:]
        del self.gap[:]
        del self.baseline_gap[:]
        del self.expr[:]

# memory for recording transition during training process
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.gap_rewards = []
        self.b_rewards = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.gap_rewards[:]
        del self.b_rewards[:]

class SYMBOL_Agent(PPO_Agent):
    def __init__(self, config):
        self.config = config

        self.config.optimizer = 'Adam'
        self.config.init_pop = 'random'
        self.config.teacher = 'MadDE'
        self.config.population_size = 100
        self.config.boarder_method = 'clipping'
        self.config.skip_step = 5
        self.config.test_skip_step = 5
        self.config.max_c = 1.
        self.config.min_c = -1.
        self.config.c_interval = 0.4
        self.config.max_layer = 6
        self.config.value_dim = 1
        self.config.hidden_dim = 16
        self.config.num_layers = 1
        self.config.lr = 1e-3
        self.config.lr_critic = 1e-3
        self.config.max_grad_norm = math.inf

        self.config.encoder_head_num = 4
        self.config.decoder_head_num = 4
        self.config.critic_head_num = 4
        self.config.embedding_dim = 16
        self.config.n_encode_layers = 1
        self.config.normalization = 'layer'
        self.config.hidden_dim1_critic = 32
        self.config.hidden_dim2_critic = 16
        self.config.hidden_dim1_actor = 32
        self.config.hidden_dim2_actor = 8
        self.config.output_dim_actor = 1
        # self.config.lr_decay = 0.9862327
        self.config.gamma = 0.99
        self.config.K_epochs = 3
        self.config.eps_clip = 0.1
        self.config.n_step = 10

        self.config.fea_dim = 9

        self.tokenizer = MyTokenizer()

        actor = LSTM(max_layer = self.config.max_layer,
                     hidden_dim = self.config.hidden_dim,
                     num_layers = self.config.num_layers,
                     max_c = self.config.max_c,
                     min_c = self.config.min_c,
                     fea_dim = self.config.fea_dim,
                     c_interval = self.config.c_interval,
                     tokenizer = self.tokenizer)
        critic = Critic(fea_dim = self.config.fea_dim,
                        value_dim = self.config.value_dim)

        super().__init__(self.config, {'actor': actor, 'critic': critic}, [self.config.lr, self.config.lr_critic])

    def __str__(self):
        return "SYMBOL"

    def train_episode(self,
                      envs,
                      seeds: Optional[Union[int, List[int], np.ndarray]],
                      para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc'] = 'dummy',
                      asynchronous: Literal[None, 'idle', 'restart', 'continue'] = None,
                      num_cpus: Optional[Union[int, None]] = 1,
                      num_gpus: int = 0,
                      tb_logger=None,
                      required_info = []):
        if self.device != 'cpu':
            num_gpus = max(num_gpus, 1)

        # set env.optimizer.is_train = True
        for env in envs:
            env.optimizer.is_train = True
        env = ParallelEnv(envs, para_mode, asynchronous, num_cpus, num_gpus)
        # set env.optimizer.is_train = True
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
        _loss = []
        # sample trajectory
        while not env.all_done():
            t_s = t
            total_cost = 0
            entropy = []
            bl_val_detached = []
            bl_val = []

            # accumulate transition
            while t - t_s < n_step and not env.all_done():

                memory.states.append(state.clone())

                self.config.require_baseline = False
                seq, const_seq, log_prob, action_dict = self.actor(state, save_data = True)

                # critic network
                baseline_val_detached, baseline_val = self.critic(state)
                bl_val_detached.append(baseline_val_detached)
                bl_val.append(baseline_val)

                # store reward for ppo
                memory.actions.append(action_dict)
                memory.logprobs.append(log_prob)

                action = []
                for s, cs in zip(seq, const_seq):
                    expr = construct_action(seq = s, const_seq = cs, tokenizer = self.tokenizer)
                    action.append({'expr': expr, 'skip_step': self.config.skip_step})

                # expr = construct_action(seq = seq, const_seq = const_seq, tokenizer = self.tokenizer)
                # action = {'expr': expr, 'skip_step': self.config.skip_step}
                state, reward, is_done, info = env.step(action)
                memory.rewards.append(reward)
                _R += reward

                t = t + 1

                try:
                    state = torch.FloatTensor(state).to(self.device)
                except:
                    pass

            # store info
            t_time = t - t_s
            total_cost = total_cost / t_time

            # begin update
            old_actions = memory.actions
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
                        log_p = self.actor(old_states[tt],fix_action = old_actions[tt])

                        logprobs.append(log_p)

                        baseline_val_detached, baseline_val = self.critic(old_states[tt])

                        bl_val_detached.append(baseline_val_detached)
                        bl_val.append(baseline_val)
                logprobs = torch.stack(logprobs).view(-1)
                bl_val_detached = torch.stack(bl_val_detached).view(-1)
                bl_val = torch.stack(bl_val).view(-1)

                # get traget value for critic
                Reward = []
                reward_reversed = memory.rewards[::-1]
                # get next value
                R = self.critic(state)[0]
                critic_output = R.clone()
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

                if torch.isnan(loss):
                    print(f'baseline_loss:{baseline_loss}')
                    print(f'reinforce_loss:{reinforce_loss}')
                    assert True, 'nan found in loss!!'

                # update gradient step
                self.optimizer.zero_grad()
                loss.backward()
                _loss.append(loss.item())

                grad_norms = clip_grad_norms(self.optimizer.param_groups, self.config.max_grad_norm)
                # Clip gradient norm and get (clipped) gradient norms for logging
                # current_step = int(pre_step + t//n_step * K_epochs  + _k)
                # grad_norms = clip_grad_norms(self.optimizer.param_groups, self.config.max_grad_norm)

                # perform gradient descent
                self.optimizer.step()
                self.learning_time += 1
                if self.learning_time >= (self.config.save_interval * self.cur_checkpoint) and self.config.end_mode == "step":
                    save_class(self.config.agent_save_dir, 'checkpoint' + str(self.cur_checkpoint), self)
                    self.cur_checkpoint += 1

                if not self.config.no_tb and self.learning_time % int(self.config.log_step) == 0:
                    self.log_to_tb_train(tb_logger, self.learning_time,
                                         grad_norms,
                                         reinforce_loss, baseline_loss,
                                         _R, Reward, memory.rewards,
                                         critic_output, logprobs, entropy, approx_kl_divergence)

                if self.learning_time >= self.config.max_learning_step:
                    memory.clear_memory()
                    _Rs = _R.detach().numpy().tolist()
                    return_info = {'return': _Rs, 'loss': np.mean(_loss), 'learn_steps': self.learning_time, }
                    env_cost = env.get_env_attr('cost')
                    return_info['normalizer'] = env_cost[0]
                    return_info['gbest'] = env_cost[-1]
                    for key in required_info:
                        return_info[key] = env.get_env_attr(key)
                    env.close()
                    return self.learning_time >= self.config.max_learning_step, return_info

            memory.clear_memory()

        is_train_ended = self.learning_time >= self.config.max_learning_step
        _Rs = _R.detach().numpy().tolist()
        return_info = {'return': _Rs, 'loss': np.mean(_loss), 'learn_steps': self.learning_time, }
        env_cost = env.get_env_attr('cost')
        return_info['normalizer'] = env_cost[0]
        return_info['gbest'] = env_cost[-1]
        for key in required_info:
            return_info[key] = env.get_env_attr(key)
        env.close()
        return is_train_ended, return_info

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
        # set env.optimizer.is_train = False
        for env in envs:
            env.optimizer.is_train = False

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
                seq, const_seq, log_prob = self.actor(state, save_data = False)
            action = []
            for s, cs in zip(seq, const_seq):
                expr = construct_action(seq = s, const_seq = cs, tokenizer = self.tokenizer)
                action.append({'expr': expr, 'skip_step': self.config.skip_step})

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
        for key in required_info.keys():
            results[key] = env.get_env_attr(required_info[key])
        return results

def construct_action(seq, const_seq, tokenizer):
    pre,c_pre = get_prefix_with_consts(seq, const_seq, 0)
    str_expr = [tokenizer.decode(pre[i]) for i in range(len(pre))]
    success,infix = prefix_to_infix(str_expr, c_pre, tokenizer)
    assert success, 'fail to construct the update function'

    return infix

