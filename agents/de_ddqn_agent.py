from agents.networks import MLP
from basic_agent.DDQN_Agent import *

class DE_DDQN_Agent(DDQN_Agent):
    def __init__(self, config):
        self.config = config
        self.config.state_size = 99
        self.config.n_act = 4
        self.config.lr_model = 1e-4
        # origin DEDDQN doesn't have decay
        self.config.lr_decay = 1
        self.config.batch_size = 64
        self.config.epsilon = 0.1
        self.config.gamma = 0.99
        self.config.target_update_interval = 1000
        self.config.memory_size = 100000
        self.config.warm_up_size = 10000
        self.config.net_config = [{'in': config.state_size, 'out': 100, 'drop_out': 0, 'activation': 'ReLU'},
                             {'in': 100, 'out': 100, 'drop_out': 0, 'activation': 'ReLU'},
                             {'in': 100, 'out': 100, 'drop_out': 0, 'activation': 'ReLU'},
                             {'in': 100, 'out': 100, 'drop_out': 0, 'activation': 'ReLU'},
                             {'in': 100, 'out': config.n_act, 'drop_out': 0, 'activation': 'None'}]
    
        self.config.device = config.device
        # origin DEDDQN doesn't have clip 
        self.config.max_grad_norm = math.inf

        # self.target_model is defined in DDQN_Agent

        self.config.optimizer = 'Adam'
        # origin code does not have lr_scheduler
        self.config.lr_scheduler = 'ExponentialLR'
        self.config.criterion = 'MSELoss'

        model = MLP(self.config.net_config).to(self.config.device)
        # self.__n_act = config.n_act
        # self.__epsilon = config.epsilon
        # self.__gamma = config.gamma
        # self.__update_target_steps = config.update_target_steps
        # self.__batch_size = config.batch_size
        # self.__replay_buffer = ReplayBuffer(config.memory_size)
        # self.__warm_up_size = config.warm_up_size
        # self.__max_learning_step = config.max_learning_step
        # self.__global_ls = 0

        # self.__cur_checkpoint=0
         
        super().__init__(self.config, {'model': model}, self.config.lr_model)
        
        # # save init agent
        # if self.__cur_checkpoint==0:
        #     save_class(self.__config.agent_save_dir,'checkpoint'+str(self.__cur_checkpoint),self)
        #     self.__cur_checkpoint+=1
    def __str__(self):
        return "DEDDEQN"

    # def update_setting(self, config):
    #     self.max_learning_step = config.max_learning_step
    #     self.__config.agent_save_dir = config.agent_save_dir
    #     self.__global_ls = 0
    #     save_class(self.__config.agent_save_dir, 'checkpoint0', self)
    #     self.__config.save_interval = config.save_interval
    #     self.__cur_checkpoint = 1


    # def __get_action(self, state, options=None):
    #     state = torch.Tensor(state).to(self.__device)
    #     action = None
    #     Q_list = self.model(state)
    #     if options['epsilon_greedy'] and np.random.rand() < self.__epsilon:
    #         action = np.random.randint(low=0, high=self.__n_act)
    #     if action is None:
    #         action = int(torch.argmax(Q_list).detach().cpu().numpy())
    #     Q = Q_list[action].detach().cpu().numpy()
    #     return action, Q

    # def train_episode(self, env):
    #     state = env.reset()
    #     done = False
    #     R = 0
    #     while not done:
    #         action = self.get_action(state, {'epsilon_greedy': True})
    #         next_state, reward, done = env.step(action)
    #         R += reward
    #         self.__replay_buffer.append((state, action, reward, next_state, done))
    #         # backward propagation
    #         if len(self.__replay_buffer) >= self.__warm_up_size:
    #             batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = self.__replay_buffer.sample(self.__batch_size)
    #             pred_Vs = self.__pred_func(batch_obs.to(self.__device))  # [batch_size, n_act]
    #             action_onehot = torch.nn.functional.one_hot(batch_action.to(self.__device), self.__n_act)  # [batch_size, n_act]
    #             predict_Q = (pred_Vs * action_onehot).sum(1)  # [batch_size]
    #             target_Q = batch_reward.to(self.__device) + (1 - batch_done.to(self.__device)) * self.__gamma * self.__target_func(batch_next_obs.to(self.__device)).max(1)[0]
    #             self.__optimizer.zero_grad()
    #             loss = self.__criterion(predict_Q, target_Q)
    #             loss.backward()
    #             self.__optimizer.step()
    #             self.__global_ls += 1
                
    #             if self.__global_ls >= (self.__config.save_interval * self.__cur_checkpoint):
    #                 save_class(self.__config.agent_save_dir,'checkpoint'+str(self.__cur_checkpoint),self)
    #                 self.__cur_checkpoint+=1

    #             if self.__global_ls >= self.__max_learning_step:
    #                 break
    #         # sync target network
    #         if self.__global_ls % self.__update_target_steps == 0:
    #             for target_parma, parma in zip(self.__target_func.parameters(), self.__pred_func.parameters()):
    #                 target_parma.data.copy_(parma.data)
    #         state = next_state
    #     return self.__global_ls >= self.__max_learning_step, {'normalizer': env.optimizer.cost[0],
    #                                                           'gbest': env.optimizer.cost[-1],
    #                                                           'return': R,
    #                                                           'learn_steps': self.__global_ls}

    # def rollout_episode(self, env):
    #     state = env.reset()
    #     done = False
    #     R=0
    #     while not done:
    #         action= self.__get_action(state, {'epsilon_greedy': False})
    #         next_state, reward, done = env.step(action)
    #         state = next_state
    #         R+=reward
    #     return {'cost': env.optimizer.cost, 'fes': env.optimizer.fes,'return':R}
