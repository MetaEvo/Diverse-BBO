import collections
import torch
import random
import numpy as np
import pickle 
import os


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


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*mini_batch)
        # print(type(obs_batch),type(action_batch),type(reward_batch),type(next_obs_batch),type(done_batch))
        # print(type(action_batch[0]))
        # obs_batch = torch.FloatTensor(np.array(obs_batch))
        obs_batch = torch.stack(obs_batch).float()
        action_batch = torch.tensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        next_obs_batch = torch.stack(next_obs_batch).float()
        # next_obs_batch = torch.FloatTensor(np.array(next_obs_batch))
        done_batch = torch.FloatTensor(done_batch)
        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def __len__(self):
        return len(self.buffer)


class ReplayBuffer_torch:
    def __init__(self, capacity, state_dim, device):

        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0  

    
        self.states = torch.zeros((capacity, state_dim), device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.next_states = torch.zeros((capacity, state_dim), device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float, device=device)

    def append(self, state, action, reward, next_state, done):

        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done


        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):

        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size

def save_class(dir, file_name, saving_class):
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(dir+file_name+'.pkl', 'wb') as f:
        pickle.dump(saving_class, f, -1)
