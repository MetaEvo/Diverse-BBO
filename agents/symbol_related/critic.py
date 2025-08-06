import torch
import torch.nn as nn
from .utils import MLP

class Critic(nn.Module):
    def __init__(self, fea_dim, value_dim) -> None:
        super().__init__()
        self.input_dim = fea_dim
        self.output_dim = value_dim
        
        self.value_net=nn.Linear(self.input_dim,self.output_dim)

    # return baseline value detach & baseling value
    def forward(self,x):
        
        baseline_val=self.value_net(x)

        return baseline_val.detach().squeeze(),baseline_val.squeeze()