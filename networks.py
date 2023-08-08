import os
import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_dims, fc1_dims, fc2_dims, action_dims,
                 name, chkpt_file='checkpoint'):
        super(ActorNetwork, self).__init__()
        self.state_dims = state_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.action_dims = action_dims
        self.name = name
        self.checkpoint_dir = chkpt_file
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 
                                            self.name+'_td3')
        
        self.fc1 = nn.Linear(self.state_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.action_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = T.device('cuda:0' if T.cuda.is_available() 
                               else 'cuda:1')
        
        self.to(self.device)

    def forward(self, state):
        action = self.fc1(state)
        action = F.relu(action)
        action = self.fc2(action)
        action = F.relu(action)

        action = F.tanh(self.mu(action))

        return action
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        print('... saving best checkpoint ...')
        T.save(self.state_dict(), os.path.join(self.checkpoint_dir, 
                                               self.name+'_best'))

class CriticNetwork(nn.Module):
    def __init__(self, beta, state_dims, fc1_dims, fc2_dims, action_dims,
                 name, chkpt_dir='checkpoint'):
        super(CriticNetwork, self).__init__()
        self.state_dims = state_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.action_dims = action_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            self.name+'_td3')
        
        self.fc1 = nn.Linear(self.state_dims + self.action_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.action_value = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        
        self.device = T.device('cuda:0' if T.cuda.is_available() 
                               else 'cuda:1')
        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        action_value = self.action_value(action_value)

        return action_value
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        print('... saving best checkpoint ...')
        T.save(self.state_dict(), os.path.join(self.checkpoint_dir, 
                                               self.name+'_best'))
