# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 14:39:30 2019

@author: Michael

Contains all classes and definitions needed for deep Q-learning, i.e. RL with neural networks for Bomberman
"""

import random
import numpy as np

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import csv


### replay memory ###

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class Memory(object):
    def __init__(self, max_len):
        self.max_len = max_len
        self.mem = []
        self.pos = 0

    def save(self, *args):
        """Saves a transition."""
        if len(self.mem) < self.max_len:
            self.mem.append(None)
        self.mem[self.pos] = Transition(*args)
        self.pos = (self.pos + 1) % self.max_len

    def sample(self, sample_size):
        return random.sample(self.mem, sample_size)

    def __len__(self):
        return len(self.mem)

### Deep Q-Network (DQN) Model ###

class DQN(nn.Module):

    # 3 layer convolution network plus linear output layer with 2 input channels and 6 output features;
    #   Conv2d shapes: Input: (N, C_{in}, H_{in}, W_{in}), Output: (N, C_{out}, H_{out}, W_{out})
    #   where H_{out}, W_{out} = floor( ( array([H_{in},W_{in}])​ − (kernel_size − 1 ) − 1)/stride + 1 )
    def __init__(self, h, w, c=8):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(c, 16, kernel_size=4, stride=2)  # in_channels = 2, out_channels = 16 
        self.bn1 = nn.BatchNorm2d(16)  # 2dim batch normalization; C=16; input: (N,C,H,W)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=2, stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        # calculate out_size (H_{out} or W_{out} in comment above) for Conv2d given in_size ([H_{in} or W_{in})
        def get_out_size(in_size, kernel_size=5, stride=1):
            return (in_size - kernel_size) // stride + 1 
        
        h_out = get_out_size(get_out_size(get_out_size(h,4,2),2),2)
        w_out = get_out_size(get_out_size(get_out_size(w,4,2),2),2)
        self.head = nn.Linear(h_out*w_out*32, 6)    # linear layer, Linear(in_features, out_features);
                                                    # input: (N,∗,in_features), 
                                                    # output: (N,∗,out_features)

    # Called with either one element to determine next action, or a batch during optimization.
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))     # rectified linear unit function (element-wise) ReLu(x) = max(0,x) 
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1)) # view() is the analogue to np.reshape()

### optimizer step ###

def optimize_model(self):
    memory = self.memory
    N = self.batch_size
    pnet = self.pnet
    tnet = self.tnet
    
    if len(memory) < N:         # not enough data
        return
    
    # get sample
    transitions = memory.sample(N)      # list of N Transition objects

    # we want Transition of batch-arrays (actually tuples) instead of a batch array (actually list) of Transitions
    batch = Transition(*zip(*transitions))

    # take care of terminal states 
    mask = torch.tensor([True if s is not None else False for s in batch.next_state ],
                        dtype=torch.uint8, device=self.device)
    next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    # make everything a tensor (each containing N elements)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # in order to get Q(s_t, a) we let the net calculate Q(s_t), then we select 
    # the columns corresponding to the actions taken

    state_action_values = pnet(state_batch)[torch.arange(N, dtype=torch.long, device=self.device), action_batch]

    # Compute Q(s_{t+1}) for all next states using target_net
    #   (0 in case of terminal state)
    next_state_values = torch.zeros(N, device=self.device)
    next_state_values[mask] = tnet(next_states).max().detach()
    
    # Compute the expected Q values using the rewards
    expected_state_action_values = (next_state_values * self.gamma) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    #self.logger.debug(f'loss={loss}')
    
    with open('loss.csv','a') as fd:
        wr = csv.writer(fd)
        wr.writerow([loss])
    
    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    for param in pnet.parameters():
        param.grad.data.clamp_(-1, 1)
    self.optimizer.step()











