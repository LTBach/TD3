import os
import numpy as np
import torch as T
import torch.nn.functional as F
T.autograd.set_detect_anomaly(True)
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork

class Agent():
    def __init__(self, alpha, beta, state_dims, action_dims, env, tau, gamma=0.99, 
                 update_actor_interval=2, warmup=1000, max_size=1000000, 
                 layer1_size=400, layer2_size=300, batch_size=100, noise_sigma=0.1):
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.memory = ReplayBuffer(max_size, state_dims, action_dims)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.action_dims = action_dims
        self.update_actor_iter = update_actor_interval

        self.beh_actor = ActorNetwork(alpha, state_dims, layer1_size, layer2_size,
                                      action_dims, 'beh_actor')
        self.beh_critic_1 = CriticNetwork(beta, state_dims, layer1_size, 
                                          layer2_size, action_dims, 'beh_critic_1')
        self.beh_critic_2 = CriticNetwork(beta, state_dims, layer1_size, 
                                          layer2_size, action_dims, 'beh_critic_2')
        self.tar_actor = ActorNetwork(alpha, state_dims, layer1_size, layer2_size,
                                      action_dims, 'tar_actor')
        self.tar_critic_1 = CriticNetwork(beta, state_dims, layer1_size, 
                                          layer2_size, action_dims, 'tar_critic_1')
        self.tar_critic_2 = CriticNetwork(beta, state_dims, layer1_size, 
                                          layer2_size, action_dims, 'beh_critic_2')
        
        self.noise_sigma = noise_sigma
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.beh_actor.eval()

        if self.time_step < self.warmup:
            mu = T.tensor(np.random.normal(scale=self.noise_sigma, 
                size=(self.action_dims,))).to(self.beh_actor.device)
        else:
            state = T.tensor(observation, dtype=T.float).to(self.beh_actor.device)
            mu = self.beh_actor.forward(state).to(self.beh_actor.device)

        noise = T.tensor(
            np.random.normal(scale=self.noise_sigma, size=(self.action_dims,)),
            dtype=T.float).to(self.beh_actor.device)
        mu_prime = mu + noise
        
        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])

        self.time_step += 1

        self.beh_actor.train()
        return mu_prime.cpu().detach().numpy()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = \
            self.memory.sample_buffer(self.batch_size)
        
        states = T.tensor(states, dtype=T.float).to(self.beh_critic_1.device)
        actions = T.tensor(actions, dtype=T.float).to(self.beh_critic_1.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.beh_critic_1.device)
        next_states = T.tensor(next_states, dtype=T.float).to(self.beh_critic_1.device)
        dones = T.tensor(dones, dtype=T.bool).to(self.beh_critic_1.device)

        target_actions = self.tar_actor.forward(next_states)
        target_actions = target_actions + \
            T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        
        target_actions = T.clamp(target_actions, self.min_action[0], 
                                 self.max_action[0])

        next_action_value_1 = self.tar_critic_1.forward(next_states, target_actions)
        next_action_value_2 = self.tar_critic_2.forward(next_states, target_actions)

        action_value_1 = self.beh_critic_1.forward(states, actions)
        action_value_2 = self.beh_critic_2.forward(states, actions)

        next_action_value_1[dones] = 0.0
        next_action_value_2[dones] = 0.0

        next_action_value_1 = next_action_value_1.view(-1)
        next_action_value_2 = next_action_value_2.view(-1)

        next_action_value = T.min(next_action_value_1, next_action_value_2)
        
        target = rewards + self.gamma * next_action_value
        target = target.view(self.batch_size, 1)

        self.beh_critic_1.optimizer.zero_grad()
        self.beh_critic_2.optimizer.zero_grad()
        critic_loss_1 = F.mse_loss(target, action_value_1)
        critic_loss_2 = F.mse_loss(target, action_value_2)
        
        critic_loss = critic_loss_1 + critic_loss_2
        critic_loss.backward()
        
        self.beh_critic_1.optimizer.step()
        self.beh_critic_2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return
        
        self.beh_actor.optimizer.zero_grad()
        actor_loss = -self.beh_critic_1.forward(states, 
                                                self.beh_actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.beh_actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        beh_actor_params = self.beh_actor.named_parameters()
        beh_critic_1_params = self.beh_critic_1.named_parameters()
        beh_critic_2_params = self.beh_critic_2.named_parameters()
        tar_actor_paramas = self.tar_actor.named_parameters()
        tar_critic_1_params = self.tar_critic_1.named_parameters()
        tar_critic_2_params = self.tar_critic_2.named_parameters()

        beh_actor_state_dict = dict(beh_actor_params)
        beh_critic_1_state_dict = dict(beh_critic_1_params)
        beh_critic_2_state_dict = dict(beh_critic_2_params)
        tar_actor_state_dict = dict(tar_actor_paramas)
        tar_critic_1_state_dict = dict(tar_critic_1_params)
        tar_critic_2_state_dict = dict(tar_critic_2_params)

        for name in tar_actor_state_dict:
            tar_actor_state_dict[name] = tau*beh_actor_state_dict[name].clone()\
                                    + (1 - tau)*tar_actor_state_dict[name].clone()
        
        for name in tar_critic_1_state_dict:
            tar_critic_1_state_dict[name] = tau*beh_critic_1_state_dict[name].clone()\
                                    + (1 - tau)*tar_critic_1_state_dict[name].clone()

        for name in tar_critic_2_state_dict:
            tar_critic_2_state_dict[name] = tau*beh_critic_2_state_dict[name].clone()\
                                    + (1 - tau)*tar_critic_2_state_dict[name].clone()

        self.tar_actor.load_state_dict(tar_actor_state_dict)
        self.tar_critic_1.load_state_dict(tar_critic_1_state_dict)
        self.tar_critic_2.load_state_dict(tar_critic_2_state_dict)

    def save_models(self):
        self.beh_actor.save_checkpoint()
        self.beh_critic_1.save_checkpoint()
        self.beh_critic_2.save_checkpoint()
        self.tar_actor.save_checkpoint()
        self.tar_critic_1.save_checkpoint()
        self.tar_critic_2.save_checkpoint()

    def load_models(self):
        self.beh_actor.load_checkpoint()
        self.beh_critic_1.load_checkpoint()
        self.beh_critic_2.load_checkpoint()
        self.tar_actor.load_checkpoint()
        self.tar_critic_1.load_checkpoint()
        self.tar_critic_2.load_checkpoint()
        
