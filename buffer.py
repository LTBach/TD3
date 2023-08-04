import numpy as np

class ReplayBuffer():
    def __init__(self, mem_size, state_dims, action_dims):
        self.mem_size = mem_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *state_dims))
        self.action_memory = np.zeros((self.mem_size, action_dims))
        self.reward_memory = np.zeros((self.mem_size))
        self.next_state_memory = np.zeros((self.mem_size, *state_dims))
        self.done_memory = np.zeros((self.mem_size), dtype=np.bool)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.done_memory[index] = done

        self.mem_cntr += 1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        dones = self.done_memory[batch]

        return states, actions, rewards, next_states, dones
