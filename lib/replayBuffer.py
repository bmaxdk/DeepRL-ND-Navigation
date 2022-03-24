# Necessary Packages
import numpy as np
import random
from collections import namedtuple, deque

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """
        Only stores the last N experience tuples in the replay memory
        
        Params
        ======
            action_size (int): Dimension of each action (output_size)
            buffer_size (int): Maximum size of buffer
            batch_size (int): Size of each training batch
            seed (int): Random seed
        """
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)                                # initialize replay memory D with capacity N
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state",
                                                                "action",
                                                                "reward",
                                                                "next_state",
                                                                "done"])       # initialize acollection of experience tuple
        self.seed = random.seed(seed)
        self.device = device
        
    def add(self, state, action, reward, next_state, done):
        """
        Store the agent's experiences to the memory at eatch time-step.
        e_t = (s_t, a_t, r_t, s_(t+1))
        """
        
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self):
        """
        Samples uniformly at random from D(D_t = {e_1, ..., e_t}) when performing update
        This is where we prevent correlation
        """
        
        # D
        experiences = random.sample(self.memory, k=self.batch_size)
        # Store in
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        # return D
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        '''
        Return the current size of internal memory
        '''
        return len(self.memory)
        