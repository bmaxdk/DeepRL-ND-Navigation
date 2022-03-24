# Necessary Packages
import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from network import QNetwork
from replayBuffer import ReplayBuffer

# Hyperparameters
LR = 5e-4                # learning rate
BUFFER_SIZE = int(1e5)   # replay buffer size N
BATCH_SIZE = 64          # minibatch size
UPDATE_EVERY = 4         # how often to update the network
GAMMA = 0.99             # Discount factor
TAU = 1e-3               # for soft update of target parameters


# Setup Gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Builed Agent(): Evaluate ddqn agent on unmodified game
class Agent():
    def __init__(self, state_size, action_size, seed):
        """
        Setup Agent object
        Implement DDQN parameter updates
        
        Params
        ======
            state_size (int): Dimension of each state (input_size)
            action_size (int): Dimension of each action (output_size)
            seed (int): Random seed
        
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # Q-Network: Neural network function approximate with weights theta θ as a Q-Network.
        # A Q-Network can be trained by adjusting the parameters θ_i at iteration i to reduce the mse in the Bellman equation
        # The outputs correspond to the predicted Q-values of the individual action for input state
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        # specify optimizer(Adam)
        # optim.Adam(Qnet.parameters(), small learning rate)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR) ###
        
        # First, use a technique known as experience replay in which we stre the agent's experience at each time-step,
        # e_t= (s_t, a_t, r_t, s_(t_1)), in a data set D_t ={e_1,...,e_t},pooled over many episodes(where the end of an episode occurs when
        # a terminal state is reached) into a replay memory.
        #Initialize replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, device) ###
        self.t_step = 0                                                 # Initialize time step (update every UPDATE_EVERY steps)
        
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)        # save experience in replay memory
        
        # Learn every UPDATE_EVERY time steps
        self.t_step =(self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # if enough samples are availabe in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE: ###
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA) ###
                
    def act(self, state, eps=0):
        '''
        Choose action A from state S using policy pi <- epsilon-Greedt(q^hat (S,A,w))
        Return actions for given state as per current policy.
        
        Params
        ======
            state (array_like): Current state
            eps (float): Epsilon, for epsilon-greedy action selection        
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        # It is off-policy: it learns about the greedy policy a = argmax Q(s,a';θ),
        # while following a behaviour distribution is often selected by an eps-greedy policy
        # that follows the greey policy with probability 1-eps and selects a random action
        # with probability eps.
        # Epsilon-greedy action selection
        # with probability epsilon select a random action a_t
        # otherwise select a_t = argmax_a Q (phi(s_t),a; θ)
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    #DDQN
    def learn(self, experiences, gamma): # only use the local and target Q-networks to compute the loss before taking a step towards minimizing the loss
        '''
        Update value parameters using given batch of experience tuples
        Use DDQN methods
        
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        
        '''
        states, actions, rewards, next_states, dones = experiences
        #### Double DQN
        self.qnetwork_local.eval()
        with torch.no_grad():
            Q_pred = self.qnetwork_local(next_states)                      
            max_actions = torch.argmax(Q_pred, dim=1).long().unsqueeze(1)  # # Fetch max action arguemnt to pass
            Q_next = self.qnetwork_target(next_states)                     # Q_targets over next statesfrom actions will be taken based on Q_pred's max_action
        self.qnetwork_local.train()
        Q_targets = rewards + (gamma * Q_next.gather(1, max_actions) * (1.0 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)        # Get expected Q values from local model


        loss = F.mse_loss(Q_expected, Q_targets)                           # calculate the loss        
        self.optimizer.zero_grad()                                         # zero the parameter (weight) gradients
        loss.backward()                                                    # backward pass to calculate the parameter gradients
        self.optimizer.step()                                              # update the parameters
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)   ###,#Update target network
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        
        
        
        
        
        