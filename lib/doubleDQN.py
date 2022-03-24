# Setting the environment
from agent import Agent

agent = Agent(state_size=37, action_size=4, seed=0)

from collections import deque
import numpy as np
import torch

#Double dqn
def dqn(n_episodes = 2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """
    Train the Agent with Deep Q-Learning
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    
    scores = []                                                 # Initialize collecting scores from each episode
    scores_window = deque(maxlen=100)                           # Initialize collecting maxlen(100) scores
    eps = eps_start                                             # initialize starting value of epsilon
    
    # for each episode
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]       # begin the episode
        state = env_info.vector_observations[0]  
        score = 0                                               # initialize the sampled score(reward)
        
        # Set constrain maximum number of time step per episode
        for t in range(max_t):
            action = agent.act(state, eps)                      # agent select an action
            env_info = env.step(action)[brain_name]             # send the action to the environment
            
            # agent performs the selected action
            #next_state, reward, done, _ = env.step(action)
            next_state = env_info.vector_observations[0]        # get the next state
            reward = env_info.rewards[0]                        # get the reward
            done = env_info.local_done[0]                       # see if episode has finished
    
            agent.step(state, action, reward, next_state, done) # agent performs internal updates based on sampled experience
            # update the sampled reward
            score += reward
            # update the state (s <- s') to next time step
            state = next_state
            if done:
                break
        scores_window.append(score)                              # Save most recent score
        scores.append(score)                                     # Save most recent score
        eps = max(eps_end, eps_decay*eps)                        # Decrease epsilon
        
        # monitor progress
        print('\rEpisode {}\t Average Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        # get average reward from last 100 episodes
        if i_episode % 100 == 0:
            print('\rEpisode {}\t Average Score:{:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=14.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100,
                                                                                         np.mean(scores_window)))
            # save model
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

scores = dqn()


# plot the scores
import matplotlib.pyplot as plt
# %matplotlib inline
from scipy.ndimage.filters import gaussian_filter1d
ysmoothed = gaussian_filter1d(scores, sigma=12)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores, label='DDQN-Dueling Data')
plt.plot(np.arange(len(scores)), ysmoothed, label='Filtered Data')
plt.title("DDQN-Dueling")
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.axhline(y=13, color='r', linestyle='--', label='Avg. Score = 13')
plt.legend(loc='best')
plt.savefig('DDQN-Dueling.png')
plt.savefig('DDQN-Dueling.pdf')
plt.show()