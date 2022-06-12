
import numpy as np
import matplotlib.pyplot as plt 
import gym


# Number of episodes played
EPOCHS = 20000
# Learning Rate
ALPHA  = 0.8
# Discount Rate
GAMMA  = 0.95
# Greedy Epsilon term
epsilon     = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate  = 0.001
# interval to see where we are
log_interval = 1000



# choose env
env = gym.make('CartPole-v1')
# reset env (just in case)
observation = env.reset()


for episode in range(EPOCHS):


    env.render()
    # ACTION
    action = env.action_space.sample()
    # OBSERVATION
    observation,reward,done,info = env.step(action)
    


env.close()
