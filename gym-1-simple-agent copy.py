from cgi import test
import os
from statistics import mode
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import gym
import time
from gym.utils import play


def simple_agent(observation):
    # OBSERVATION
    position, velocity = observation

    if -0.1 < position < 0.4 :
        action = 2
    elif velocity < 0 and position < -0.2:
        action = 0
    else:
        action = 1
    # ACTION
    return action

# choose env
env = gym.make('MountainCar-v0')
# random seed
env.seed(101)
# reset env (just in case)
observation = env.reset()

for step in range(200):

    # 0 : <--
    # 1 : stays put
    # 2 : -->

    # render the step
    env.render()
    # take random actions
    action = simple_agent(observation)
    # render the step according to the action
    observation,reward,done,info = env.step(action)
    # print stuff
    print(f"Reward {reward}")
    print(f"Done   {done}")
    print(f"Info   {info}")

    # if the game is won
    if done :
        # exit the loop
        break

    #time.sleep(0.1)

env.close()