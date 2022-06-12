
import numpy as np
import matplotlib.pyplot as plt 
import gym




NUM_BINS = 10
# Number of episodes played
EPOCHS = 20000
# Learning Rate
ALPHA  = 0.8
# Discount Rate
GAMMA  = 0.95





def create_bins(num_bins_per_action=10):

    bins_cart_position         = np.linspace( -4.8 , 4.8 ,num_bins_per_action)
    bins_cart_velocity         = np.linspace(  -5  ,  5  ,num_bins_per_action)
    bins_pole_angle            = np.linspace(-0.418,0.418,num_bins_per_action)
    bins_pole_angular_velocity = np.linspace(  -5  ,  5  ,num_bins_per_action)

    bins = np.array([bins_cart_position,bins_cart_velocity,bins_pole_angle,bins_pole_angular_velocity])

    return bins



def discretize_observation_function(observations,bins):
    # init matrix
    binned_observations = []
    # for every values in a row of observation
    for i,observation in enumerate(observations):
        # discretize the value
        discretized_observation = np.digitize(observation,bins[i])
        # add it into matrix
        binned_observations.append(discretized_observation)
    # return the final binned values
    return tuple(binned_observations)



# choose env
env = gym.make('CartPole-v1')
# reset env (just in case)
observation = env.reset()


Q_table_shape = (NUM_BINS,NUM_BINS,NUM_BINS,NUM_BINS,env.action_space.n)
Q_table = np.zeros([Q_table_shape])

"""
for episode in range(EPOCHS):

    env.render()
    # ACTION
    action = env.action_space.sample()
    # OBSERVATION
    observation,reward,done,info = env.step(action)
    
env.close()
"""