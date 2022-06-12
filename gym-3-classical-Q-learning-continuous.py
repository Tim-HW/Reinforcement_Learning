
from pickle import FALSE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
import gym




NUM_BINS = 10
# Number of episodes played
EPOCHS = 20000
# Learning Rate
ALPHA  = 0.8
# Discount Rate
GAMMA  = 0.95
# Epislon value
epsilon         = 1.0
BURN_IN         = 1.0
EPISLON_END     = 10000
EPSILON_REDUCE  = 0.0001

point_log       = []
mean_points_log = []
epochs          = []
log_interval    = 100





#!############################################################################
#!               Reduce epsilon
#!############################################################################

def reduce_epsilon(epsilon,epoch):

    if BURN_IN <= epoch <= EPISLON_END:

        epsilon -= EPSILON_REDUCE

    return epsilon



#!############################################################################
#!               Reward Function
#!############################################################################

def fail(done,points,reward):
    
    if done and points < 150 :
        reward = -200

    return reward

#!############################################################################
#!               Compute new Q values
#!############################################################################
def compute_next_Q(old_Q,reward,next_optimal_Q):

    return old_Q + ALPHA  * (reward + GAMMA * next_optimal_Q - old_Q)

#!#############################################################################
#!               Select Action (exploration vs Exploitation)
#!############################################################################
def epsilon_greedy_actions_selection(epsilon,Q_table,discrete_state):

    # random number [0 and 1]
    random_number = np.random.random()
    # Exploitation (choose the action that maximize Q)
    if random_number > epsilon:
        # choose the max Q action in this row
        action = np.argmax(Q_table[discrete_state])

    # Exploration (choose random action)
    else :
        action = np.random.randint(0,env.action_space.n)

    return action

#!#############################################################################
#!                   Create Bins
#!#############################################################################

def create_bins(num_bins_per_action=10):

    bins_cart_position         = np.linspace( -4.8 , 4.8 ,num_bins_per_action)
    bins_cart_velocity         = np.linspace(  -5  ,  5  ,num_bins_per_action)
    bins_pole_angle            = np.linspace(-0.418,0.418,num_bins_per_action)
    bins_pole_angular_velocity = np.linspace(  -5  ,  5  ,num_bins_per_action)

    bins = np.array([bins_cart_position,bins_cart_velocity,bins_pole_angle,bins_pole_angular_velocity])

    return bins


#!#############################################################################
#!                   discretize observations
#!#############################################################################

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

def animate(i):


    plt.cla()
    plt.plot(epoch,mean_points_log)

#!#############################################################################
#!                               Main
#!#############################################################################

# choose env
env = gym.make('CartPole-v1')
# reset env (just in case)
observation = env.reset()

# initiate Q table
Q_table_shape = (NUM_BINS,NUM_BINS,NUM_BINS,NUM_BINS,env.action_space.n)
Q_table = np.zeros(Q_table_shape)
print((Q_table.shape))
# initiate BINS
BINS = create_bins()

for epoch in range(EPOCHS):
    # init the environment
    initial_state = env.reset()
    # grab the current state and discretize it
    discretized_state = discretize_observation_function(initial_state, BINS)
    # set done to false
    done = False
    # reset points
    points = 0
    # epoch takes + 1
    epochs.append(epoch)

    # while the game is not complete
    while not done :
        # ACTION
        action = epsilon_greedy_actions_selection(epsilon,Q_table,discretized_state)
        # OBSERVATION
        next_state,reward,done,info = env.step(action)
        # adding reward malus
        reward = fail(done,points,reward)
        # discrete state
        next_state_discretized = discretize_observation_function(next_state,BINS,)
        # grab the current Q value
        current_Q = Q_table[discretized_state + (action,)]
        # find the max Q value in the row
        next_optimal_Q = np.max(Q_table[next_state_discretized])
        # compute the next Q value
        next_Q = compute_next_Q(current_Q,reward,next_optimal_Q)
        # update the Q value
        Q_table[discretized_state + (action,)] = next_Q
        # the current state become the old one
        discretized_state = next_state_discretized
        # a point +1
        points += 1

    epsilon = reduce_epsilon(epsilon,epoch)

    
    if epoch % log_interval == 0:
        point_log.append(points)
        running_mean = round(np.mean(point_log[-30:]),2)
        mean_points_log.append(running_mean)
        print("Mean reward : " + str(running_mean))
        print("epoch : "+ str(100*(epoch/EPOCHS)) + " %")
        print(" ")


env.close()


############
#   TEST
############

observation = env.reset()
rewards = 0
for _ in range(1000):
    env.render()
    discrete_state = discretize_observation_function(observation, BINS)  # get bins
    action = np.argmax(Q_table[discrete_state])  # and chose action from the Q-Table
    observation, reward, done, info = env.step(action) # Finally perform the action
    rewards+=1
    if done:
        print(f"You got {rewards} points!")
        break
env.close()


