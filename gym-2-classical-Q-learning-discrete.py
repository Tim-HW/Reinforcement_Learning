
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


def reduce_epsilon(epsilon,epoch):

    return min_epsilon + (max_epsilon-min_epsilon)*np.exp(-decay_rate*epoch)

def compute_next_Q(old_Q,reward,next_optimal_Q):

    return old_Q + ALPHA  * (reward + GAMMA * next_optimal_Q - old_Q)

def epsilon_greedy_actions_selection(epsilon,Q_table,discrete_state):

    # random number [0 and 1]
    random_number = np.random.random()
    # Exploitation (choose the action that maximize Q)
    if random_number > epsilon:
        # retrieve Q data from that state
        state_row = Q_table[discrete_state,:]
        # choose the max Q action in this row
        action = np.argmax(state_row)

    # Exploration (choose random action)
    else :
        action = env.action_space.sample()

    return action



########################################
epoch_plot_tracker = []
total_reward_plot_tracker = []
########################################


# choose env
env = gym.make('FrozenLake-v1', is_slippery=False)
# reset env (just in case)
observation = env.reset()

# Column => actions
action_size = env.action_space.n
# Row => states
state_size  = env.observation_space.n
# Init our Q-Table
Q_table = np.zeros([state_size,action_size])

rewards = []

for episode in range(EPOCHS):

    state = env.reset()
    done = False
    total_reward = 0

    # agent plays the game
    while not done :

        # ACTION
        action = epsilon_greedy_actions_selection(epsilon,Q_table,state)
        # OBSERVATION
        new_state,reward,done,info = env.step(action)
        # Get current Q value
        current_Q = Q_table[state,action] 
        # Get next Optimal Q Value
        next_optimal_Q = np.max(Q_table[new_state,:])
        # Compute next Q value
        next_Q = compute_next_Q(current_Q,reward,next_optimal_Q)
        # update Table
        Q_table[state,action] = next_Q
        # Track reward
        total_reward += reward
        # state <= new state
        state = new_state


    # agent finished a round of the game
    episode += 1
    epsilon = reduce_epsilon(epsilon,episode)
    rewards.append(total_reward)

    total_reward_plot_tracker.append(np.sum(rewards))
    epoch_plot_tracker.append(episode)
    ####################################
    if episode % log_interval == 0:

        plt.scatter(total_reward_plot_tracker, epoch_plot_tracker)
        plt.title("Real Time plot")
        plt.xlabel("x")
        plt.ylabel("sinx")
        plt.pause(0.05)

plt.show()


# plays the game in GOD mod
state = env.reset()

for steps in range(100):

    env.render()
    action = np.argmax(Q_table[state,:])
    state,reward,done,info = env.step(action)

    if done :
        break
env.close()
