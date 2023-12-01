"""import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

class CustomFrozenLakeEnv(FrozenLakeEnv):
    def __init__(self, desc=None, **kwargs):
        super().__init__(desc=desc, **kwargs)

def run(episodes, render=False, is_training=True, slippery=True, custom_map=None):
    if custom_map is not None:
        desc = np.asarray(custom_map, dtype='c')
        env = CustomFrozenLakeEnv(desc=desc, is_slippery=slippery, render_mode="human" if render else None)
    else:
        # default to the built-in 8x8 map if no custom map provided
        env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=slippery, render_mode="human" if render else None)

    q = np.zeros((env.observation_space.n, env.action_space.n)) if is_training else load_q_table()

    learning_rate = 0.001
    discount_factor = 0.85
    epsilon = 1
    epsilon_decay = 0.0001
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)
    
    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        total_reward = 0
        
        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])
            
            new_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if is_training:
                q[state, action] = q[state, action] + learning_rate * (
                    reward + discount_factor * np.max(q[new_state, :]) - q[state, action]
                )
            state = new_state
        
        epsilon = max(epsilon - epsilon_decay, 0)
        if epsilon == 0:
            learning_rate = 0.0001
        if reward == 1:
            rewards_per_episode[i] = 1
    
    env.close()
    
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
        
    plt.plot(sum_rewards)
    plt.show()

    if is_training:
        save_q_table(q)

def save_q_table(q_table):
    file_name = "frozenlake_custom.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(q_table, f)

def load_q_table():
    file_name = "frozenlake_custom.pkl"
    with open(file_name, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    custom_map = [
        "SFFFFFFF",
        "FFFFFFFH",
        "FFFHFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FHFFFFHF",
        "FFFFHFHF",
        "FFFHFFFG",
    ]
    run(10000, is_training=False, slippery=True, render=False, custom_map=custom_map)

"""
#LEVEL 2 WORKS
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

class CustomFrozenLakeEnv(FrozenLakeEnv):
    def __init__(self, desc=None, **kwargs):
        super().__init__(desc=desc, **kwargs)

def run(episodes, render=False, is_training=True, slippery=True, custom_map=None):
    if custom_map is not None:
        desc = np.asarray(custom_map, dtype='c')
        env = CustomFrozenLakeEnv(desc=desc, is_slippery=slippery, render_mode="human" if render else None)
    else:
        env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=slippery, render_mode="human" if render else None)

    q = np.zeros((env.observation_space.n, env.action_space.n)) if is_training else load_q_table()

    learning_rate = 0.9
    discount_factor = 0.85
    epsilon = 1
    epsilon_decay = 0.0001
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        total_reward = 0
        path = [state]

        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            path.append(new_state)
            
            if is_training:
                q[state, action] = q[state, action] + learning_rate * (
                    reward + discount_factor * np.max(q[new_state, :]) - q[state, action]
                )
            state = new_state

        epsilon = max(epsilon - epsilon_decay, 0)
        if epsilon == 0:
            learning_rate = 0.0001
        rewards_per_episode[i] = total_reward

        # Display episode results
        print(f"Episode {i+1}/{episodes}:")
        print(f"  Path: {path}")
        print(f"  Goal reached: {'Yes' if terminated and reward > 0 else 'No'}")
        print(f"  Total Reward: {total_reward}\n")

    env.close()

    if is_training:
        save_q_table(q)

    # Plotting the results
    plt.plot(rewards_per_episode)
    plt.title("Rewards per Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.savefig("D:/7TH SEM/RL/HACKATHON")
    #plt.show()

def save_q_table(q_table):
    file_name = "frozenlake_custom.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(q_table, f)

def load_q_table():
    file_name = "frozenlake_custom.pkl"
    with open(file_name, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    custom_map = [
        "SFFFFFFF",
        "FFFFFFFH",
        "FFFHFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FHFFFFHF",
        "FFFFHFHF",
        "FFFHFFFG",
    ]
    run(10000, is_training=False, slippery=True, render=False, custom_map=custom_map)
