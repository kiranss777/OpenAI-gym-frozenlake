import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, render = False, is_training = True, slippery = True, map_name = "4x4"):
    env = gym.make("FrozenLake-v1", map_name = map_name, is_slippery=slippery, render_mode="human" if render else None)
    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))
#"D:/7TH SEM/RL/HACKATHON/frozenlake_4x4_slippery.pkl"
    else:
        file_name = "D:/7TH SEM/RL/HACKATHON//frozenlake_"+ map_name + ("_slippery" if slippery else "_not_slippery") +".pkl"
        with open(file_name, "rb") as f:
            q = pickle.load(f)
            
    # print(q)
    
    learning_rate = 0.95 #alpha
    discount_factor = 0.85 # gamma
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
                action = np.argmax(q[state,:])
            new_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if is_training:
                q[state, action] = q[state,action] + learning_rate * (
                    reward + discount_factor * np.max(q[new_state,:]) - q[state,action]
                )
            
            state = new_state
            
        epsilon = max(epsilon - epsilon_decay,0)
        
        if epsilon == 0:
            learning_rate = 0.0001
            
        if reward == 1:
            rewards_per_episode[i] = 1
    print("the total reward is ",total_reward)
    env.close()
    
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0,t-100):(t+1)])
        
    plt.plot(sum_rewards)
    file_name = "D:/7TH SEM/RL/HACKATHON/FILES_" + map_name + ("_slippery" if slippery else "_not_slippery") + ("_training" if is_training else "_testing")+".png"
    plt.savefig(file_name)
    file_name = "D:/7TH SEM/RL/HACKATHON/FILES_" + map_name + ("_slippery" if slippery else "_not_slippery") +".pkl"
    if is_training:
        with open(file_name, "wb") as f:
            pickle.dump(q,f)
if __name__ == "__main__":
    # run(1, is_training = False, slippery = False, render = True, map_name = "8x8")
    run(10000, is_training = True, slippery = True, render = False, map_name = "4x4")