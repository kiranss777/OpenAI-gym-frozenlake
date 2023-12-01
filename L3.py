import gymnasium as gym
import numpy as np
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv, generate_random_map

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution.
    Each row specifies class probabilities.
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

class CustomFrozenLakeEnv(FrozenLakeEnv):
    def __init__(self, desc=None, map_size=8, num_holes=5, num_bonus=5, **kwargs):
        if desc is None:
            # Generate a random map with specified number of holes and bonuses
            desc = generate_random_map(size=map_size, p=0)  # Start with no holes or bonuses
            desc = self.add_random_elements(desc, 'H', num_holes)  # Add holes
            desc = self.add_random_elements(desc, 'B', num_bonus)  # Add bonuses

        super().__init__(desc=desc, **kwargs)
        self.bonus_tiles = set(self.locate_elements(desc, 'B'))
        self.used_bonus_tiles = set()

    def add_random_elements(self, desc, element, num):
        """ Add specified number of elements randomly to the grid. """
        positions = set()
        while len(positions) < num:
            pos = (np.random.randint(len(desc)), np.random.randint(len(desc[0])))
            if desc[pos[0]][pos[1]] == 'F':
                desc[pos[0]] = desc[pos[0]][:pos[1]] + element + desc[pos[0]][pos[1]+1:]
                positions.add(pos)
        return desc

    def locate_elements(self, desc, element):
        """ Find positions of certain elements in the grid. """
        return [(i, j) for i in range(len(desc)) for j in range(len(desc[i])) if desc[i][j] == element]

    def step(self, action):
        """ Modify the step function to handle bonus tiles. """
        transitions = self.P[self.s][action]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, done = transitions[i]
        
        # Handle bonus rewards
        if self.s in self.bonus_tiles and self.s not in self.used_bonus_tiles:
            r += 10  # Bonus reward
            self.used_bonus_tiles.add(self.s)

        self.s = s
        self.lastaction = action
        return (s, r, done, {"prob": p})

def run_episodes(env, num_episodes=10):
    rewards = []
    paths = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        path = [state]

        while not done:
            action = env.action_space.sample()  # Random action
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            if not done:
                path.append(next_state)

        rewards.append(total_reward)
        paths.append(path)
        print(f"Episode {episode + 1}: Reward = {total_reward}, Path = {path}")

    return rewards, paths

# Instantiate and run the custom environment
env = CustomFrozenLakeEnv()
rewards, paths = run_episodes(env)
