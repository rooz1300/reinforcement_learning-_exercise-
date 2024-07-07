import gym
from tqdm import tqdm
import time

# Create the FrozenLake environment with RGB_ARRAY rendering mode
env = gym.make("FrozenLake-v1",render_mode="human")

# Reset the environment to get the initial state
env.reset()
env.render()
episodes = 1000

# Wrap your range(episodes) with tqdm for a progress bar
for i in tqdm(range(episodes), desc='Running Episodes'):
    env.reset()
    while True:
        # Sample an action randomly
        action = env.action_space.sample()
        # Print the action
        print(f'Episode {i}, Action: {action}')
        # Execute the action and observe the new state and reward
        next_state, reward, done, info,_ = env.step(action)
        # print(next_state, reward, done, info)
        if done: break 
        

# Close the environment after running all the episodes
env.close()
