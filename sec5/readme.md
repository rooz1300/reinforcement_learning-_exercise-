# Blackjack Environment and On-Policy Monte Carlo Control

This Jupyter notebook demonstrates the use of the On-Policy First-Visit Monte Carlo Control algorithm to find the optimal policy for the Blackjack environment. The algorithm is implemented using the OpenAI Gym library.

## Importing Libraries

We start by importing the necessary libraries.

```python
import numpy as np
import gym
from collections import defaultdict
import pandas as pd
```

## Environment and Hyperparameters

We define the discount factor for future rewards and the number of possible actions in the environment.

```python
# Discount factor for future rewards
gamma = 1
# Number of possible actions in the environment
number_actions = 2
```

We create the Blackjack environment using the Gym library.

```python
# Create the Blackjack environment
env = gym.make("Blackjack-v1")
```

## Helper Functions

### `argmax_radd`

This function returns a random index from the array that has the maximum value.

```python
def argmax_radd(arr):
    temp= np.random.choice(
                np.flatnonzero(
                    arr==np.max(arr)
            )
    )
    return temp
```

### `generate_episode`

This function generates an episode using the given policy and environment. It returns the generated episode and the length of the episode.

```python
def generate_episode(policy, env, pi):
    """
    Generate an episode using the given policy and environment.

    Args:
        policy (function): Policy function to determine actions.
        env (gym.Env): Gym environment to interact with.
        pi (dict): Policy table.

    Returns:
        tuple: The generated episode and the length of the episode.
    """
    # ...
```

### `on_policy_cn_control`

This function implements the On-Policy First-Visit Monte Carlo Control algorithm to find the optimal policy. It returns the action-value function Q.

```python
def on_policy_cn_control(env, ep, gamma, eps):
    """
    On-policy first-visit Monte Carlo control algorithm to find the optimal policy.

    Args:
        env (gym.Env): Gym environment to interact with.
        ep (int): Number of episodes to run.
        gamma (float): Discount factor for future rewards.
        eps (float): Epsilon for exploration in epsilon-greedy policy.

    Returns:
        dict: The action-value function Q.
    """
    # ...
```

### `policy`

This function implements the epsilon-greedy policy to choose actions.

```python
def policy(st, pi):
    """
    Epsilon-greedy policy to choose actions.

    Args:
        st (tuple): Current state.
        pi (dict): Policy table.

    Returns:
        int: Chosen action.
    """
    # ...
```

## Running the Algorithm

We run the on-policy control algorithm for 500,000 episodes and print the resulting action-value function Q.

```python
# Run the on-policy control algorithm
Q,pi = on_policy_cn_control(env, ep=500000, gamma=gamma, eps=0.1)

# Print the resulting action-value function
print(Q)
print(pi)
```

The output of the action-value function Q and the policy pi will provide insights into the optimal strategy for playing Blackjack.

