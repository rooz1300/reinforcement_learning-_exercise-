import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau



import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 
import time
from datetime import datetime
import imageio

os.system("cls")



def save_episode_as_gif(policy_net, env_name, gif_path="episode.gif", max_steps=2500, fps=30):
    """
    Run an episode using the given policy network and save it as a GIF.
    
    Args:
        policy_net: PyTorch model that outputs action probabilities.
        env_name: Gym environment name (e.g., 'CartPole-v1').
        gif_path: Path to save the output GIF.
        max_steps: Maximum number of steps to run.
        fps: Frames per second for the GIF.
    """
    # Create environment with rgb_array render mode for frame capture
    eval_env = gym.make(env_name, render_mode="rgb_array")
    state, _ = eval_env.reset()
    
    frames = []  # To store rendered frames

    for step in range(max_steps):
        # Capture current frame
        frame = eval_env.render()
        frames.append(frame)
        
        # Get action from policy network
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs = policy_net(state_tensor)
            action = torch.argmax(action_probs).item()
        
        # Take step in environment
        state, reward, terminated, truncated, _ = eval_env.step(action)
        
        if terminated or truncated:
            print(f"Episode finished at step {step + 1}!")
            # Capture final frame
            frame = eval_env.render()
            frames.append(frame)
            break

    eval_env.close()

    # Save frames as GIF
    print(f"Saving GIF to {gif_path}...")
    imageio.mimsave(gif_path, frames, fps=fps)
    print("GIF saved successfully!")
    
def plot_results(rewards_history, save_path=None, moving_avg_window=100):
    """
    Generates and saves a publication-quality plot of training rewards.

    Args:
        rewards_history (list): A list of total rewards for each episode.
        save_path (str, optional): Path to save the plot image. If None, shows the plot.
        moving_avg_window (int): The window size for the moving average.
    """
    # Use a professional style for the plot
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate the moving average of the rewards
    rewards_series = pd.Series(rewards_history)
    moving_avg = rewards_series.rolling(window=moving_avg_window, min_periods=1).mean()
    
    # Plot the raw rewards with low opacity
    ax.plot(rewards_series, alpha=0.3, color='gray', label='Raw Episode Score')
    
    # Plot the moving average
    ax.plot(moving_avg, color='blue', linewidth=2, label=f'{moving_avg_window}-Episode Moving Average')
    
    # Add a horizontal line for the "solved" criteria for CartPole-v1
    # The environment is considered solved if avg reward is >= 475 over 100 consecutive trials.
    ax.axhline(y=475, color='red', linestyle='--', linewidth=2, label='Solved Threshold (475)')
    
    # Set titles and labels with appropriate font sizes
    ax.set_title('REINFORCE Performance on CartPole-v1', fontsize=20, weight='bold')
    ax.set_xlabel('Episode', fontsize=16)
    ax.set_ylabel('Score (Total Reward)', fontsize=16)
    
    # Customize ticks and legend
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=14)
    
    # Adjust layout and save/show the plot
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300) # High resolution for publications
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def save_checkpoint(model, optimizer, rewards_history, save_dir='checkpoints'):
    """
    Saves the model state, optimizer state, and rewards history to a file.

    Args:
        model (nn.Module): The PyTorch model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        rewards_history (list): A list of rewards from training.
        save_dir (str): The directory to save the checkpoint in.
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate a unique ID for the training run using the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Define the file path
    file_path = os.path.join(save_dir, f'checkpoint_id_{timestamp}.pth')
    
    # Create a dictionary with all the necessary information
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rewards_history': rewards_history,
        'timestamp_id': timestamp
    }
    
    # Save the checkpoint dictionary
    torch.save(checkpoint, file_path)
    print(f"✅ Checkpoint and training data saved to: {file_path}")
    return file_path














class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        # Initialize the PolicyNetwork class which inherits from nn.Module
        super(PolicyNetwork, self).__init__()
        # Define the neural network architecture using nn.Sequential
        self.network = nn.Sequential(
            # First fully connected layer that maps state_size inputs to 128 outputs
            nn.Linear(state_size, 128),
            # ReLU activation function for non-linearity
            nn.ReLU(),
            # Second fully connected layer that maps 128 inputs to action_size outputs
            nn.Linear(128, action_size),
            # Softmax activation function to convert outputs to probabilities
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        # Define the forward pass of the network
        # Input: state - the input state to the policy network
        # Output: probability distribution over actions
        return self.network(state)




# Hyperparameters
env_name = "CartPole-v1"
learning_rate = 0.01
gamma = 0.99
num_episodes = 5000
# Initialization
env = gym.make(env_name)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

policy_net = PolicyNetwork(state_size, action_size)
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.85, patience=10)

print(f"State size: {state_size}")
print(f"Action size: {action_size}")



# Select device: GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device= "cpu"
print(f"Using device: {device}")

# Move policy network to the selected device BEFORE creating the optimizer

policy_net = policy_net.to(device)

# Move policy network to the selected device BEFORE creating the optimizer
policy_net = policy_net.to(device)


# Main training loop
all_rewards = []
progress_bar = tqdm(range(num_episodes), desc="Training Progress")















# Loop for each episode, collecting a trajectory τ for each one.
for episode in progress_bar:
    # Get the initial state s_0 from the environment.
    state, _ = env.reset()
    # Initialize lists to store log probabilities and rewards for the current trajectory.
    log_probs = []   # Will store log π_θ(a_t|s_t) for t=0, 1, ...
    rewards = []     # Will store r_t for t=0, 1, ...
    entropies = []   # Will store entropy of the action distribution (for exploration bonus)
    
    # 1. Play an episode and collect a trajectory τ = (s_0, a_0, r_0, s_1, a_1, r_1, ...)
    while True:
        # Convert the current state s_t into a PyTorch tensor.
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        # Get the action probability distribution from the policy network.
        # This computes π_θ(a|s_t) for all possible actions 'a'.
        action_probs = policy_net(state_tensor)
        
        # Create a categorical distribution object from the action probabilities.
        dist = Categorical(action_probs)
        # Sample an action a_t from the distribution.
        # This is the step: a_t ~ π_θ(·|s_t)
        action = dist.sample()
        
        # Calculate and store the log-probability of the chosen action.
        # This computes and stores log π_θ(a_t|s_t).
        log_probs.append(dist.log_prob(action))
        # Store the entropy of the distribution (encourages exploration).
        entropies.append(dist.entropy())
        
        # Execute action a_t in the environment to get the next state s_{t+1} and reward r_t.
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        # Store the received reward r_t.
        rewards.append(reward)
        # Update the state for the next timestep: s_t ← s_{t+1}.
        state = next_state
        
        # End the episode if the state is terminal.
        if terminated or truncated:
            break
            
    # Store the total undiscounted reward for this episode, for monitoring purposes.
    all_rewards.append(sum(rewards))

    # 2. Calculate discounted returns (rewards-to-go) for each timestep t.
    returns = []
    discounted_reward = 0 # This will be G_t at each step of the loop.
    
    # Iterate backwards through the rewards list from t = T-1 down to 0.
    for r in reversed(rewards):
        # Calculate the return G_t = r_t + γ * r_{t+1} + γ^2 * r_{t+2} + ...
        # This is efficiently calculated using the recursive formula: G_t = r_t + γ * G_{t+1}
        # where we start with G_T = 0.
        discounted_reward = r + gamma * discounted_reward
        returns.insert(0, discounted_reward)
        
    # Convert the list of returns G_t into a tensor.
    returns = torch.tensor(returns, device=device)
    # Subtract the mean as a baseline (reduces variance and stabilizes training).
    # This is less destructive than dividing by std per episode.
    returns = returns - returns.mean()
    
    # 3. Calculate the loss and update the policy parameters θ.
    policy_loss = []
    # Loop through the log probabilities and their corresponding baseline-adjusted returns.
    for log_prob, R in zip(log_probs, returns):
        # The REINFORCE objective is to maximize J(θ) = E[G_t].
        # The policy gradient is ∇_θ J(θ) = E[G_t * ∇_θ log π_θ(a_t|s_t)].
        # To perform gradient ascent, we use gradient descent on the negative objective.
        # We calculate the loss for each timestep as -Ĝ_t * log π_θ(a_t|s_t).
        policy_loss.append(-log_prob * R)
        
    # Reset the gradients to zero before backpropagation.
    optimizer.zero_grad()
    # Sum the loss terms for all timesteps in the episode to get the final policy loss.
    # L(θ) = - Σ_{t=0}^{T-1} Ĝ_t * log π_θ(a_t|s_t)
    loss = torch.stack(policy_loss).sum()
    
    # Add an entropy bonus to the loss to encourage exploration.
    # This prevents the policy from becoming too deterministic too early.
    entropy_bonus = torch.stack(entropies).sum()
    loss = loss - 0.01 * entropy_bonus
    
    # Compute the gradient of the loss with respect to the policy network parameters.
    # This calculates ∇_θ L(θ).
    loss.backward()
    # Clip gradients to avoid exploding gradients (improves stability).
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    # Update the policy network's parameters using the optimizer (e.g., Adam).
    # This performs the update: θ ← θ - α * ∇_θ L(θ), which is equivalent to
    # θ ← θ + α * ∇_θ J(θ) for our objective J.
    optimizer.step()
    
    # Update the progress bar with the average reward of the last 100 episodes for monitoring.
    if episode % 10 == 0:
        avg_reward = np.mean(all_rewards[-100:])
        new_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix(
        avg_reward=f'{avg_reward:.2f}',
        lr=f'{new_lr:.6f}')
        scheduler.step(avg_reward)  # <-- Add this line













# 1. Save the final model and all training data
saved_checkpoint_path = save_checkpoint(
    model=policy_net,
    optimizer=optimizer,
    rewards_history=all_rewards
)

# 2. Plot the results and save the figure
# We can use the unique ID from the checkpoint to name the plot
plot_id = os.path.basename(saved_checkpoint_path).split('_based_lined.')[0]
plot_results(
    rewards_history=all_rewards,
    save_path=f'checkpoints/{plot_id}_based.png'
)
save_episode_as_gif(
    policy_net=policy_net,
    env_name="CartPole-v1",
    gif_path="cartpole_episode_with_baseline.gif",
    max_steps=500,
    fps=30
)