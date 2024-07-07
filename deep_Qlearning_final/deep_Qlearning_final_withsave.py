import os
import random
import keras
import tensorflow as tf
from keras import models, layers
from keras.optimizers import Adam
import gym
import numpy as np
import pandas as pd
from collections import deque
from tqdm import tqdm

M = 250  # number of episodes
T = 210  # number of iterations of inner loop
batch_size = 24

# Initialize the environment
env = gym.make("CartPole-v1")

class DQN:
    def __init__(self, states, actions, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
        # Initialize hyperparameters and other variables
        self.gamma = gamma
        self.memory = deque([], maxlen=2500)
        self.ns = states
        self.nA = actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()
        self.loss = []

    def build_model(self):
        # Build a neural network model
        model = models.Sequential([
            layers.Dense(24, activation="relu", input_dim=self.ns),
            layers.Dense(24, activation="relu"),
            layers.Dense(self.nA, activation="linear")
        ])
        model.compile(optimizer=Adam(learning_rate=self.alpha), loss="mse")
        return model

    def epsilon_greedy(self, state):
        # Choose an action using epsilon-greedy strategy
        if random.random() <= self.epsilon:
            return env.action_space.sample()
        else:
            action_values = self.model.predict(state, verbose=0)[0]
            return np.argmax(action_values)

    def store(self, state, action, reward, next_state, done):
        # Store experience in replay memory
        self.memory.append((state, action, reward, next_state, done))

    def train_net(self):
        # Train the model using a mini-batch of experiences
        mini_batch = random.sample(self.memory, batch_size)
        states = np.vstack([x[0] for x in mini_batch])
        actions = np.array([x[1] for x in mini_batch])
        rewards = np.array([x[2] for x in mini_batch])
        next_states = np.vstack([x[3] for x in mini_batch])
        dones = np.array([x[4] for x in mini_batch])

        st_pre = self.model.predict(states, verbose=0)
        nst_pre = self.model.predict(next_states, verbose=0)

        x = []
        y = []

        for index in range(len(mini_batch)):
            state, action, reward, next_state, done = mini_batch[index]
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(nst_pre[index])

            target_train = st_pre[index]
            target_train[action] = target
            x.append(state)
            y.append(target_train)

        x = np.array(x).reshape(batch_size, self.ns)
        y = np.array(y)

        hist = self.model.fit(x, y, epochs=1, verbose=0)
        self.loss.append(hist.history['loss'][0])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    ns = env.observation_space.shape[0]
    nA = env.action_space.n
    dqn = DQN(states=ns, actions=nA, alpha=0.001, gamma=0.95, epsilon=1, epsilon_min=0.001, epsilon_decay=0.995)

    # Create a DataFrame to store results
    results = pd.DataFrame(columns=['Episode', 'Score', 'Epsilon', 'Loss'])

    # Main training loop
    for episode in tqdm(range(M)):
        state, info = env.reset()
        state = np.reshape(state, (1, -1))
        total_reward = 0
        for step_time in range(T):
            action = dqn.epsilon_greedy(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, (1, -1))
            total_reward += reward
            dqn.store(state, action, reward, next_state, done)
            if len(dqn.memory) > batch_size:
                dqn.train_net()
            state = next_state
            if done or step_time == 209:
                # Log results for the current episode
                new_row = pd.DataFrame([{
                    'Episode': episode,
                    'Score': total_reward,
                    'Epsilon': dqn.epsilon,
                    'Loss': dqn.loss[-1] if dqn.loss else None
                }])
                results = pd.concat([results, new_row], ignore_index=True)
                print(f"Number of episode {episode}, Score: {total_reward}, Epsilon: {dqn.epsilon}")
                break

        # Save the model every 25 episodes
        if (episode + 1) % 25 == 0:
            model_filename = f'dqn_model_episode_{episode + 1}.h5'
            dqn.model.save(model_filename)
            print(f"Model saved as {model_filename}")
            results.to_csv(f'dqn_training_results_{episode + 1}.csv', index=False)

    # Save results to a CSV file
    results.to_csv('dqn_training_results.csv', index=False)

    # Save the final model
    dqn.model.save('dqn_model.h5')
