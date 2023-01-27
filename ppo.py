# Use PPO, HomeOdd, DrawOdd, AwayOd, History of last 5 matches for each team and 2 matches head to head to build profitable betting strategy

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data.csv')
df = df.dropna()

# Get features and labels
X = df[['HomeOdd', 'DrawOdd', 'AwayOdd', 'HomeLast5', 'DrawLast5', 'AwayLast5', 'HomeLast2', 'DrawLast2', 'AwayLast2']]
y = df['Result']

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float)
X_test = torch.tensor(X_test, dtype=torch.float)
y_train = torch.tensor(y_train.values, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.long)

# Define policy network
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 3)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_scores = self.fc2(x)
        return F.softmax(action_scores, dim=1)

# Define PPO agent
class PPO:
    def __init__(self):
        self.policy = Policy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.eps = np.finfo(np.float32).eps.item()

        self.epochs = 10
        self.clip_param = 0.2
        self.max_grad_norm = 0.5
        self.ppo_update_time = 10
        self.buffer_capacity, self.batch_size = 32, 32
        self.data_buffer = []

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.policy.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def get_reward(self, state, action):
        if action == 0:
            return state[0]
        elif action == 1:
            return state[1]
        else:
            return state[2]

    def store_data(self, state, action, reward):
        self.data_buffer.append((state, action, reward))

    def ppo_iter(self, mini_batch_size):
        state_batch, action_batch, reward_batch = [], [], []
        for state, action, reward in self.data_buffer:
            state_batch.append(state)
            action_batch.append([action])
            reward_batch.append([reward])

        state_batch = torch.tensor(state_batch, dtype=torch.float)
        action_batch = torch.tensor(action_batch)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float)

        for i in range(state_batch.size(0) // mini_batch_size):
            ind = np.random.randint(0, state_batch.size(0), mini_batch_size)
            yield state_batch[ind], action_batch[ind], reward_batch[ind]

    def ppo_update(self, i_epoch):
        state_batch, action_batch, reward_batch = [], [], []
        for state, action, reward in self.data_buffer:
            state_batch.append(state)
            action_batch.append([action])
            reward_batch.append([reward])

        state_batch = torch.tensor(state_batch, dtype=torch.float)
        action_batch = torch.tensor(action_batch)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float)

        for _ in range(self.epochs):
            for state, action, reward in self.ppo_iter(self.batch_size):
                probs = self.policy(state)
                
                m = Categorical(probs)
                action_probs = m.log_prob(action)
                dist_entropy = m.entropy().mean()

                old_action_probs = action_probs.detach()
                ratio = torch.exp(action_probs - old_action_probs)

                surr1 = ratio * reward
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * reward
                action_loss = -torch.min(surr1, surr2).mean()

                self.optimizer.zero_grad()
                (action_loss - dist_entropy * 0.01).backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def clear_buffer(self):
        self.data_buffer.clear()

# Train PPO agent
agent = PPO()
all_rewards = []
avg_rewards = []
for i_episode in range(1000):
    state = X_train[i_episode].numpy()
    for t in range(1000):
        action = agent.select_action(state)
        reward = agent.get_reward(state, action)
        agent.store_data(state, action, reward)

        if t % agent.ppo_update_time == 0:
            agent.ppo_update(i_episode)
            agent.clear_buffer()

        if t == 999:
            all_rewards.append(reward)
            avg_rewards.append(np.mean(all_rewards[-10:]))

            if i_episode % 100 == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(i_episode, reward, np.mean(all_rewards[-10:])))
            break

# Plot rewards
plt.plot(avg_rewards)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.show()

# Test PPO agent
agent.policy.eval()
correct = 0
with torch.no_grad():
    for i in range(len(X_test)):
        state = X_test[i].numpy()
        action = agent.select_action(state)
        if action == y_test[i]:
            correct += 1

print('Accuracy: {}/{} ({:.0f}%)'.format(correct, len(X_test), 100. * correct / len(X_test)))