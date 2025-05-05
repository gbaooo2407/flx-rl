# agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import psutil
from config import *


def inject_noise(state, std=0.02):
    return state + np.random.normal(0, std, size=state.shape)


class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, HIDDEN_SIZE1),
            nn.LayerNorm(HIDDEN_SIZE1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_SIZE1, HIDDEN_SIZE2),
            nn.LayerNorm(HIDDEN_SIZE2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE2, HIDDEN_SIZE3),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE3, action_size)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.net(x)


class DQNAgent:
    def __init__(self, state_size, action_size, lr=1e-4, gamma=0.99, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.model = DQNNetwork(state_size, action_size)
        self.target_model = DQNNetwork(state_size, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.memory = []
        self.batch_size = BATCH_SIZE
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = 100
        self.train_count = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, valid_actions, env=None, eval_mode=False):
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if state.shape != (self.state_size,):
            state = state.reshape(self.state_size)

        if not valid_actions:
            return 0  # fallback action

        if eval_mode:
            self.model.eval()
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                q_values = self.model(state_tensor)[0]
                masked_q = q_values.clone()
                invalid_actions = set(range(self.action_size)) - set(valid_actions)
                for a in invalid_actions:
                    masked_q[a] = float('-inf')
                return torch.argmax(masked_q).item()
        else:
            if random.random() < self.epsilon:
                return random.choice(valid_actions)
            noisy_state = inject_noise(state)
            state_tensor = torch.from_numpy(noisy_state).float().unsqueeze(0)
            q_values = self.model(state_tensor)[0]
            masked_q = torch.full((self.action_size,), float('-inf'))
            for a in valid_actions:
                if a < len(q_values):
                    masked_q[a] = q_values[a]
            return torch.argmax(masked_q).item()

    def remember(self, s, a, r, s_, done):
        if r < -300:
            return  # bỏ sample tệ ngay từ đầu

        s = np.array(s, dtype=np.float32).reshape(self.state_size)
        s_ = np.array(s_, dtype=np.float32).reshape(self.state_size)

        self.memory.append((s, a, r, s_, done))

        # Nếu vượt quá kích thước bộ nhớ, loại bỏ phần tử cũ nhất
        if len(self.memory) > MEMORY_SIZE:
            self.memory.pop(0)

        # Optional: Cảnh báo khi bộ nhớ RAM vượt 80%
        mem_usage = psutil.Process().memory_info().rss / 1024**2
        total_mem = psutil.virtual_memory().total / 1024**2
        if mem_usage > 0.8 * total_mem:
            print(f"[WARNING] High memory usage: {mem_usage:.2f} MB ({mem_usage/total_mem:.2%} of system)")


    def train(self):
        if len(self.memory) < self.batch_size:
            return
        self.model.train()
        batch = random.sample(self.memory, self.batch_size)
        states = torch.from_numpy(np.array([b[0] for b in batch], dtype=np.float32))
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(1)
        next_states = torch.from_numpy(np.array([b[3] for b in batch], dtype=np.float32))
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32).unsqueeze(1)

        q_values = self.model(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.model(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_model(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.train_count += 1
        if self.train_count % self.target_update_freq == 0:
            self.update_target_model()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.update_target_model()