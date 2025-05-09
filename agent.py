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


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.alpha = alpha

    def push(self, transition, td_error=1.0):
        max_prio = max(self.priorities.max(), abs(td_error))
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], [], []

        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, torch.tensor(weights, dtype=torch.float32)


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
    def __init__(self, state_size, action_size, lr=1e-4, gamma=0.99, epsilon_start=0.8, epsilon_end=0.05, epsilon_decay=0.9985):
        self.state_size = state_size
        self.action_size = action_size
        self.model = DQNNetwork(state_size, action_size)
        self.target_model = DQNNetwork(state_size, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.memory = PrioritizedReplayBuffer(MEMORY_SIZE)
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
            return 0

        if eval_mode:
            self.model.eval()
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                q_values = self.model(state_tensor)[0]
                masked_q = q_values.clone()
                invalid_actions = set(range(self.action_size)) - set(valid_actions)
                for a in invalid_actions:
                    masked_q[a] = float('-inf')
                if env:
                    neighbors = list(env.neighbor_fn(env.current_node))
                    penalties = [env.visited_nodes.count(node) for node in neighbors]
                    for i, a in enumerate(valid_actions):
                        masked_q[a] -= penalties[i] * 5.0
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
        if r < -1000:
            return

        s = np.array(s, dtype=np.float32).reshape(self.state_size)
        s_ = np.array(s_, dtype=np.float32).reshape(self.state_size)
        self.memory.push((s, a, r, s_, done), td_error=1.0)  # placeholder nhẹ

        mem_usage = psutil.Process().memory_info().rss / 1024**2
        total_mem = psutil.virtual_memory().total / 1024**2
        if mem_usage > 0.8 * total_mem:
            print(f"[WARNING] High memory usage: {mem_usage:.2f} MB ({mem_usage/total_mem:.2%} of system)")

    def train(self):
        if len(self.memory.buffer) < self.batch_size:
            return

        self.model.train()
        batch, indices, weights = self.memory.sample(self.batch_size)

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
            target_q = torch.clamp(target_q, -1000, 1000)

        loss_fn = nn.SmoothL1Loss(reduction='none')
        losses = loss_fn(q_values, target_q)
        weighted_loss = (losses * weights.unsqueeze(1)).mean()

        # ✅ Cập nhật td-error vào buffer
        with torch.no_grad():
            td_errors = (q_values - target_q).squeeze().abs().cpu().numpy()
            for i, idx in enumerate(indices):
                self.memory.priorities[idx] = td_errors[i] + 1e-4  # tránh 0

        # Backprop
        self.optimizer.zero_grad()
        weighted_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.train_count += 1
        if self.train_count % self.target_update_freq == 0:
            self.update_target_model()

        if self.train_count % 20 == 0:
            print(f"[TRAIN] Step={self.train_count} | Loss={weighted_loss.item():.4f} | ε={self.epsilon:.3f} | Memory={len(self.memory.buffer)}")

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.update_target_model()
