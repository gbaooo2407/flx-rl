# federated.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from config import *


def federated_averaging(global_model, local_models, weights=None):
    if weights is None:
        weights = [1.0 / len(local_models)] * len(local_models)
    else:
        weights = np.array(weights, dtype=np.float32)
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-10)
        weights /= weights.sum()

    global_dict = global_model.state_dict()
    for key in global_dict:
        global_dict[key] = sum(weights[i] * local_models[i].state_dict()[key].float() for i in range(len(local_models)))
    global_model.load_state_dict(global_dict)
    return global_model


def knowledge_distillation(global_model, local_models, env, episodes=200, batch_size=32):
    student = global_model
    teacher_ensemble = [model.eval() for model in local_models]
    optimizer = optim.Adam(student.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()

    replay_buffer = []
    for _ in range(100):
        state = env.reset()
        done = False
        while not done:
            neighbors = list(env.neighbor_fn(env.current_node))
            if not neighbors:
                break
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = student(state_tensor).argmax().item()
            next_state, _, done, _ = env.step(action)
            replay_buffer.append(state)
            state = next_state
            if len(replay_buffer) > 5000:
                break
    if len(replay_buffer) < batch_size:
        print(" Replay buffer too small for distillation")
        return student

    for ep in range(episodes):
        batch_states = random.sample(replay_buffer, batch_size)
        state_tensor = torch.FloatTensor(batch_states)
        with torch.no_grad():
            teacher_outputs = torch.stack([teacher(state_tensor) for teacher in teacher_ensemble]).mean(dim=0)
        student_output = student(state_tensor)
        loss = loss_fn(student_output, teacher_outputs)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        if (ep + 1) % 10 == 0:
            print(f"Distill [{ep+1}/{episodes}] - Loss: {loss.item():.4f}")
    return student
