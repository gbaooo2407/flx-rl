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


def knowledge_distillation(global_model, local_models, graph, global_bounds,
                                     episodes=200, batch_size=32, n_buffer_episodes=30):

    from env import OSMGraphEnv
    from utils import sample_start_goal

    student = global_model
    teacher_ensemble = [model.eval() for model in local_models]
    optimizer = optim.Adam(student.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()

    # Step 1: Chọn teacher tốt nhất để tạo replay buffer
    best_teacher = teacher_ensemble[0]
    env_buffer = []

    for _ in range(n_buffer_episodes):
        s, g = sample_start_goal(graph, min_dist=1000, max_dist=6000)
        env = OSMGraphEnv(graph, s, g, global_bounds)
        state = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                q_values = best_teacher(torch.FloatTensor(state).unsqueeze(0))
                action = torch.argmax(q_values).item()
            next_state, _, done, _ = env.step(action)
            env_buffer.append(state)
            state = next_state
            if len(env_buffer) >= 5000:
                break
        if len(env_buffer) >= 5000:
            break

    if len(env_buffer) < batch_size:
        print("[Distill] ⚠️ Replay buffer quá nhỏ.")
        return student

    # Step 2: Train student model để bắt chước teacher ensemble
    for ep in range(episodes):
        batch_states = random.sample(env_buffer, batch_size)
        state_tensor = torch.FloatTensor(batch_states)
        with torch.no_grad():
            teacher_outputs = torch.stack([t(state_tensor) for t in teacher_ensemble]).mean(dim=0)
        student_outputs = student(state_tensor)
        loss = loss_fn(student_outputs, teacher_outputs)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        if (ep + 1) % 10 == 0:
            print(f"[Distill-Advanced] Ep {ep+1}/{episodes} | Loss: {loss.item():.4f}")

    return student

