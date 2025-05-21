# demo_flxrl.py
# A minimized FLX-RL demo using a synthetic graph with logic aligned to the real project

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from agent import DQNAgent, DQNNetwork
from env import OSMGraphEnv
from federated import federated_averaging
from utils import sample_start_goal
from train import train_cycle, evaluate_agent
import os
import random
from config import *

# ======= Step 1: Create a synthetic graph that mimics OSM =======
def create_mini_city_graph(num_nodes=100, seed=42):
    np.random.seed(seed)
    G = nx.random_geometric_graph(num_nodes, radius=0.2)
    for u, v in G.edges:
        dist = np.linalg.norm(np.array(G.nodes[u]['pos']) - np.array(G.nodes[v]['pos']))
        G.edges[u, v]['length'] = dist * 1000  # convert to meters
    for i in G.nodes:
        G.nodes[i]['x'], G.nodes[i]['y'] = G.nodes[i]['pos']
    return G

# ======= Step 2: Set up environment and parameters =======
def main():
    G = create_mini_city_graph()
    global_bounds = (0.0, 1.0, 0.0, 1.0)

    agents, perfs = [], []
    for agent_idx in range(3):
        start, goal = sample_start_goal(G, min_dist=0.2, max_dist=1.0, force_far=True)
        env = OSMGraphEnv(G, start, goal, global_bounds)
        agent = DQNAgent(state_size=STATE_SIZE, action_size=env.action_space.n,epsilon_decay=0.95)
        
        print(f"[TRAIN] Agent {agent_idx} | Start: {start} → Goal: {goal}")
        train_cycle(env, agent, episodes=300, max_steps=150, cycle_num=1, graph=G, agent_idx=agent_idx)

        agents.append(agent.model)
        perf, success, _ = evaluate_agent(agent, env, episodes=5, max_steps=100)
        perfs.append(perf)
        print(f"[EVAL] Agent {agent_idx} | Perf: {perf:.2f} | Success: {success:.2f}")

    # ======= Step 3: Federated Averaging =======
    weights = (np.array(perfs) - np.min(perfs)) / (np.ptp(perfs) + 1e-10)
    weights /= weights.sum()
    global_model = DQNNetwork(state_size=env.state_size, action_size=env.action_space.n)
    global_model = federated_averaging(global_model, agents, weights)

    # ======= Step 4: Deploy global model on unseen start-goal =======
    start, goal = sample_start_goal(G, min_dist=0.3, max_dist=1.2)
    env_final = OSMGraphEnv(G, start, goal, global_bounds)
    global_agent = DQNAgent(state_size=env.state_size, action_size=env.action_space.n)
    global_agent.model.load_state_dict(global_model.state_dict())
    global_agent.update_target_model()
    
    print("\n[FINAL EVAL] Using global model")
    evaluate_agent(global_agent, env_final, episodes=3, max_steps=100)

if __name__ == "__main__":
    os.makedirs("training_progress", exist_ok=True)
    os.makedirs("training_progress/paths", exist_ok=True)
    main()
