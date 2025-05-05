# main.py

import os
import matplotlib.pyplot as plt
from config import *
from env import OSMGraphEnv
from agent import DQNAgent
from train import train_cycle, evaluate_agent
from federated import federated_averaging, knowledge_distillation
from utils import load_graph, sample_start_goal
import numpy as np
import networkx as nx
import pickle

def main():
    os.makedirs("training_progress", exist_ok=True)
    os.makedirs("training_progress/paths", exist_ok=True)
    plt.ioff()

    place_names = [
        'District 1, Ho Chi Minh City, Vietnam',
        'District 3, Ho Chi Minh City, Vietnam',
        'District 4, Ho Chi Minh City, Vietnam',
    ]
    eval_districts = [
    'District 8, Ho Chi Minh City, Vietnam',
    'District 10, Ho Chi Minh City, Vietnam',
    'District 12, Ho Chi Minh City, Vietnam',
    ]

    # Calculate global bounds for state normalization
    all_places = place_names + eval_districts + ['District 5, Ho Chi Minh City, Vietnam', 'District 6, Ho Chi Minh City, Vietnam']
    global_x_min, global_x_max = float('inf'), float('-inf')
    global_y_min, global_y_max = float('inf'), float('-inf')
    for place_name in all_places:
        G = load_graph(place_name)
        x_coords = [G.nodes[n]['x'] for n in G.nodes]
        y_coords = [G.nodes[n]['y'] for n in G.nodes]
        global_x_min = min(global_x_min, min(x_coords))
        global_x_max = max(global_x_max, max(x_coords))
        global_y_min = min(global_y_min, min(y_coords))
        global_y_max = max(global_y_max, max(y_coords))
    global_bounds = (global_x_min, global_x_max, global_y_min, global_y_max)

    # Federated learning
    local_models, local_perfs = [], []
    for cycle, place in enumerate(place_names):
        cache_file = f"cache/{place}.pkl"
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                G = pickle.load(f)
        else:
            G = load_graph(place_name)
            with open(cache_file, 'wb') as f:
                pickle.dump(G, f)
        for _ in range(3):  # 3 local agents
            start, goal = sample_start_goal(G, min_dist=START_GOAL_MIN_DIST, max_dist=START_GOAL_MAX_DIST, force_far=True)
            env = OSMGraphEnv(G, start, goal, global_bounds) 
            agent = DQNAgent(state_size=STATE_SIZE, action_size=env.action_space.n)
            train_cycle(env, agent, episodes=400, max_steps=700, cycle_num=cycle+1, graph=G,start=start, goal=goal)
            perf, _, _ = evaluate_agent(agent, env, episodes=10,start=start, goal=goal)
            local_perfs.append(perf)
            local_models.append(agent.model)
            print(f"Cycle {cycle+1}: Start={start}, Goal={goal}, Distance={nx.shortest_path_length(G, start, goal, weight='length')}")

    # Aggregation
    from agent import DQNNetwork
    weights = (np.array(local_perfs) - np.min(local_perfs)) / (np.ptp(local_perfs) + 1e-10)
    weights /= weights.sum()
    global_model = DQNNetwork(state_size=STATE_SIZE, action_size=max(m.net[-1].out_features for m in local_models))
    global_model = federated_averaging(global_model, local_models, weights)

    # Distillation
    G_distill = load_graph('District 5, Ho Chi Minh City, Vietnam')
    s, g = sample_start_goal(G_distill, min_dist=500, max_dist=10000, force_far=True)
    env_distill = OSMGraphEnv(G_distill, s, g, global_bounds)
    global_model = knowledge_distillation(global_model, local_models, env_distill)

    # Fine-tune
    G_ft = load_graph('District 6, Ho Chi Minh City, Vietnam')
    s, g = sample_start_goal(G_ft, min_dist=500, max_dist=10000, force_far=True)
    env_ft = OSMGraphEnv(G_ft, s, g, global_bounds)
    fine_agent = DQNAgent(state_size=STATE_SIZE, action_size=env_ft.action_space.n)
    fine_agent.model.load_state_dict(global_model.state_dict())
    fine_agent.update_target_model()
    train_cycle(env_ft, fine_agent, episodes=400, max_steps=700, cycle_num=99, graph=G_ft)
    
    # Final eval
    for place in eval_districts:
        cache_file = f"cache/{place_name}.pkl"
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                G = pickle.load(f)
        else:
            G = load_graph(place_name)
            with open(cache_file, 'wb') as f:
                pickle.dump(G, f)
        G = load_graph(place)
        s, g = sample_start_goal(G, min_dist=2000, max_dist=10000, force_far=True)
        env = OSMGraphEnv(G, s, g, global_bounds)
        evaluate_agent(fine_agent, env, episodes=10)

    os.makedirs("models", exist_ok=True)
    fine_agent.save("models/global_model.pt")
    print("Saved global model to models/global_model.pt")

if __name__ == "__main__":
    main()