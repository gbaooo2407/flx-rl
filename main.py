
import os
import matplotlib.pyplot as plt
from config import *
from env import OSMGraphEnv
from agent import DQNAgent, DQNNetwork
from train import train_cycle, evaluate_agent
from federated import federated_averaging, knowledge_distillation
from utils import load_graph, sample_start_goal
import numpy as np
import networkx as nx
import pickle
from slugify import slugify


def filter_models_by_action_size(models, perfs):
    """
    Giữ lại các model có cùng action_size phổ biến nhất.
    Trả về: models_filtered, perfs_filtered, action_size
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for i, m in enumerate(models):
        action_size = m.net[-1].out_features
        groups[action_size].append((m, perfs[i]))

    # Chọn action_size có nhiều model nhất
    best_action_size = max(groups.items(), key=lambda x: len(x[1]))[0]
    models_filtered, perfs_filtered = zip(*groups[best_action_size])
    print(f"[Filter] ✔ Kept {len(models_filtered)} models with action_size = {best_action_size}")
    return list(models_filtered), list(perfs_filtered), best_action_size

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
        G_tmp = load_graph(place_name)
        x_coords = [G_tmp.nodes[n]['x'] for n in G_tmp.nodes]
        y_coords = [G_tmp.nodes[n]['y'] for n in G_tmp.nodes]
        global_x_min = min(global_x_min, min(x_coords))
        global_x_max = max(global_x_max, max(x_coords))
        global_y_min = min(global_y_min, min(y_coords))
        global_y_max = max(global_y_max, max(y_coords))
    global_bounds = (global_x_min, global_x_max, global_y_min, global_y_max)

   # Federated learning
    local_models, local_perfs = [], []

    for cycle, place in enumerate(place_names):

        safe_name = slugify(place)
        cache_file = f"cache/{safe_name}.pkl"
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                G = pickle.load(f)
            print(f"[Cycle {cycle+1}] Loaded graph for {place} ({G.number_of_nodes()} nodes)")

        else:
            G = load_graph(place)
            with open(cache_file, 'wb') as f:
                pickle.dump(G, f)
            print(f"[Cycle {cycle+1}] Loaded graph for {place} ({G.number_of_nodes()} nodes)")

        for agent_idx in range(3):
            model_path = f"models/agent_cycle{cycle+1}_agent{agent_idx}_best.pt"

            # Chỉ train nếu model chưa tồn tại
            if not os.path.exists(model_path):
                print(f"[TRAIN] ➤ Training Agent {agent_idx} in Cycle {cycle+1}")
                start, goal = sample_start_goal(G, min_dist=800, max_dist=3000, force_far=False)
                env = OSMGraphEnv(G, start, goal, global_bounds)
                agent = DQNAgent(state_size=STATE_SIZE, action_size=env.action_space.n)

                train_cycle(env, agent, episodes=400, max_steps=700,
                            cycle_num=3, graph=G,
                            start=start, goal=goal, agent_idx=agent_idx)
            else:
                print(f"[SKIP] ✅ Found pre-trained model: {model_path}")

            # Load model để đưa vào federated aggregation
            env_for_eval = OSMGraphEnv(G, *sample_start_goal(G), global_bounds)  # random eval start-goal
            best_agent = DQNAgent(state_size=STATE_SIZE, action_size=env_for_eval.action_space.n)
            best_agent.load(model_path)
            best_agent.update_target_model()

            perf, _, _ = evaluate_agent(best_agent, env_for_eval, episodes=10)
            local_perfs.append(perf)
            local_models.append(best_agent.model)

            print(f"[EVAL] Cycle {cycle+1} | Agent {agent_idx} | Perf = {perf:.2f}")

    # Lọc models theo action_size thống nhất
    local_models, local_perfs, chosen_action_size = filter_models_by_action_size(local_models, local_perfs)

    # Tiến hành federated averaging
    weights = (np.array(local_perfs) - np.min(local_perfs)) / (np.ptp(local_perfs) + 1e-10)
    weights /= weights.sum()
    global_model = DQNNetwork(state_size=STATE_SIZE, action_size=chosen_action_size)
    global_model = federated_averaging(global_model, local_models, weights)

    # Distillation phase
    G_distill = load_graph(place_name='District 5, Ho Chi Minh City, Vietnam')
    s, g = sample_start_goal(G_distill, min_dist=500, max_dist=5000, force_far=True)
    env_distill = OSMGraphEnv(G_distill, s, g, global_bounds)
    global_model = knowledge_distillation(global_model, local_models, G_distill, global_bounds)

    # Fine-tuning on a new district
    G_ft = load_graph(place_name='District 6, Ho Chi Minh City, Vietnam')
    s, g = sample_start_goal(G_ft, min_dist=500, max_dist=5000, force_far=True)
    env_ft = OSMGraphEnv(G_ft, s, g, global_bounds)
    fine_agent = DQNAgent(state_size=STATE_SIZE, action_size=env_ft.action_space.n)
    fine_agent.model.load_state_dict(global_model.state_dict())
    fine_agent.update_target_model()
    train_cycle(env_ft, fine_agent, episodes=400, max_steps=700,
                cycle_num=99, graph=G_ft, start=s, goal=g, agent_idx=0)

    # Final evaluation on unseen districts
    for place in eval_districts:
        cache_file = f"cache/{place}.pkl"
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                G = pickle.load(f)
        else:
            G = load_graph(place)
            with open(cache_file, 'wb') as f:
                pickle.dump(G, f)

        s, g = sample_start_goal(G, min_dist=2000, max_dist=10000, force_far=True)
        env = OSMGraphEnv(G, s, g, global_bounds)
        evaluate_agent(fine_agent, env, episodes=10, start=s, goal=g)

    os.makedirs("models", exist_ok=True)
    fine_agent.save("models/global_model.pt")
    print("✅ Saved global model to models/global_model.pt")

if __name__ == "__main__":
    main()
