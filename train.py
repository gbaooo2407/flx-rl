import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from tqdm import trange
from config import *
from env import OSMGraphEnv
from utils import sample_start_goal
import random
import csv

def evaluate_agent(agent, base_env, episodes=10, max_steps=2000, start=None, goal=None):
    agent.model.eval()
    total_rewards, goal_reached, avg_distances, paths = [], 0, [], []

    for ep in range(episodes):
        env = OSMGraphEnv(base_env.graph, start, goal, global_bounds=(base_env.x_min, base_env.x_max, base_env.y_min, base_env.y_max))
        state = env.reset()
        done, total_reward, steps, path, distances = False, 0, 0, [env.current_node], []

        while not done and steps < max_steps:
            neighbors = list(env.neighbor_fn(env.current_node))
            valid_next_nodes = []
            for i, node in enumerate(neighbors):
                try:
                    _ = nx.shortest_path(env.graph, node, env.goal_node, weight='length')
                    valid_next_nodes.append((i, node))
                except:
                    continue
            valid_actions = list(range(len(valid_next_nodes)))
            if not valid_actions:
                break
            action = agent.act(state, valid_actions, env, eval_mode=True)
            next_state, reward, done, info = env.step(action)
            visit_count = path.count(env.current_node)

            state = next_state
            total_reward += reward
            path.append(env.current_node)
            distances.append(info.get("dist_to_goal", 0.0))
            steps += 1
            print(f"[Eval-Ep{ep+1}] Step={steps} | Current Node={env.current_node} | Action={action} | Reward={reward:.2f} | Dist={info.get('dist_to_goal'):.2f}")

        total_rewards.append(total_reward)
        avg_distances.append(np.mean(distances) if distances else float('inf'))
        paths.append(path)

        if info.get("dist_to_goal", float('inf')) < 100.0:
            goal_reached += 1
        if steps > 20 and len(set(path[-8:])) <= 2:
            print(f"[Early Exit] Loop suspected at node {env.current_node} at step {steps}")
            break
        env.render_path(path, f"training_progress/paths/eval_ep{ep+1}_s{start}_g{goal}.png")
        # Sau khi evaluate xong
        # for i, path in enumerate(paths):
        #     env.render_path(path, filename=f"training_progress/paths/eval_debug_ep{i+1}.png")
    return np.mean(total_rewards), goal_reached / episodes, paths

def train_cycle(env, agent, episodes, max_steps, cycle_num, graph, start, goal, agent_idx=0):
    import time
    best_reward = float('-inf')
    best_path = None
    rewards_history = []
    goal_reached_count = 0
    avg_distances = []
    patience = 50
    best_avg_reward = float('-inf')
    early_stop_counter = 0
    current_start, current_goal = start, goal

    for episode in trange(episodes, desc=f"Cycle {cycle_num} Training Agent {agent_idx}"):
        start_time = time.perf_counter()

        env.set_start_goal(current_start, current_goal)
        state = env.reset()
        path = [env.current_node]
        total_reward, steps = 0, 0
        done, distances, guided_steps = False, [], 0
        n = 10  # num of episode for full semi-guide
        while not done and steps < max_steps:
            neighbors = list(env.neighbor_fn(env.current_node))
            valid_next_nodes = []
            for i, node in enumerate(neighbors):
                try:
                    _ = nx.shortest_path(env.graph, node, env.goal_node, weight='length')
                    valid_next_nodes.append((i, node))
                except:
                    continue
            valid_actions = list(range(len(valid_next_nodes)))
            if not valid_actions:
                break
            guided_ratio = max(0.9 * ((1 - episode / episodes)**3), 0.0)

            try:
                sp = nx.shortest_path(graph, env.current_node, env.goal_node, weight='length')
            except:
                sp = []

            guided = False
            valid_indices = [node for _, node in valid_next_nodes]
            if episode < n:
                if sp and len(sp) > 1 and sp[1] in valid_indices:
                    action = valid_indices.index(sp[1])
                    guided_steps += 1
                    guided = True
            elif len(sp) > 1 and steps < len(sp) and random.random() < guided_ratio:
                if sp[1] in valid_indices:
                    action = valid_indices.index(sp[1])
                    guided_steps += 1
                    guided = True

            if not guided:
                action = agent.act(state, valid_actions, env)

            next_state, reward, done, info = env.step(action)
            clipped_reward = float(np.clip(reward, MIN_REWARD, MAX_REWARD))
            if clipped_reward > -500:
                agent.remember(state, action, clipped_reward, next_state, done)
            # visit_count = path.count(env.current_node)
            # if episode > 20 and visit_count > 2:
            #     clipped_reward -= 1.0
            if steps % TRAIN_INTERVAL == 0:
                agent.train()

            state = next_state
            total_reward += clipped_reward 
            path.append(env.current_node)
            distances.append(info.get("dist_to_goal", 0.0))
            steps += 1 

        rewards_history.append(total_reward)
        avg_distances.append(np.mean(distances) if distances else float('inf'))
        goal_reached_flag = env.current_node == env.goal_node or info.get("dist_to_goal", float('inf')) < 25.0

        suffix = f"c{cycle_num}_a{agent_idx}_ep{episode+1}_s{start}_g{goal}.png"
        if goal_reached_flag:
            goal_reached_count += 1
            print(f"Ep {episode+1}: GOAL REACHED | Reward={total_reward:.2f} | Steps={steps} | Guided={guided_steps}")
            success_dir = f"training_progress/paths/success/cycle{cycle_num}_agent{agent_idx}"
            os.makedirs(success_dir, exist_ok=True)
            env.render_path(path, f"{success_dir}/{suffix}")
        else:
            print(f"Ep {episode+1}: FAILED | Final node={info.get('final_node', env.current_node)} | Reward={total_reward:.2f} | Guided={guided_steps}")
            fail_dir = f"training_progress/paths/fail/cycle{cycle_num}_agent{agent_idx}"
            os.makedirs(fail_dir, exist_ok=True)
            env.render_path(path, f"{fail_dir}/{suffix}")

        log_data = {
            "episode": episode + 1,
            "steps": steps,
            "total_reward": total_reward,
            "goal_reached": int(goal_reached_flag),
            "final_node": info.get("final_node", env.current_node),
            "dist_to_goal": info.get("dist_to_goal", float('inf')),
            "guided_steps": guided_steps,
            "time_sec": time.perf_counter() - start_time,
            "start": start,
            "goal": goal,
            "cycle": cycle_num,
            "agent_id": agent_idx
        }
        if episode % 10 == 0:
            print(f"[Episode {episode}] ε={agent.epsilon:.3f}, memory={len(agent.memory)}, guided={guided_steps}")

        os.makedirs("training_progress/logs", exist_ok=True)
        log_path = f"training_progress/logs/train_log_cycle{cycle_num}_agent{agent_idx}.csv"
        write_header = not os.path.exists(log_path)
        with open(log_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_data.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(log_data)

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        # Sau khi xử lý goal_reached_flag
        if goal_reached_flag and total_reward > best_reward and guided_steps / max(steps, 1) < 0.3:
            best_reward = total_reward
            best_path = path.copy()
            
            # Lưu model tốt nhất
            os.makedirs("models", exist_ok=True)
            model_path = f"models/agent_cycle{cycle_num}_agent{agent_idx}_best.pt"
            agent.save(model_path)
            print(f"Saved BEST model after cycle {cycle_num}, agent {agent_idx}, episode {episode+1} → {model_path}")
            
            # Vẽ đường đi tốt nhất
            env.render_path(best_path, f"training_progress/paths/best_cycle{cycle_num}_agent{agent_idx}_ep{episode+1}.png")
            
        recent_avg = np.mean(rewards_history[-10:]) if len(rewards_history) >= 10 else 0
        if recent_avg > best_avg_reward:
            best_avg_reward = recent_avg
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"Early stopping at episode {episode+1} due to no improvement in {patience} episodes.")
            print(f"Best reward = {best_avg_reward:.2f}")
            break
        agent.train() 

    plt.figure()
    plt.plot(rewards_history)
    plt.title(f"Training Reward - Cycle {cycle_num} Agent {agent_idx}")
    plt.savefig(f"training_progress/reward_curve_cycle_{cycle_num}_agent{agent_idx}.png")
    plt.close()

    plt.figure()
    plt.plot(avg_distances)
    plt.title(f"Avg Distance to Goal - Cycle {cycle_num} Agent {agent_idx}")
    plt.savefig(f"training_progress/distance_curve_cycle_{cycle_num}_agent{agent_idx}.png")
    plt.close()

    return best_path, best_reward, rewards_history