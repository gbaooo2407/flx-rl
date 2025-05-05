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

def evaluate_agent(agent, base_env, episodes=10, max_steps=2000, start=None, goal=None, diversify=True):
    agent.model.eval()
    total_rewards, goal_reached, avg_distances, paths = [], 0, [], []

    for ep in range(episodes):
        try:
            s, g = sample_start_goal(base_env.graph, min_dist=1000, max_dist=10000, max_attempts=50, force_far=True)
        except:
            s, g = base_env.start_node, base_env.goal_node

        env = OSMGraphEnv(base_env.graph, s, g, global_bounds=(base_env.x_min, base_env.x_max, base_env.y_min, base_env.y_max))

        state = env.reset()
        done, total_reward, steps, path, distances = False, 0, 0, [env.current_node], []

        while not done and steps < max_steps:
            neighbors = list(env.neighbor_fn(env.current_node))
            valid_actions = list(range(len(neighbors)))
            if not valid_actions:
                break
            action = agent.act(state, valid_actions, env, eval_mode=True)
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
            path.append(env.current_node)
            distances.append(info.get("dist_to_goal", 0.0))
            steps += 1

        total_rewards.append(total_reward)
        avg_distances.append(np.mean(distances) if distances else float('inf'))
        paths.append(path)

        if info.get("dist_to_goal", float('inf')) < 100.0:
            goal_reached += 1

        env.render_path(path, f"training_progress/paths/eval_ep{ep+1}.png")

    return np.mean(total_rewards), goal_reached / episodes, paths

def train_cycle(env, agent, episodes, max_steps, cycle_num, graph, start=None, goal=None):
    import time
    best_reward = float('-inf')
    best_path = None
    rewards_history = []
    goal_reached_count = 0
    avg_distances = []
    patience = 50
    best_avg_reward = float('-inf')
    early_stop_counter = 0
    episodes_per_goal = 20
    current_start, current_goal = None, None

    for episode in trange(episodes, desc=f"Cycle {cycle_num} Training"):
        start_time = time.perf_counter()

        if episode % episodes_per_goal == 0 or current_start is None:
            if episode < 50:
                current_start, current_goal = sample_start_goal(graph, min_dist=300, max_dist=1000, force_far=False)
            elif episode < 150:
                current_start, current_goal = sample_start_goal(graph, min_dist=800, max_dist=3000, force_far=False)
            else:
                current_start, current_goal = sample_start_goal(graph, min_dist=2000, max_dist=6000, force_far=True)

        env.set_start_goal(current_start, current_goal)
        state = env.reset()
        path = [env.current_node]
        total_reward, steps = 0, 0
        done, distances, guided_steps = False, [], 0

        while not done and steps < max_steps:
            neighbors = list(env.neighbor_fn(env.current_node))
            valid_actions = list(range(len(neighbors)))
            if not valid_actions:
                break
            guided_ratio = max(0.9 * (1 - episode / episodes), 0.0)

            if steps < 50 and random.random() < guided_ratio:
                try:
                    sp = nx.shortest_path(graph, env.current_node, env.goal_node, weight='length')
                    if len(sp) > 1 and sp[1] in neighbors:
                        action = neighbors.index(sp[1])
                        guided_steps += 1
                    else:
                        print(f"[Guide-Fail] step={steps} | curr={env.current_node} → next {sp[1]} NOT in neighbors")
                        action = agent.act(state, valid_actions, env)
                except Exception as e:
                    print(f"[Guide-Exception] step={steps} | node={env.current_node} → Error: {e}")
                    action = agent.act(state, valid_actions, env)
            else:
                action = agent.act(state, valid_actions, env)

            next_state, reward, done, info = env.step(action)
            clipped_reward = float(np.clip(reward, MIN_REWARD, MAX_REWARD))
            if clipped_reward > -300:
                agent.remember(state, action, clipped_reward, next_state, done)

            if steps % TRAIN_INTERVAL == 0:
                agent.train()

            state = next_state
            total_reward += reward
            path.append(env.current_node)
            distances.append(info.get("dist_to_goal", 0.0))
            steps += 1

        rewards_history.append(total_reward)
        avg_distances.append(np.mean(distances) if distances else float('inf'))
        goal_reached_flag = env.current_node == env.goal_node or info.get("dist_to_goal", float('inf')) < 25.0

        if goal_reached_flag:
            goal_reached_count += 1
            print(f" Ep {episode+1}: GOAL REACHED | Reward={total_reward:.2f} | Steps={steps} | Guided={guided_steps}")
            if total_reward > best_reward:
                best_reward = total_reward
                best_path = path.copy()
                env.render_path(best_path, f"training_progress/paths/best_cycle{cycle_num}_ep{episode+1}.png")
        else:
            print(f" Ep {episode+1}: Failed | Final node={info.get('final_node', env.current_node)} | Reward={total_reward:.2f} | Guided={guided_steps}")
            if episode % 50 == 0:
                env.render_path(path, f"training_progress/paths/fail_cycle{cycle_num}_ep{episode+1}.png")

        if total_reward > best_avg_reward:
            best_avg_reward = total_reward
            early_stop_counter = 0
            if episode >= EVALUATE_AFTER_EPISODE:
                val_reward, val_success, _ = evaluate_agent(agent, env, episodes=3)
                print(f"[Quick Eval] Ep {episode+1} | Reward: {val_reward:.2f} | Success: {val_success:.2%}")
        else:
            early_stop_counter += 1

        print(f"Final dist to goal: {info.get('dist_to_goal', float('inf')):.2f} meters")

        if early_stop_counter >= patience:
            print(f"Early stopping at episode {episode+1} due to no improvement in {patience} episodes.")
            print(f"Early stopping triggered — Best reward = {best_avg_reward:.2f}")
            break

        log_data = {
            "episode": episode + 1,
            "steps": steps,
            "total_reward": total_reward,
            "goal_reached": int(goal_reached_flag),
            "final_node": info.get("final_node", env.current_node),
            "dist_to_goal": info.get("dist_to_goal", float('inf')),
            "guided_steps": guided_steps,
            "time_sec": time.perf_counter() - start_time
        }

        os.makedirs("training_progress/logs", exist_ok=True)
        log_path = f"training_progress/logs/train_log_cycle{cycle_num}.csv"
        write_header = not os.path.exists(log_path)
        with open(log_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_data.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(log_data)

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        if total_reward > best_avg_reward:
            best_avg_reward = total_reward
            best_reward = total_reward  # cập nhật giá trị tốt nhất toàn kỳ
            best_path = path.copy()     # lưu lại path tốt nhất

            if episode >= EVALUATE_AFTER_EPISODE:
                val_reward, val_success, _ = evaluate_agent(agent, env, episodes=3)
                print(f"[Quick Eval] Ep {episode+1} | Reward: {val_reward:.2f} | Success: {val_success:.2%}")

            # Save best model so far
            os.makedirs("models", exist_ok=True)
            model_path = f"models/agent_cycle{cycle_num}_best.pt"
            agent.save(model_path)
            print(f"Saved BEST model after cycle {cycle_num}, episode {episode+1} → {model_path}")
                
    plt.figure()
    plt.plot(rewards_history)
    plt.title(f"Training Reward - Cycle {cycle_num}")
    plt.savefig(f"training_progress/reward_curve_cycle_{cycle_num}.png")
    plt.close()

    plt.figure()
    plt.plot(avg_distances)
    plt.title(f"Avg Distance to Goal - Cycle {cycle_num}")
    plt.savefig(f"training_progress/distance_curve_cycle_{cycle_num}.png")
    plt.close()

    return best_path, best_reward, rewards_history