import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from tqdm import trange
from config import *
from env import OSMGraphEnv
from utils import sample_start_goal,sample_by_spatial_distribution


def evaluate_agent(agent, base_env, episodes=10, max_steps=700, start=None, goal=None):
    import csv
    agent.model.eval()
    total_rewards, goal_reached, avg_distances, paths = [], 0, [], []
    eval_log = []

    for ep in range(episodes):
        start, goal = sample_start_goal(base_env.graph, min_dist=2000, max_dist=10000, force_far=True)
        env = OSMGraphEnv(base_env.graph, start, goal,
                          global_bounds=(base_env.x_min, base_env.x_max, base_env.y_min, base_env.y_max))
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

            state = next_state
            total_reward += reward
            path.append(env.current_node)
            distances.append(info.get("dist_to_goal", 0.0))
            steps += 1

            if steps > 20 and len(set(path[-8:])) <= 2:
                print(f"[Early Exit] Stuck loop at {env.current_node}")
                break

        total_rewards.append(total_reward)
        avg_distances.append(np.mean(distances) if distances else float('inf'))
        paths.append(path)
        reached = int(info.get("dist_to_goal", float('inf')) < 100.0)
        goal_reached += reached

        env.render_path(path, f"training_progress/paths/eval_ep{ep+1}_s{start}_g{goal}.png")
        eval_log.append({
            "episode": ep + 1,
            "start": start,
            "goal": goal,
            "final_node": env.current_node,
            "goal_reached": reached,
            "reward": total_reward,
            "steps": steps,
            "dist_to_goal": info.get("dist_to_goal", float('inf'))
        })

    log_path = f"training_progress/logs/eval_log.csv"
    write_header = not os.path.exists(log_path)
    with open(log_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=eval_log[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(eval_log)

    return np.mean(total_rewards), goal_reached / episodes, paths

def train_cycle(env, agent, episodes, max_steps, cycle_num, graph, start, goal, agent_idx=0):
    import time, os, csv, random
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import trange
    from utils import sample_start_goal, sample_by_spatial_distribution
    from agent import PrioritizedReplayBuffer
    os.makedirs("training_progress/paths/success", exist_ok=True)
    os.makedirs("training_progress/paths/fail", exist_ok=True)
    os.makedirs("training_progress/logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    best_reward = float('-inf')
    best_path = None
    rewards_history, success_history, guided_ratios = [], [], []
    best_success_rate, early_stop_counter, patience = 0, 0, 50
    phase_split = 0

    for episode in trange(episodes, desc=f"Cycle {cycle_num} Training Agent {agent_idx}"):
        start_time = time.perf_counter()

        # --- Phase Switching ---
        if phase_split == 0 and episode >= 20 and len(success_history) >= 15:
            recent_success = sum(success_history[-15:])
            recent_guided_ratio = np.mean(guided_ratios[-15:])
            print(f"[CHECK] Episode {episode} | Success={recent_success}/15 | Guided Ratio={recent_guided_ratio:.2f}")
            if recent_success / 15.0 >= 0.8 and recent_guided_ratio < 0.3:
                phase_split = episode
                print(f"[PHASE SHIFT] → Agent enters Phase 2 at episode {episode}")
                early_stop_counter = 0
                # ✅ Reset replay buffer to remove phase 1 bias
                agent.memory = PrioritizedReplayBuffer(agent.memory.capacity)

        # --- Adjust start-goal pair after phase split ---
        if phase_split > 0 and episode >= phase_split:
            r = random.random()
            if r < 0.25:
                current_start, current_goal = start, goal
            elif r < 0.6:
                current_start, current_goal = sample_start_goal(graph, min_dist=1000, max_dist=4000)
            else:
                current_start, current_goal = sample_by_spatial_distribution(graph)
        else:
            current_start, current_goal = start, goal

        env.set_start_goal(current_start, current_goal)
        state = env.reset()
        path = [env.current_node]
        total_reward, steps, guided_steps = 0, 0, 0
        done, distances = False, []

        # --- Guided Ratio Setup ---
        if episode < phase_split or phase_split == 0:
            guided_ratio = max(0.8 * np.exp(-episode / (episodes * 0.1)), 0.05)
        else:
            guided_ratio = max(0.4 * np.exp(-(episode - phase_split) / (episodes * 0.3)), 0.05)

        while not done and steps < max_steps:
            neighbors = list(env.neighbor_fn(env.current_node))
            valid_next_nodes = []
            for i, node in enumerate(neighbors):
                try:
                    _ = nx.shortest_path(graph, node, env.goal_node, weight='length')
                    valid_next_nodes.append((i, node))
                except:
                    continue
            valid_actions = list(range(len(valid_next_nodes)))
            if not valid_actions:
                break

            guided = False
            try:
                sp = nx.shortest_path(graph, env.current_node, env.goal_node, weight='length')
            except:
                sp = []

            valid_indices = [node for _, node in valid_next_nodes]
            if sp and len(sp) > 1 and sp[1] in valid_indices and random.random() < guided_ratio:
                if guided_steps / max(steps, 1) < 0.3:
                    action = valid_indices.index(sp[1])
                    guided_steps += 1
                    guided = True

            if not guided:
                action = agent.act(state, valid_actions, env)

            next_state, reward, done, info = env.step(action)
            clipped_reward = float(np.clip(reward, -300, 300))

            visit_count = path.count(env.current_node)
            if visit_count > 3:
                clipped_reward -= 1.0 * (visit_count - 3)  # ✅ tăng phạt loop node trong train
            
            agent.remember(state, action, clipped_reward, next_state, done)
            if steps % 2 == 0 or done:
                agent.train()

            state = next_state
            total_reward += clipped_reward
            path.append(env.current_node)
            distances.append(info.get("dist_to_goal", 0.0))
            steps += 1

        goal_reached = env.current_node == env.goal_node or info.get("dist_to_goal", float('inf')) < 25.0
        rewards_history.append(total_reward)
        success_history.append(1 if goal_reached else 0)
        guided_ratios.append(guided_steps / max(steps, 1))

        suffix = f"c{cycle_num}_a{agent_idx}_ep{episode+1}_s{current_start}_g{current_goal}.png"
        render_dir = "success" if goal_reached else "fail"
        env.render_path(path, f"training_progress/paths/{render_dir}/{suffix}")

        print(f"Ep {episode+1}: {'GOAL REACHED' if goal_reached else 'FAILED'} | Reward={total_reward:.2f} | Steps={steps} | Guided={guided_steps}")

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

        if goal_reached and total_reward > best_reward:
            best_reward = total_reward
            best_path = path.copy()
            model_path = f"models/agent_cycle{cycle_num}_agent{agent_idx}_best.pt"
            agent.save(model_path)
            env.render_path(best_path, f"training_progress/paths/best_cycle{cycle_num}_agent{agent_idx}_ep{episode+1}.png")

        if len(success_history) >= 10:
            recent_success = sum(success_history[-10:]) / 10.0
            if recent_success >= best_success_rate:
                best_success_rate = recent_success
                early_stop_counter = 0
            else:
                early_stop_counter += 1

        if episode > 30 and early_stop_counter >= patience:
            print(f"[EARLY STOP] No improvement for {patience} episodes.")
            break

        log_data = {
            "episode": episode + 1,
            "steps": steps,
            "total_reward": total_reward,
            "goal_reached": int(goal_reached),
            "final_node": info.get("final_node", env.current_node),
            "dist_to_goal": info.get("dist_to_goal", float('inf')),
            "guided_steps": guided_steps,
            "guided_ratio": guided_ratio,
            "epsilon": agent.epsilon,
            "start": current_start,
            "goal": current_goal,
            "cycle": cycle_num,
            "agent_id": agent_idx,
            "time_sec": time.perf_counter() - start_time
        }
        log_path = f"training_progress/logs/train_log_cycle{cycle_num}_agent{agent_idx}.csv"
        write_header = not os.path.exists(log_path)
        with open(log_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_data.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(log_data)

    # Plot
    plt.figure()
    plt.plot(rewards_history)
    if phase_split > 0:
        plt.axvline(phase_split, color='orange', linestyle='--', label='Phase Shift')
        plt.legend()
    plt.title(f"Training Reward - Cycle {cycle_num} Agent {agent_idx}")
    plt.savefig(f"training_progress/reward_curve_cycle_{cycle_num}_agent{agent_idx}.png")
    plt.close()

    return best_path, best_reward, rewards_history