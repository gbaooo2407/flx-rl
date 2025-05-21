import gym
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from gym import spaces
from config import *
import random
import os
import csv
from utils import sample_start_goal
class OSMGraphEnv(gym.Env):
    def __init__(self, graph, start_node, goal_node, global_bounds=None):
        super(OSMGraphEnv, self).__init__()
        self.graph = graph
        self.start_node = start_node
        self.goal_node = goal_node
        self.current_node = start_node
        self.max_steps = 2000
        self.reward_scale = REWARD_SCALE
        self.goal_reward = GOAL_REWARD
        self.step_penalty = STEP_PENALTY
        self.dead_end_penalty = DEAD_END_PENALTY
        self.max_step_penalty = 10
        self.is_directed = isinstance(graph, (nx.DiGraph, nx.MultiDiGraph))
        self.neighbor_fn = graph.successors if self.is_directed else graph.neighbors

        self.action_space = spaces.Discrete(MAX_ACTIONS)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(11,), dtype=np.float32)

        x = np.array([graph.nodes[n]['x'] for n in graph.nodes])
        y = np.array([graph.nodes[n]['y'] for n in graph.nodes])
        if global_bounds:
            self.x_min, self.x_max, self.y_min, self.y_max = global_bounds
        else:
            self.x_min, self.x_max = x.min(), x.max()
            self.y_min, self.y_max = y.min(), y.max()
        self.max_distance = np.sqrt((self.x_max - self.x_min)**2 + (self.y_max - self.y_min)**2)
        self.visited_nodes = []

    def step(self, action):
        neighbors = list(self.neighbor_fn(self.current_node))
        if not neighbors:
            return self.get_state(), float(MIN_REWARD), True, {
                "dist_to_goal": float('inf'),
                "reachable": False,
                "final_node": self.current_node
            }

        valid_next_nodes = []
        for a, node in enumerate(neighbors):
            try:
                _ = nx.shortest_path(self.graph, node, self.goal_node, weight='length')
                valid_next_nodes.append((a, node))
            except:
                continue

        if not valid_next_nodes:
            print(f"[All Dead-End] Node {self.current_node} → no reachable neighbors")
            return self.get_state(), float(MIN_REWARD), True, {
                "dist_to_goal": float('inf'),
                "reachable": False,
                "final_node": self.current_node
            }

        # Lấy hành động được chọn trong danh sách hợp lệ
        if action >= len(valid_next_nodes):
            print(f"[ERROR] Action {action} out of range! Valid range: 0 to {len(valid_next_nodes)-1}")
            return self.get_state(), float(MIN_REWARD), True, {
                "dist_to_goal": float('inf'),
                "reachable": False,
                "final_node": self.current_node
            }

        _, next_node = valid_next_nodes[action]
        # Path-based distances
        try:
            sp_current = nx.shortest_path(self.graph, self.current_node, self.goal_node, weight='length')
            dist_current_goal = nx.path_weight(self.graph, sp_current, weight='length')
        except:
            dist_current_goal = float('inf')

        try:
            sp_next = nx.shortest_path(self.graph, next_node, self.goal_node, weight='length')
            dist_next_goal = nx.path_weight(self.graph, sp_next, weight='length')
        except:
            dist_next_goal = float('inf')

        reachable = np.isfinite(dist_next_goal)
        if not reachable:
            print(f"[Dead-End] Node: {self.current_node} → Action {action} leads to unreachable node {next_node}")
            return self.get_state(), float(MIN_REWARD), True, {
                "dist_to_goal": float('inf'),
                "reachable": False,
                "final_node": next_node
            }

        # Initial distance used for reward normalization
        if self.steps == 0:
            self.initial_path_len = dist_current_goal if np.isfinite(dist_current_goal) else self.max_distance

        # Điều chỉnh cách tính reward
        improvement = dist_current_goal - dist_next_goal
        
        # Cải thiện reward scale dựa trên tiến trình
        progress_percentage = improvement / (self.initial_path_len + 1e-6)
        reward = progress_percentage * self.reward_scale
        
        # Thêm reward trung gian cho việc đi đúng hướng
        if improvement > 0:
            # Khuyến khích bất kỳ sự tiến bộ nào, nhỏ hay lớn
            reward += 1.0 + (progress_percentage * 10)  # Reward nhỏ cho tiến bộ, tỷ lệ với mức độ cải thiện
        
        # Giảm mức phạt cho backtracking
        if improvement < 0 and len(self.visited_nodes) > 1 and next_node == list(self.visited_nodes)[-2]:
            reward -= 0.5  # Giảm từ 1.0 xuống 0.5
        
        # Step penalty vẫn giữ nguyên
        reward -= self.step_penalty
        
        # Sửa đổi phạt lặp lại node để ít khắc nghiệt hơn
        if next_node in self.visited_nodes:
            # Phạt nhỏ với node mới ghé thăm, tăng dần với mỗi lần lặp lại
            visit_count = self.visited_nodes.count(next_node)
            repeat_penalty = min(REPEAT_PENALTY * (visit_count * 0.5), REPEAT_PENALTY * 2)
            reward -= repeat_penalty
        
        # Cải thiện reward cho tiến bộ lớn
        if improvement > 0.03 * self.initial_path_len:  # Giữ ngưỡng này
            reward += 3.0  
        
        # Reward bổ sung khi gần đến mục tiêu, tạo hiệu ứng "hút" agent về đích
        if dist_next_goal < self.initial_path_len * 0.2:  # Trong 20% cuối đường đi
            reward += 2.5
        elif dist_next_goal < self.initial_path_len * 0.5:  # Trong 50% cuối đường đi
            reward += 1.0

        self.visited_nodes.append(next_node)

        goal_reached = next_node == self.goal_node
        very_close_to_goal = dist_next_goal < 25.0
        done = False

        if goal_reached:
            reward += self.goal_reward
            done = True
        elif very_close_to_goal:
            reward += self.goal_reward * 0.8  # 80% phần thưởng khi đến rất gần
            done = True
        if not goal_reached and done:
            print(f"[Fail] Final node: {next_node}, Distance to goal: {dist_next_goal:.2f} m")
        self.current_node = next_node
        self.steps += 1
        if self.steps >= self.max_steps:
            reward -= self.max_step_penalty
            done = True

        # Logging
        log_data = {
            "episode": self.current_episode if hasattr(self, "current_episode") else -1,
            "step": self.steps,
            "node": self.current_node,
            "goal_node": self.goal_node,
            "dist_to_goal": dist_next_goal,
            "reward": reward,
            "improvement": improvement,
            "done": done,
            "reachable": reachable,
            "goal_reached": goal_reached
        }

        os.makedirs("training_progress/logs", exist_ok=True)
        log_file = "training_progress/logs/step_log.csv"
        write_header = not os.path.exists(log_file)
        with open(log_file, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_data.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(log_data)

        state = self.get_state()
        return state, float(np.clip(reward, MIN_REWARD, MAX_REWARD)), done, {
            "dist_to_goal": dist_next_goal,
            "reachable": reachable,
            "final_node": next_node,
            "goal_reached": goal_reached,
            "visited_count": self.visited_nodes.count(next_node)

        }

    def get_state(self):
        n = lambda i: self.graph.nodes[i]
        current, goal = n(self.current_node), n(self.goal_node)

        norm = lambda x, min_val, max_val: (x - min_val) / (max_val - min_val + 1e-8)
        
        # Tính toán một số đặc trưng bổ sung
        try:
            shortest_path = nx.shortest_path(self.graph, self.current_node, self.goal_node, weight="length")
            path_length = nx.path_weight(self.graph, shortest_path, weight="length")
            normalized_path_length = path_length / 1000  # Đổi sang km
            
            # Đếm số neighbor nodes dẫn về phía mục tiêu
            neighbors = list(self.neighbor_fn(self.current_node))
            good_neighbors = 0
            for neighbor in neighbors:
                try:
                    nb_path = nx.shortest_path(self.graph, neighbor, self.goal_node, weight="length")
                    if len(nb_path) < len(shortest_path):
                        good_neighbors += 1
                except:
                    continue
            
            good_neighbors_ratio = good_neighbors / max(1, len(neighbors))
        except:
            normalized_path_length = 1.0  # Giá trị mặc định cao
            good_neighbors_ratio = 0.0
        
        # Cải thiện state vector với thông tin thêm
        state = [
            norm(current["x"], self.x_min, self.x_max),
            norm(current["y"], self.y_min, self.y_max),
            norm(goal["x"], self.x_min, self.x_max),
            norm(goal["y"], self.y_min, self.y_max),
            self.graph.degree[self.current_node] / 10,
            self.graph.degree[self.goal_node] / 10,
            normalized_path_length,
            current.get("street_type", 0) / 10,
            goal.get("street_type", 0) / 10,
            int(current.get("highway", "no") == "yes"),
            int(goal.get("highway", "no") == "yes"),
            # Thêm 2 đặc trưng mới
            good_neighbors_ratio,  # Tỷ lệ neighbor nodes dẫn về phía mục tiêu
            min(1.0, self.steps / 200)  # Thêm thông tin về thời gian (chuẩn hóa)
        ]
        
        return np.array(state, dtype=np.float32)

    def reset(self):
        self.current_node = self.start_node
        self.steps = 0
        self.visited_nodes = []
        return self.get_state()

    def set_start_goal(self, start, goal):
        if start is not None and goal is not None:
            self.start_node = start
            self.goal_node = goal

    def render_path(self, path, filename=None):
        if not path: return
        pos = {n: (self.graph.nodes[n]['x'], self.graph.nodes[n]['y']) for n in self.graph.nodes}
        plt.figure(figsize=(10, 8))
        nx.draw(self.graph, pos, node_size=10, node_color='gray', edge_color='lightgray', alpha=0.5)

        # Draw agent path
        if len(path) > 1:
            edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            nx.draw_networkx_edges(self.graph, pos, edgelist=edges, edge_color='red', width=2)

        # Always draw correct start (green) and goal (blue)
        if self.start_node in pos:
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[self.start_node], node_color='green', node_size=100)
        if self.goal_node in pos:
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[self.goal_node], node_color='blue', node_size=100)

        # (Optional) Mark where the agent actually stopped (purple if fail)
        if path[-1] != self.goal_node:
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[path[-1]], node_color='purple', node_size=80)

        try:
            sp = nx.shortest_path(self.graph, self.start_node, self.goal_node, weight='length')
            sp_edges = list(zip(sp[:-1], sp[1:]))
            nx.draw_networkx_edges(self.graph, pos, edgelist=sp_edges, edge_color='green', style='dashed', width=2)
        except:
            pass

        plt.axis('off')
        plt.title("Agent Path")
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

