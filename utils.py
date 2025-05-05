import networkx as nx
import numpy as np
import osmnx as ox
import random
import logging
from networkx.algorithms.components import weakly_connected_components

def create_synthetic_graph():
    G = nx.grid_2d_graph(5, 5)
    G = nx.convert_node_labels_to_integers(G)
    for node in G.nodes:
        G.nodes[node]['x'] = node % 5
        G.nodes[node]['y'] = node // 5
    for u, v in G.edges:
        G[u][v]['length'] = 1.0
    return G

def load_graph(place_name='District 1, Ho Chi Minh City, Vietnam', use_synthetic=False):
    if use_synthetic:
        logging.info("Using synthetic 5x5 grid graph")
        G = create_synthetic_graph()
        return G

    G = ox.graph_from_place(place_name, network_type='drive', simplify=False)
    if not isinstance(G, nx.MultiDiGraph):
        G = nx.utils_graph.get_digraph(G)
    largest_cc = max(weakly_connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    return G

def calculate_graph_diameter(G, samples=30):
    """
    Estimate the graph diameter by sampling random node pairs
    """
    nodes = list(G.nodes)
    max_dist = 0
    for _ in range(samples):
        try:
            s, g = random.sample(nodes, 2)
            path = nx.shortest_path(G, s, g, weight='length')
            path_len = nx.path_weight(G, path, weight='length')
            max_dist = max(max_dist, path_len)
        except:
            continue
    return max_dist

def sample_by_spatial_distribution(G, min_distance_factor=0.4):
    x_coords = np.array([G.nodes[n]['x'] for n in G.nodes])
    y_coords = np.array([G.nodes[n]['y'] for n in G.nodes])
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    diagonal = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
    min_spatial_dist = diagonal * min_distance_factor

    for _ in range(100):
        n1, n2 = random.sample(list(G.nodes), 2)
        x1, y1 = G.nodes[n1]['x'], G.nodes[n1]['y']
        x2, y2 = G.nodes[n2]['x'], G.nodes[n2]['y']
        euclidean_dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        if euclidean_dist >= min_spatial_dist:
            try:
                path_len = nx.shortest_path_length(G, n1, n2, weight='length')
                if path_len >= min_spatial_dist * 0.5:
                    print(f"Chọn điểm từ các quadrant khác nhau: {n1} → {n2} | path length = {path_len:.2f} m | euclidean = {euclidean_dist:.2f}")
                    return n1, n2
            except:
                continue
    return sample_far_start_goal(G)  # Fallback nếu không tìm được

def sample_far_start_goal(G, min_dist=2000, max_attempts=100):
    nodes = list(G.nodes)
    best_pair = None
    best_dist = 0

    for _ in range(max_attempts):
        start = random.choice(nodes)
        try:
            # Tính khoảng cách ngắn nhất từ start đến tất cả các điểm khác
            lengths = nx.single_source_dijkstra_path_length(G, start, weight='length')
            # Sắp xếp các điểm theo khoảng cách giảm dần, chỉ lấy những điểm >= min_dist
            far_nodes = sorted([(n, d) for n, d in lengths.items() if d >= min_dist],
                               key=lambda x: x[1], reverse=True)
            if far_nodes:
                goal, dist = far_nodes[0]  # Chọn điểm xa nhất
                if dist > best_dist:
                    best_dist = dist
                    best_pair = (start, goal)
        except:
            continue

    if best_pair:
        start, goal = best_pair
        print(f"Chọn điểm xa: {start} → {goal} | path length = {best_dist:.2f} m")
        return start, goal
    else:
        raise RuntimeError("Không tìm được cặp điểm xa phù hợp")

def sample_start_goal(G, min_dist=2000, max_dist=6000, max_attempts=100, force_far=True):
    # Thử chọn điểm xa trước nếu force_far=True
    if force_far:
        try:
            return sample_far_start_goal(G, min_dist=min_dist)
        except Exception as e:
            print(f"Far sampling failed: {e}")

    # Thử chọn theo phân bố không gian
    try:
        return sample_by_spatial_distribution(G)
    except Exception as e:
        print(f"Spatial distribution failed: {e}")

    # Fallback: Lấy mẫu ngẫu nhiên với kiểm tra khoảng cách
    nodes = list(G.nodes)
    for _ in range(max_attempts):
        s, g = random.sample(nodes, 2)
        try:
            path_len = nx.shortest_path_length(G, s, g, weight='length')
            if min_dist <= path_len <= max_dist:
                print(f"Path length: {path_len:.2f} m | Nodes: {s} → {g}")
                return s, g
        except:
            continue

    # Nếu tất cả thất bại, chọn ngẫu nhiên
    s, g = random.sample(nodes, 2)
    print(f"Fallback to random nodes: {s} → {g}")
    return s, g