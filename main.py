# import os
# import matplotlib.pyplot as plt
# from config import *
# from env import OSMGraphEnv
# from agent import DQNAgent, DQNNetwork
# from train import train_cycle, evaluate_agent
# from federated import federated_averaging, knowledge_distillation
# from utils import load_graph, sample_start_goal
# import numpy as np
# import networkx as nx
# import pickle
# from slugify import slugify


# def filter_models_by_action_size(models, perfs):
#     """
#     Giữ lại các model có cùng action_size phổ biến nhất.
#     Trả về: models_filtered, perfs_filtered, action_size
#     """
#     from collections import defaultdict

#     groups = defaultdict(list)
#     for i, m in enumerate(models):
#         action_size = m.net[-1].out_features
#         groups[action_size].append((m, perfs[i]))

#     # Chọn action_size có nhiều model nhất
#     best_action_size = max(groups.items(), key=lambda x: len(x[1]))[0]
#     models_filtered, perfs_filtered = zip(*groups[best_action_size])
#     print(f"[Filter] ✔ Kept {len(models_filtered)} models with action_size = {best_action_size}")
#     return list(models_filtered), list(perfs_filtered), best_action_size


# def evaluate_phases(place_name, agent_idx, cycle_num, global_bounds):
#     """
#     Đánh giá hiệu suất của cả hai mô hình phase 1 và phase 2 cho một agent cụ thể.
    
#     Args:
#         place_name: Tên địa điểm để tải đồ thị
#         agent_idx: Chỉ số của agent
#         cycle_num: Số chu kỳ
#         global_bounds: Giới hạn toàn cục cho state normalization
    
#     Returns:
#         phase1_perf: Hiệu suất của mô hình phase 1
#         phase2_perf: Hiệu suất của mô hình phase 2 (hoặc None nếu không tồn tại)
#         model_to_use: Mô hình tốt nhất để sử dụng (phase 1 hoặc phase 2)
#     """
#     # Tải đồ thị
#     safe_name = slugify(place_name)
#     cache_file = f"cache/{safe_name}.pkl"
#     if os.path.exists(cache_file):
#         with open(cache_file, 'rb') as f:
#             G = pickle.load(f)
#         print(f"[Eval] Loaded graph for {place_name} ({G.number_of_nodes()} nodes)")
#     else:
#         G = load_graph(place_name)
#         os.makedirs("cache", exist_ok=True)
#         with open(cache_file, 'wb') as f:
#             pickle.dump(G, f)
#         print(f"[Eval] Created and loaded graph for {place_name} ({G.number_of_nodes()} nodes)")
    
#     # Đường dẫn đến các mô hình
#     phase1_path = f"models/agent_cycle{cycle_num}_agent{agent_idx}_phase1.pt"
#     phase2_path = f"models/agent_cycle{cycle_num}_agent{agent_idx}_best.pt"
    
#     # Khởi tạo môi trường đánh giá
#     env_for_eval = OSMGraphEnv(G, *sample_start_goal(G), global_bounds)
    
#     # Đánh giá mô hình phase 1
#     phase1_perf = None
#     phase1_model = None
#     if os.path.exists(phase1_path):
#         phase1_agent = DQNAgent(state_size=STATE_SIZE, action_size=env_for_eval.action_space.n)
#         phase1_agent.load(phase1_path)
#         phase1_agent.update_target_model()
#         phase1_perf, phase1_success, _ = evaluate_agent(phase1_agent, env_for_eval, episodes=10)
#         phase1_model = phase1_agent.model
#         print(f"[EVAL] Phase 1 | Cycle {cycle_num} | Agent {agent_idx} | Perf = {phase1_perf:.2f} | Success = {phase1_success:.2f}")
#     else:
#         print(f"[EVAL] Phase 1 model not found: {phase1_path}")
    
#     # Đánh giá mô hình phase 2 (final)
#     phase2_perf = None
#     phase2_model = None
#     if os.path.exists(phase2_path):
#         phase2_agent = DQNAgent(state_size=STATE_SIZE, action_size=env_for_eval.action_space.n)
#         phase2_agent.load(phase2_path)
#         phase2_agent.update_target_model()
#         phase2_perf, phase2_success, _ = evaluate_agent(phase2_agent, env_for_eval, episodes=10)
#         phase2_model = phase2_agent.model
#         print(f"[EVAL] Phase 2 | Cycle {cycle_num} | Agent {agent_idx} | Perf = {phase2_perf:.2f} | Success = {phase2_success:.2f}")
#     else:
#         print(f"[EVAL] Phase 2 model not found: {phase2_path}")
    
#     # Xác định mô hình tốt nhất để sử dụng cho federated learning
#     if phase1_perf is not None and phase2_perf is not None:
#         if phase2_perf >= phase1_perf:
#             print(f"[SELECT] Using Phase 2 model (better performance)")
#             return phase1_perf, phase2_perf, phase2_model
#         else:
#             print(f"[SELECT] Using Phase 1 model (better performance)")
#             return phase1_perf, phase2_perf, phase1_model
#     elif phase1_perf is not None:
#         print(f"[SELECT] Using Phase 1 model (only available)")
#         return phase1_perf, None, phase1_model
#     elif phase2_perf is not None:
#         print(f"[SELECT] Using Phase 2 model (only available)")
#         return None, phase2_perf, phase2_model
#     else:
#         print(f"[ERROR] No models found for Cycle {cycle_num} Agent {agent_idx}")
#         return None, None, None


# def main():
#     os.makedirs("training_progress", exist_ok=True)
#     os.makedirs("training_progress/paths", exist_ok=True)
#     plt.ioff()

#     place_names = [
#         'District 1, Ho Chi Minh City, Vietnam',
#         'District 3, Ho Chi Minh City, Vietnam',
#         'District 4, Ho Chi Minh City, Vietnam',
#     ]
#     eval_districts = [
#         'District 8, Ho Chi Minh City, Vietnam',
#         'District 10, Ho Chi Minh City, Vietnam',
#         'District 12, Ho Chi Minh City, Vietnam',
#     ]

#     # Calculate global bounds for state normalization
#     all_places = place_names + eval_districts + ['District 5, Ho Chi Minh City, Vietnam', 'District 6, Ho Chi Minh City, Vietnam']
#     global_x_min, global_x_max = float('inf'), float('-inf')
#     global_y_min, global_y_max = float('inf'), float('-inf')
#     for place_name in all_places:
#         G_tmp = load_graph(place_name)
#         x_coords = [G_tmp.nodes[n]['x'] for n in G_tmp.nodes]
#         y_coords = [G_tmp.nodes[n]['y'] for n in G_tmp.nodes]
#         global_x_min = min(global_x_min, min(x_coords))
#         global_x_max = max(global_x_max, max(x_coords))
#         global_y_min = min(global_y_min, min(y_coords))
#         global_y_max = max(global_y_max, max(y_coords))
#     global_bounds = (global_x_min, global_x_max, global_y_min, global_y_max)

#     # Federated learning
#     local_models, local_perfs = [], []
#     phase_results = []  # Lưu kết quả của từng phase

#     for cycle, place in enumerate(place_names):
#         safe_name = slugify(place)
#         cache_file = f"cache/{safe_name}.pkl"
#         if os.path.exists(cache_file):
#             with open(cache_file, 'rb') as f:
#                 G = pickle.load(f)
#             print(f"[Cycle {cycle+1}] Loaded graph for {place} ({G.number_of_nodes()} nodes)")
#         else:
#             G = load_graph(place)
#             os.makedirs("cache", exist_ok=True)
#             with open(cache_file, 'wb') as f:
#                 pickle.dump(G, f)
#             print(f"[Cycle {cycle+1}] Loaded graph for {place} ({G.number_of_nodes()} nodes)")

#         for agent_idx in range(3):
#             phase1_path = f"models/agent_cycle{cycle+1}_agent{agent_idx}_phase1.pt"
#             phase2_path = f"models/agent_cycle{cycle+1}_agent{agent_idx}_best.pt"

#             # Chỉ train nếu cả hai model chưa tồn tại
#             if not os.path.exists(phase1_path) and not os.path.exists(phase2_path):
#                 print(f"[TRAIN] ➤ Training Agent {agent_idx} in Cycle {cycle+1}")
#                 start, goal = sample_start_goal(G, min_dist=800, max_dist=4000, force_far=False)
#                 env = OSMGraphEnv(G, start, goal, global_bounds)
#                 agent = DQNAgent(state_size=STATE_SIZE, action_size=env.action_space.n)

#                 train_cycle(env, agent, episodes=400, max_steps=700,
#                             cycle_num=cycle+1, graph=G,
#                             start=start, goal=goal, agent_idx=agent_idx)
#             else:
#                 print(f"[SKIP] ✅ Found pre-trained models for Cycle {cycle+1} Agent {agent_idx}")

#             # Đánh giá cả hai phase và chọn mô hình tốt nhất
#             phase1_perf, phase2_perf, best_model = evaluate_phases(place, agent_idx, cycle+1, global_bounds)
            
#             # Lưu kết quả đánh giá
#             phase_results.append({
#                 'cycle': cycle+1,
#                 'agent': agent_idx,
#                 'place': place,
#                 'phase1_perf': phase1_perf,
#                 'phase2_perf': phase2_perf
#             })
            
#             # Nếu có mô hình tốt nhất, thêm vào danh sách cho federated averaging
#             if best_model is not None:
#                 local_perfs.append(phase2_perf if phase2_perf is not None else phase1_perf)
#                 local_models.append(best_model)
#                 print(f"[FEDERATED] Added model from Cycle {cycle+1} Agent {agent_idx} to federated pool")

#     # Lưu kết quả phase để phân tích
#     import pandas as pd
#     phase_df = pd.DataFrame(phase_results)
#     os.makedirs("results", exist_ok=True)
#     phase_df.to_csv("results/phase_comparison.csv", index=False)
#     print("✅ Saved phase comparison results to results/phase_comparison.csv")

#     # Lọc models theo action_size thống nhất
#     if local_models:
#         local_models, local_perfs, chosen_action_size = filter_models_by_action_size(local_models, local_perfs)

#         # Tiến hành federated averaging
#         weights = (np.array(local_perfs) - np.min(local_perfs)) / (np.ptp(local_perfs) + 1e-10)
#         weights /= weights.sum()
#         global_model = DQNNetwork(state_size=STATE_SIZE, action_size=chosen_action_size)
#         global_model = federated_averaging(global_model, local_models, weights)

#         # Distillation phase
#         G_distill = load_graph(place_name='District 5, Ho Chi Minh City, Vietnam')
#         s, g = sample_start_goal(G_distill, min_dist=500, max_dist=5000, force_far=True)
#         global_model = knowledge_distillation(global_model, local_models, G_distill, global_bounds)

#         # Fine-tuning on a new district
#         place = 'District 6, Ho Chi Minh City, Vietnam'
#         safe_name_ft = slugify(place)
#         cache_file = f"cache/{safe_name_ft}.pkl"
#         if os.path.exists(cache_file):
#             with open(cache_file, 'rb') as f:
#                 G_ft = pickle.load(f)
#             print(f"[Fine-tuning] Loaded graph for {place} ({G_ft.number_of_nodes()} nodes)")
#         else:
#             G_ft = load_graph(place)
#             with open(cache_file, 'wb') as f:
#                 pickle.dump(G_ft, f)
#             print(f"[Fine-tuning] Created and loaded graph for {place} ({G_ft.number_of_nodes()} nodes)")
            
#         s, g = sample_start_goal(G_ft, min_dist=800, max_dist=4000, force_far=False)
#         env_ft = OSMGraphEnv(G_ft, s, g, global_bounds)
#         fine_agent = DQNAgent(state_size=STATE_SIZE, action_size=env_ft.action_space.n)
#         fine_agent.model.load_state_dict(global_model.state_dict())
#         fine_agent.update_target_model()
#         train_cycle(env_ft, fine_agent, episodes=400, max_steps=700,
#                     cycle_num=4, graph=G_ft, start=s, goal=g, agent_idx=0)

#         # Final evaluation on unseen districts
#         final_results = []
#         for place in eval_districts:
#             safe_name_eval = slugify(place)
#             cache_file = f"cache/{safe_name_eval}.pkl"
#             if os.path.exists(cache_file):
#                 with open(cache_file, 'rb') as f:
#                     G_final = pickle.load(f)
#                 print(f"[Eval] Loaded graph for {place} ({G_final.number_of_nodes()} nodes)")
#             else:
#                 G_final = load_graph(place)
#                 with open(cache_file, 'wb') as f:
#                     pickle.dump(G_final, f)
#                 print(f"[Eval] Created and loaded graph for {place} ({G_final.number_of_nodes()} nodes)")

#             s, g = sample_start_goal(G_final, min_dist=800, max_dist=4000, force_far=False)
#             env_final = OSMGraphEnv(G_final, s, g, global_bounds)
#             perf, success_rate, paths = evaluate_agent(fine_agent, env_final, episodes=10, start=s, goal=g)
            
#             final_results.append({
#                 'place': place,
#                 'performance': perf,
#                 'success_rate': success_rate
#             })
            
#             print(f"[FINAL EVAL] {place} | Performance: {perf:.2f} | Success Rate: {success_rate:.2f}")

#         # Lưu kết quả đánh giá cuối cùng
#         final_df = pd.DataFrame(final_results)
#         final_df.to_csv("results/final_evaluation.csv", index=False)
#         print("✅ Saved final evaluation results to results/final_evaluation.csv")

#         os.makedirs("models", exist_ok=True)
#         fine_agent.save("models/global_model.pt")
#         print("✅ Saved global model to models/global_model.pt")
#     else:
#         print("[ERROR] No valid models found for federated averaging!")


# if __name__ == "__main__":
#     main()

import os
import matplotlib.pyplot as plt
from config import *
from env import OSMGraphEnv
from agent import DQNAgent, DQNNetwork
from train import train_cycle, evaluate_agent
from federated import federated_averaging, knowledge_distillation
from utils import load_graph, sample_start_goal
import numpy as np
import pickle
from slugify import slugify

def filter_models_by_action_size(models, perfs):
    from collections import defaultdict
    groups = defaultdict(list)
    for i, m in enumerate(models):
        action_size = m.net[-1].out_features
        groups[action_size].append((m, perfs[i]))
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
            os.makedirs("cache", exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(G, f)
            print(f"[Cycle {cycle+1}] Loaded graph for {place} ({G.number_of_nodes()} nodes)")

        for agent_idx in range(3):
            model_path = f"models/agent_cycle{cycle+1}_agent{agent_idx}_best.pt"
            if not os.path.exists(model_path):
                print(f"[TRAIN] ➤ Training Agent {agent_idx} in Cycle {cycle+1}")
                start, goal = sample_start_goal(G, min_dist=800, max_dist=4000, force_far=False)
                env = OSMGraphEnv(G, start, goal, global_bounds)
                agent = DQNAgent(state_size=STATE_SIZE, action_size=env.action_space.n)

                train_cycle(env, agent, episodes=400, max_steps=700,
                            cycle_num=cycle+1, graph=G, agent_idx=agent_idx)
            else:
                print(f"[SKIP] ✅ Found pre-trained model: {model_path}")

            agent = DQNAgent(state_size=STATE_SIZE, action_size=env.action_space.n)
            agent.load(model_path)
            agent.update_target_model()
            perf, success_rate, _ = evaluate_agent(agent, env)
            local_models.append(agent.model)
            local_perfs.append(perf)

    import pandas as pd
    os.makedirs("results", exist_ok=True)

    if local_models:
        local_models, local_perfs, chosen_action_size = filter_models_by_action_size(local_models, local_perfs)

        weights = (np.array(local_perfs) - np.min(local_perfs)) / (np.ptp(local_perfs) + 1e-10)
        weights /= weights.sum()
        global_model = DQNNetwork(state_size=STATE_SIZE, action_size=chosen_action_size)
        global_model = federated_averaging(global_model, local_models, weights)

        G_distill = load_graph(place_name='District 5, Ho Chi Minh City, Vietnam')
        s, g = sample_start_goal(G_distill, min_dist=500, max_dist=5000, force_far=True)
        global_model = knowledge_distillation(global_model, local_models, G_distill, global_bounds)

        place = 'District 6, Ho Chi Minh City, Vietnam'
        safe_name_ft = slugify(place)
        cache_file = f"cache/{safe_name_ft}.pkl"
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                G_ft = pickle.load(f)
        else:
            G_ft = load_graph(place)
            with open(cache_file, 'wb') as f:
                pickle.dump(G_ft, f)

        s, g = sample_start_goal(G_ft, min_dist=800, max_dist=4000, force_far=False)
        env_ft = OSMGraphEnv(G_ft, s, g, global_bounds)
        fine_agent = DQNAgent(state_size=STATE_SIZE, action_size=env_ft.action_space.n)
        fine_agent.model.load_state_dict(global_model.state_dict())
        fine_agent.update_target_model()
        train_cycle(env_ft, fine_agent, episodes=400, max_steps=700,
                    cycle_num=4, graph=G_ft, agent_idx=0)

        final_results = []
        for place in eval_districts:
            safe_name_eval = slugify(place)
            cache_file = f"cache/{safe_name_eval}.pkl"
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    G_final = pickle.load(f)
            else:
                G_final = load_graph(place)
                with open(cache_file, 'wb') as f:
                    pickle.dump(G_final, f)

            s, g = sample_start_goal(G_final, min_dist=800, max_dist=4000, force_far=False)
            env_final = OSMGraphEnv(G_final, s, g, global_bounds)
            perf, success_rate, paths = evaluate_agent(fine_agent, env_final, episodes=10, start=s, goal=g)

            final_results.append({
                'place': place,
                'performance': perf,
                'success_rate': success_rate
            })

            print(f"[FINAL EVAL] {place} | Performance: {perf:.2f} | Success Rate: {success_rate:.2f}")

        final_df = pd.DataFrame(final_results)
        final_df.to_csv("results/final_evaluation.csv", index=False)
        print("✅ Saved final evaluation results to results/final_evaluation.csv")

        os.makedirs("models", exist_ok=True)
        fine_agent.save("models/global_model.pt")
        print("✅ Saved global model to models/global_model.pt")
    else:
        print("[ERROR] No valid models found for federated averaging!")

if __name__ == "__main__":
    main()