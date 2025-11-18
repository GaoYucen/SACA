import torch
import numpy as np
from AM import AttentionRouteModel, calculate_avg_passenger_distance, evaluate_model_cost
from AM import load_supervised_data
import os, pathlib
BASEPATH = pathlib.Path(__file__).resolve().parent

def eval():
    # 加载模型，评估效果
    EMBEDDING_DIM = 64
    N_HEADS = 8
    N_LAYERS = 3
    LOAD_PATH = BASEPATH / "model" / "best_model.pth"
    EVALBATCH_SIZE = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AttentionRouteModel(EMBEDDING_DIM, N_HEADS, N_LAYERS).to(device)
    model.load_state_dict(torch.load(LOAD_PATH))
    model.eval()

    file_paths = [f"C:\\Users\\Administrator\\Desktop\\TSC_trans\\SCAC\\dataset\\supervised_dataset_{n}_stations.json"
                    for n in range(5, 16)]
    data = load_supervised_data(file_paths)

    STATS_PATH = BASEPATH / "model" / "normalization_stats.pt"
    stats = torch.load(STATS_PATH)

    abs_errors, approx_ratios = evaluate_model_cost(
        model, data, EVALBATCH_SIZE, device, stats
    )

def presingle(s):
    # 加载模型，单条样本测试
    EMBEDDING_DIM = 64
    N_HEADS = 8
    N_LAYERS = 3
    LOAD_PATH = BASEPATH / "model" / "best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AttentionRouteModel(EMBEDDING_DIM, N_HEADS, N_LAYERS).to(device)
    model.load_state_dict(torch.load(LOAD_PATH))
    model.eval()

    # 单条样本测试
    sample = s

    STATS_PATH = BASEPATH / "model" / "normalization_stats.pt"
    stats = torch.load(STATS_PATH)

    # 构造张量
    loc    = torch.tensor(sample["loc"], device=device).unsqueeze(0)      # (1,N,2)
    start  = torch.tensor(sample["start"], device=device).unsqueeze(0)    # (1,2)
    weight = torch.tensor(sample["weight"], device=device).unsqueeze(0)   # (1,N)

    # 归一化
    mean = stats['mean'].to(device)
    std = stats['std'].to(device)
    loc_norm = (loc - mean[None, None, :2]) / std[None, None, :2]
    weight_norm = (weight - mean[2]) / std[2]
    start_norm = (start - mean[None, :2]) / std[None, :2]

    with torch.no_grad():
        route_all_indices, log_probs = model(loc_norm, start_norm, weight_norm)   # (1,N)

    # 转为局部顺序（0..N-1），减去1
    pred_local_order = (route_all_indices.squeeze(0).cpu().numpy() - 1).tolist()

    # 计算平均乘客距离作为成本
    avg_dist = calculate_avg_passenger_distance(
        route_all_indices, loc, start, weight, device
    ).item()

    print("预测的局部访问顺序:", pred_local_order)           # 与 loc 的行对应
    print("log_probs:", log_probs.squeeze(0).cpu().numpy().round(3).tolist())
    print(f"平均乘客乘坐距离成本: {avg_dist:.4f} km")

    # 若需要查看按照该顺序的坐标
    ordered_coords = sample["loc"][pred_local_order]
    print("按预测顺序的坐标:\n", ordered_coords)

if __name__ == "__main__":
    # eval()

    sample = {
    "loc": np.array([[104.04413, 30.70977], [104.103498, 30.706014], [104.06094, 30.6717], [104.075863, 30.668877], [104.141627, 30.628662], [104.08916, 30.68314], [104.13192, 30.7533], [103.95682, 30.57135], [103.9764, 30.74614], [104.10869, 30.674], [103.99295, 30.67061], [104.02119, 30.64567], [104.04924, 30.68836], [104.05706, 30.7635], [104.01371, 30.70523], [104.08009, 30.65479], [104.02908, 30.68183], [104.06028, 30.57539], [104.06776, 30.61859]], dtype=np.float32),
    "weight": np.array([4, 4, 2, 3, 2, 1, 1, 5, 2, 0, 2, 1, 3, 5, 4, 2, 3, 4, 2], dtype=np.float32),
    "start": np.array([104.44473, 30.323036], dtype=np.float32),
    }
    presingle(sample)