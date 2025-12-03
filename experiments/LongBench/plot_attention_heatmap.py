import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

def load_and_process_data(data_dir, layer_idx, num_bins=1000, max_ratio=1.0):
    """
    读取数据并计算每个 Head 的平均累积注意力分布 (CDF)。
    """
    json_files = glob.glob(os.path.join(data_dir, "*_sample_*.json"))
    
    if not json_files:
        print(f"Error: No JSON files found in {data_dir}")
        return None

    print(f"Found {len(json_files)} sample files. Processing Layer {layer_idx}...")

    # 存储累加后的 CDF 矩阵 [num_heads, num_bins]
    accumulated_cdf = None
    sample_count = 0
    
    # 定义统一的 X 轴网格 (0% -> 100%)
    x_grid = np.linspace(0, 1.0, num_bins)

    for file_path in tqdm(json_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # JSON 中 key 是字符串，需要转换
            layer_key = str(layer_idx)
            if layer_key not in data['scores']:
                continue

            # scores shape: [num_heads, seq_len]
            # 注意：如果之前用了 pooling='none'，这里的 seq_len 就是历史长度
            scores = np.array(data['scores'][layer_key])
            
            num_heads, seq_len = scores.shape
            
            # 初始化累加器
            if accumulated_cdf is None:
                accumulated_cdf = np.zeros((num_heads, num_bins))

            # 处理每一个 Head
            for h in range(num_heads):
                head_scores = scores[h]
                
                # 1. 排序 (从大到小)
                # 我们关心的是：最重要的 Token 占据了多少比例
                sorted_scores = np.sort(head_scores)[::-1]
                
                # 2. 计算累积和 (CDF)
                cdf = np.cumsum(sorted_scores)
                
                # 3. 归一化 CDF (确保最大值为 1，消除精度误差或 GQA 聚合带来的数值波动)
                if cdf[-1] > 0:
                    cdf = cdf / cdf[-1]
                
                # 4. 准备插值坐标
                # 原数据的 X 坐标：0/N, 1/N, ..., (N-1)/N
                original_x = np.arange(seq_len) / max(1, seq_len - 1)
                
                # 5. 插值到统一网格
                interpolated_cdf = np.interp(x_grid, original_x, cdf)
                
                # 累加
                accumulated_cdf[h] += interpolated_cdf
            
            sample_count += 1

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    if sample_count == 0:
        print("No valid data processed.")
        return None

    # 计算平均值
    average_cdf = accumulated_cdf / sample_count
    
    # 截断到 max_ratio (例如只看前 50%)
    bin_limit = int(num_bins * max_ratio)
    return average_cdf[:, :bin_limit], x_grid[:bin_limit]

def plot_heatmap(matrix, x_grid, layer_idx, output_path):
    """
    绘制热力图
    """
    plt.figure(figsize=(10, 6))
    
    # 使用 'hot' colormap (黑->红->黄->白) 对应论文视觉效果
    # 设置 vmin=0.5, vmax=1.0
    img = plt.imshow(
        matrix, 
        aspect='auto', 
        cmap='hot', 
        interpolation='nearest',
        extent=[0, x_grid[-1], matrix.shape[0], 0],
        vmin=0.5, vmax=1.0
    )
    
    # 复刻论文图例：水平放置，特定刻度
    cbar = plt.colorbar(img, orientation='horizontal', fraction=0.046, pad=0.15)
    cbar.set_label('Aggregated Attention Weight', labelpad=-40, y=1.5) # 调整 Label 到 bar 上方
    cbar.set_ticks([0.50, 0.62, 0.75, 0.88, 1.00]) # 论文中的刻度
    cbar.ax.xaxis.set_ticks_position('bottom')
    
    plt.title(f"Attention Concentration - Layer {layer_idx}")
    plt.xlabel("Aggregated Top Ratio")
    plt.ylabel("Head Index i")
    
    # 调整 X 轴刻度显示为百分比
    plt.xticks(np.arange(0, x_grid[-1]+0.1, 0.1), [f"{int(x*100)}%" for x in np.arange(0, x_grid[-1]+0.1, 0.1)])
    
    # 调整 Y 轴刻度
    plt.yticks(np.arange(0, matrix.shape[0], 4)) # 每4个Head标一次，避免太拥挤

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing .json sample files")
    parser.add_argument("--layer", type=int, default=20, help="Layer index to visualize")
    parser.add_argument("--out", type=str, default="heatmap.png", help="Output image path")
    parser.add_argument("--max_ratio", type=float, default=0.5, help="Max ratio for X-axis (e.g., 0.5 for 50%)")
    
    args = parser.parse_args()
    
    result = load_and_process_data(args.data_dir, args.layer, max_ratio=args.max_ratio)
    
    if result:
        matrix, x_grid = result
        plot_heatmap(matrix, x_grid, args.layer, args.out)