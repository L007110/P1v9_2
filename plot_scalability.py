import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 配置 ---
LOG_DIR = "training_results"
RESULTS_FILE = "scalability_results.csv"
PLOT_FILE = "scalability_plot.png"


# --- 结束配置 ---

def plot_scalability_results():
    print(f"--- Loading results from {LOG_DIR}/{RESULTS_FILE} ---")

    results_path = os.path.join(LOG_DIR, RESULTS_FILE)

    if not os.path.exists(results_path):
        print(f"!!! Error: Results file not found at {results_path}")
        print("Please run Main.py in 'TEST' mode first.")
        return

    try:
        df = pd.read_csv(results_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    print("Results loaded successfully:")
    print(df)

    # 设置绘图风格
    sns.set_theme(style="whitegrid", palette="deep")

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    fig.suptitle("Scalability Performance (GNN vs No-GNN)", fontsize=16, fontweight='bold')

    # --- 图 1: V2V 成功率 vs. 车辆数量 ---
    ax1 = axes[0]
    sns.lineplot(
        data=df,
        x="vehicle_count",
        y="v2v_success_rate",
        hue="model",
        style="model",
        markers=True,
        markersize=10,
        linewidth=2.5,
        ax=ax1
    )
    ax1.set_title("V2V Success Rate vs. Vehicle Density", fontsize=14)
    ax1.set_xlabel("Number of Vehicles", fontsize=12)
    ax1.set_ylabel("V2V Success Rate", fontsize=12)
    ax1.set_ylim(0, 1.05)

    # 设置 Y 轴为百分比
    from matplotlib.ticker import PercentFormatter
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.legend(title="Model", fontsize=11)

    # --- 图 2: V2I 和容量 vs. 车辆数量 ---
    ax2 = axes[1]
    sns.lineplot(
        data=df,
        x="vehicle_count",
        y="v2i_sum_capacity_mbps",
        hue="model",
        style="model",
        markers=True,
        markersize=10,
        linewidth=2.5,
        ax=ax2
    )
    ax2.set_title("V2I Sum Capacity vs. Vehicle Density", fontsize=14)
    ax2.set_xlabel("Number of Vehicles", fontsize=12)
    ax2.set_ylabel("V2I Sum Capacity (Mbps)", fontsize=12)
    ax2.set_ylim(bottom=0)
    ax2.legend(title="Model", fontsize=11)

    ax3 = axes[2]
    sns.lineplot(
        data=df,
        x="vehicle_count",
        y="decision_time_ms",
        hue="model",
        style="model",
        markers=True,
        markersize=10,
        linewidth=2.5,
        ax=ax3
    )
    ax3.set_title("Inference Time vs. Vehicle Density", fontsize=14)
    ax3.set_xlabel("Number of Vehicles", fontsize=12)
    ax3.set_ylabel("Decision Time (ms)", fontsize=12)
    ax3.set_ylim(bottom=0)
    ax3.legend(title="Model", fontsize=11)

    # 保存图表
    plt.tight_layout()
    plot_path = os.path.join(LOG_DIR, PLOT_FILE)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    print(f"--- Scalability plot saved to {plot_path} ---")

    # 自动打开图片 (可选)
    try:
        os.startfile(plot_path)
    except AttributeError:
        # os.startfile is Windows only
        try:
            subprocess.run(['xdg-open', plot_path])  # Linux
        except:
            subprocess.run(['open', plot_path])  # macOS
    except Exception as e:
        print(f"Could not open plot automatically: {e}")


if __name__ == "__main__":
    plot_scalability_results()