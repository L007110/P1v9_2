import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 配置 ---
LOG_DIR = "training_results"
GNN_LOG = "convergence_GNN.csv"
NO_GNN_LOG = "convergence_NO_GNN.csv"
DQN_LOG = "convergence_DQN.csv"
PLOT_FILE = "convergence_plot.png"
ROLLING_WINDOW = 50  # 平滑窗口大小


# --- 结束配置 ---

def plot_convergence_results():
    print("--- Loading convergence data ---")

    try:
        df_gnn = pd.read_csv(os.path.join(LOG_DIR, GNN_LOG))
        df_no_gnn = pd.read_csv(os.path.join(LOG_DIR, NO_GNN_LOG))
        df_dqn = pd.read_csv(os.path.join(LOG_DIR, DQN_LOG))
    except FileNotFoundError as e:
        print(f"!!! Error: Log file not found. {e}")
        print(
            "Please make sure 'convergence_GNN.csv', 'convergence_NO_GNN.csv', and 'convergence_DQN.csv' are in the 'training_results' directory.")
        return

    # 添加模型标签
    df_gnn['model'] = 'GNN-DRL'
    df_no_gnn['model'] = 'No-GNN DRL'
    df_dqn['model'] = 'Standard DQN'

    # 合并数据
    df = pd.concat([df_gnn, df_no_gnn, df_dqn], ignore_index=True)

    # 计算平滑后的指标
    df['reward_smooth'] = df.groupby('model')['cumulative_reward'].transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())
    df['v2v_success_smooth'] = df.groupby('model')['v2v_success_rate'].transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())
    df['v2i_cap_smooth'] = df.groupby('model')['v2i_sum_capacity'].transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())

    print("Data loaded and smoothed successfully.")

    # --- 开始绘图 ---
    sns.set_theme(style="whitegrid", palette="deep")
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    fig.suptitle(f"Training Convergence Comparison (Smoothed, Window={ROLLING_WINDOW})", fontsize=16, fontweight='bold')

    # --- 图 1: 累积奖励 ---
    ax1 = axes[0]
    sns.lineplot(
        data=df,
        x="epoch",
        y="reward_smooth",
        hue="model",
        style="model",
        linewidth=2.5,
        ax=ax1
    )
    ax1.set_title("Cumulative Reward Convergence", fontsize=14)
    ax1.set_xlabel("Training Epoch", fontsize=12)
    ax1.set_ylabel("Smoothed Cumulative Reward", fontsize=12)
    ax1.legend(title="Model", fontsize=11)

    # --- 图 2: V2V 成功率 ---
    ax2 = axes[1]
    sns.lineplot(
        data=df,
        x="epoch",
        y="v2v_success_smooth",
        hue="model",
        style="model",
        linewidth=2.5,
        ax=ax2
    )
    ax2.set_title("V2V Success Rate Convergence", fontsize=14)
    ax2.set_xlabel("Training Epoch", fontsize=12)
    ax2.set_ylabel("Smoothed V2V Success Rate", fontsize=12)
    ax2.set_ylim(0, 1.05)
    from matplotlib.ticker import PercentFormatter
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.legend(title="Model", fontsize=11)

    # --- 图 3: V2I 和容量 ---
    ax3 = axes[2]
    sns.lineplot(
        data=df,
        x="epoch",
        y="v2i_cap_smooth",
        hue="model",
        style="model",
        linewidth=2.5,
        ax=ax3
    )
    ax3.set_title("V2I Sum Capacity Convergence", fontsize=14)
    ax3.set_xlabel("Training Epoch", fontsize=12)
    ax3.set_ylabel("Smoothed V2I Capacity (Mbps)", fontsize=12)
    ax3.set_ylim(bottom=0)
    ax3.legend(title="Model", fontsize=11)

    # --- 保存 ---
    plt.tight_layout()
    plot_path = os.path.join(LOG_DIR, PLOT_FILE)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    print(f"--- Convergence plot saved to {plot_path} ---")


if __name__ == "__main__":
    plot_convergence_results()