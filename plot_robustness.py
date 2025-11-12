import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


LOG_DIR = "training_results"
ROBUSTNESS_FILE = "robustness_vs_speed_results.csv"
POLICY_FILE = "policy_analysis_data.csv"
PLOT_FILE_ROBUSTNESS = "robustness_vs_speed_plot.png"
PLOT_FILE_POLICY = "policy_analysis_plot.png"


# plot_robustness (1x3 布局)
def plot_robustness():
    print(f"--- Loading robustness results from {LOG_DIR}/{ROBUSTNESS_FILE} ---")
    results_path = os.path.join(LOG_DIR, ROBUSTNESS_FILE)

    if not os.path.exists(results_path):
        print(f"!!! Error: Results file not found. Run 'test_robustness.py' first.")
        return

    df = pd.read_csv(results_path)

    sns.set_theme(style="whitegrid", palette="deep")
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    fig.suptitle("Robustness to Vehicle Speed (Trained at 60 km/h)", fontsize=16, fontweight='bold')

    # 图 1: V2V 成功率 vs. 速度
    ax1 = axes[0]
    sns.lineplot(
        data=df, x="speed_kmh", y="v2v_success_rate", hue="model",
        style="model", markers=True, markersize=10, linewidth=2.5, ax=ax1
    )
    ax1.set_title("V2V Success Rate vs. Vehicle Speed", fontsize=14)
    ax1.set_xlabel("Vehicle Speed (km/h)", fontsize=12)
    ax1.set_ylabel("V2V Success Rate", fontsize=12)
    ax1.set_ylim(0.8, 1.05)
    from matplotlib.ticker import PercentFormatter
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.legend(title="Model", fontsize=11)
    ax1.grid(True, linestyle='--')

    # 图 2: V2I 和容量 vs. 速度
    ax2 = axes[1]
    sns.lineplot(
        data=df, x="speed_kmh", y="v2i_sum_capacity_mbps", hue="model",
        style="model", markers=True, markersize=10, linewidth=2.5, ax=ax2
    )
    ax2.set_title("V2I Sum Capacity vs. Vehicle Speed", fontsize=14)
    ax2.set_xlabel("Vehicle Speed (km/h)", fontsize=12)
    ax2.set_ylabel("V2I Sum Capacity (Mbps)", fontsize=12)
    ax2.set_ylim(bottom=0)
    ax2.legend(title="Model", fontsize=11)
    ax2.grid(True, linestyle='--')

    #  图 3: P95 延迟 vs. 速度
    ax3 = axes[2]
    sns.lineplot(
        data=df, x="speed_kmh", y="p95_delay_ms", hue="model",
        style="model", markers=True, markersize=10, linewidth=2.5, ax=ax3
    )
    ax3.set_title("V2V P95 Latency vs. Vehicle Speed", fontsize=14)
    ax3.set_xlabel("Vehicle Speed (km/h)", fontsize=12)
    ax3.set_ylabel("P95 Delay (ms)", fontsize=12)
    ax3.set_ylim(bottom=0)
    ax3.legend(title="Model", fontsize=11)
    ax3.grid(True, linestyle='--')


    plt.tight_layout()
    plot_path = os.path.join(LOG_DIR, PLOT_FILE_ROBUSTNESS)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"--- Robustness plot saved to {plot_path} ---")
    plt.close()


def plot_policy():
    print(f"--- Loading policy data from {LOG_DIR}/{POLICY_FILE} ---")
    policy_path = os.path.join(LOG_DIR, POLICY_FILE)

    if not os.path.exists(policy_path):
        print(f"!!! Error: Policy file not found. Run 'test_robustness.py' first.")
        return

    df = pd.read_csv(policy_path)

    # 将 SNR 分箱
    snr_bins = [-100, 0, 5, 10, 15, 100]
    snr_labels = ["< 0 dB", "0-5 dB", "5-10 dB", "10-15 dB", "> 15 dB"]
    df['snr_bin'] = pd.cut(df['snr_dB'], bins=snr_bins, labels=snr_labels, right=False)
    # 功率等级从 0-9 映射到 0.1-1.0
    df['power_ratio'] = (df['power_level'] + 1) / 10.0

    sns.set_theme(style="whitegrid", palette="viridis")

    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)  # 宽度从20改为24
    fig.suptitle("Policy Analysis: Chosen Power Ratio vs. Previous V2V SNR", fontsize=16, fontweight='bold')

    df_gnn = df[df['model'] == 'GNN-DRL']
    if not df_gnn.empty:
        sns.violinplot(
            data=df_gnn,
            x="snr_bin",
            y="power_ratio",
            ax=axes[0],
            inner="quartile",
            scale="width",
            order=snr_labels
        )
        axes[0].set_title("Policy: GNN-DRL", fontsize=14)
        axes[0].set_xlabel("Previous V2V SNR", fontsize=12)
        axes[0].set_ylabel("Chosen Power Ratio", fontsize=12)
        axes[0].set_ylim(0, 1.1)
    else:
        axes[0].set_title("Policy: GNN-DRL (No Data)", fontsize=14)

    df_no_gnn = df[df['model'] == 'No-GNN DRL']
    if not df_no_gnn.empty:
        sns.violinplot(
            data=df_no_gnn,
            x="snr_bin",
            y="power_ratio",
            ax=axes[1],
            inner="quartile",
            scale="width",
            order=snr_labels
        )
        axes[1].set_title("Policy: No-GNN DRL", fontsize=14)
        axes[1].set_xlabel("Previous V2V SNR", fontsize=12)
        axes[1].set_ylabel("")
        axes[1].set_ylim(0, 1.1)
    else:
        axes[1].set_title("Policy: No-GNN DRL (No Data)", fontsize=14)

    df_dqn = df[df['model'] == 'Standard DQN']
    if not df_dqn.empty:
        sns.violinplot(
            data=df_dqn,
            x="snr_bin",
            y="power_ratio",
            ax=axes[2],
            inner="quartile",
            scale="width",
            order=snr_labels
        )
        axes[2].set_title("Policy: Standard DQN", fontsize=14)
        axes[2].set_xlabel("Previous V2V SNR", fontsize=12)
        axes[2].set_ylabel("")
        axes[2].set_ylim(0, 1.1)
    else:
        axes[2].set_title("Policy: Standard DQN (No Data)", fontsize=14)

    plt.tight_layout()
    plot_path_combined = os.path.join(LOG_DIR, PLOT_FILE_POLICY)
    plt.savefig(plot_path_combined, dpi=300, bbox_inches='tight')
    print(f"--- Combined Policy plot saved to {plot_path_combined} ---")
    plt.close()


if __name__ == "__main__":
    plot_robustness()
    plot_policy()