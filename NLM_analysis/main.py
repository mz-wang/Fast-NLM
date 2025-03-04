import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_psnr_ssim(df, x_field, curve_field):
    """
    绘制 PSNR 和 SSIM 的变化曲线，分析 x_field 对 PSNR/SSIM 的影响，
    其中不同的 curve_field 取值对应不同曲线。

    参数：
    - df: DataFrame，包含数据
    - x_field: 在 X 轴上变化的字段，比如 "H"、"Patch"、"Search"
    - curve_field: 用来区分不同曲线的字段，比如 "SR"、"RF"

    使用示例：
    plot_psnr_ssim(df, x_field="H", curve_field="SR")
    """
    # 确保字段有效
    if x_field not in df.columns or curve_field not in df.columns:
        print(f"Error: '{x_field}' 或 '{curve_field}' 不在数据列中！")
        print(f"可用列: {df.columns}")
        return

    # 获取 curve_field 的所有唯一值（不同曲线）
    curve_values = df[curve_field].unique()

    # 颜色映射
    colors = plt.cm.binary(np.linspace(0, 1, len(curve_values)))

    # 创建画布
    fig, ax1 = plt.subplots(figsize=(8, 5))

    for i, curve_val in enumerate(curve_values):
        # 选择当前 curve_field（比如 SR = 0.002）
        filtered_df = df[df[curve_field] == curve_val]

        # 按 x_field 分组，计算 PSNR 和 SSIM 的均值
        result = filtered_df.groupby(x_field)[["PSNR", "SSIM"]].mean()

        if result.empty:
            continue

        # 画 PSNR 曲线
        ax1.plot(result.index, result["PSNR"], marker="o", linestyle="-", color=colors[i],
                 label=f"PSNR ({curve_field}={curve_val})")

    # 设置 PSNR 轴
    ax1.set_xlabel(x_field)
    # ax1.set_ylabel("PSNR", color="tab:blue")
    # ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylabel("PSNR")
    ax1.tick_params(axis="y")

    # 创建第二个 y 轴
    ax2 = ax1.twinx()

    for i, curve_val in enumerate(curve_values):
        # 选择当前 curve_field
        filtered_df = df[df[curve_field] == curve_val]
        result = filtered_df.groupby(x_field)[["PSNR", "SSIM"]].mean()

        if result.empty:
            continue

        # 画 SSIM 曲线
        ax2.plot(result.index, result["SSIM"], marker="s", linestyle="dashed", color=colors[i],
                 label=f"SSIM ({curve_field}={curve_val})")

    # 设置 SSIM 轴
    # ax2.set_ylabel("SSIM", color="tab:orange")
    # ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylabel("SSIM")
    ax2.tick_params(axis="y")

    # 添加图例
    ax1.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    ax2.legend(loc="lower left", bbox_to_anchor=(1.05, 0))

    plt.title(f"PSNR and SSIM vs {x_field} (Different {curve_field})")
    fig.tight_layout()
    plt.show()


def plot_psnr_ssim_subplot(df, x_fields, curve_field):
    """
    绘制多个子图，每个子图分析一个 x_field 对 PSNR/SSIM 的影响，
    其中不同的 curve_field 取值对应不同曲线，并使用双 y 轴。

    参数：
    - df: DataFrame，包含数据
    - x_fields: 列表，每个元素是 X 轴上的变量，比如 ["H", "Search", "Patch", "RF"]
    - curve_field: 用来区分不同曲线的字段，比如 "SR"

    使用示例：
    plot_psnr_ssim_subplot(df, x_fields=["H", "Search", "Patch", "RF"], curve_field="SR")
    """
    num_plots = len(x_fields)  # 需要画的子图个数
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2 行 2 列布局
    axes = axes.flatten()  # 展平数组，方便索引

    curve_values = df[curve_field].unique()  # 获取不同曲线的分类
    colors = plt.cm.viridis_r(np.linspace(0, 1, len(curve_values)))  # 颜色映射

    for idx, (ax, x_field) in enumerate(zip(axes, x_fields)):
        if x_field not in df.columns or curve_field not in df.columns:
            print(f"Error: '{x_field}' 或 '{curve_field}' 不在数据列中！")
            continue

        ax2 = ax.twinx()  # 创建第二个 y 轴

        for i, curve_val in enumerate(curve_values):
            filtered_df = df[df[curve_field] == curve_val]
            result = filtered_df.groupby(x_field)[["PSNR", "SSIM"]].mean()

            if result.empty:
                continue

            # PSNR 曲线（左 y 轴）
            ax.plot(result.index, result["PSNR"], marker="o", linestyle="-", color=colors[i],
                    label=f"PSNR ({curve_field}={curve_val})")
            # SSIM 曲线（右 y 轴）
            ax2.plot(result.index, result["SSIM"], marker="s", linestyle="dashed", color=colors[i], alpha=0.7,
                     label=f"SSIM ({curve_field}={curve_val})")

        ax.set_xlabel(x_field)
        ax.set_ylabel("PSNR", color="blue")
        ax2.set_ylabel("SSIM", color="green")
        ax.set_title(f"PSNR & SSIM vs {x_field}")

        # 只在最后一个子图加图例
        if idx == len(axes) - 1:
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_psnr_ssim_subplot_bw(df, x_fields, curve_field):
    """
    绘制多个子图，每个子图分析一个 x_field 对 PSNR/SSIM 的影响，
    其中不同的 curve_field 取值对应不同曲线，并使用双 y 轴。

    参数：
    - df: DataFrame，包含数据
    - x_fields: 列表，每个元素是 X 轴上的变量，比如 ["H", "Search", "Patch", "RF"]
    - curve_field: 用来区分不同曲线的字段，比如 "SR"

    使用示例：
    plot_psnr_ssim_subplot(df, x_fields=["H", "Search", "Patch", "RF"], curve_field="SR")
    """
    num_plots = len(x_fields)  # 需要画的子图个数
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2 行 2 列布局
    axes = axes.flatten()  # 展平数组，方便索引

    curve_values = df[curve_field].unique()  # 获取不同曲线的分类
    colors = plt.cm.binary(np.linspace(0.3, 1, len(curve_values)))  # 颜色映射

    for idx, (ax, x_field) in enumerate(zip(axes, x_fields)):
        if x_field not in df.columns or curve_field not in df.columns:
            print(f"Error: '{x_field}' 或 '{curve_field}' 不在数据列中！")
            continue

        ax2 = ax.twinx()  # 创建第二个 y 轴

        for i, curve_val in enumerate(curve_values):
            filtered_df = df[df[curve_field] == curve_val]
            result = filtered_df.groupby(x_field)[["PSNR", "SSIM"]].mean()

            if result.empty:
                continue

            # PSNR 曲线（左 y 轴）
            ax.plot(result.index, result["PSNR"], marker="o", linestyle="-", color=colors[i],
                    label=f"PSNR ({curve_field}={curve_val})")
            # SSIM 曲线（右 y 轴）
            ax2.plot(result.index, result["SSIM"], marker="s", linestyle="dashed", color=colors[i], alpha=0.7,
                     label=f"SSIM ({curve_field}={curve_val})")

        ax.set_xlabel(x_field)
        ax.set_ylabel("PSNR")
        ax2.set_ylabel("SSIM")
        ax.set_title(f"PSNR & SSIM vs {x_field}")

        # 只在最后一个子图加图例
        if idx == len(axes) - 1:
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 读取数据
    df = pd.read_csv("benchmark_results_1.csv")
    df.columns = df.columns.str.strip()  # 去除列名空格

    # 使用函数：x 轴是 H，不同 SR 画不同曲线
    # plot_psnr_ssim(df, x_field="H", curve_field="SR")
    # plot_psnr_ssim(df, x_field="Search", curve_field="SR")
    # plot_psnr_ssim(df, x_field="Patch", curve_field="SR")
    # plot_psnr_ssim(df, x_field="RF", curve_field="SR")

    plot_psnr_ssim_subplot_bw(df, x_fields=["H", "Search", "Patch", "RF"], curve_field="SR")
    plot_psnr_ssim_subplot_bw(df, x_fields=["SR", "H", "Patch", "RF"], curve_field="Search")
    plot_psnr_ssim_subplot_bw(df, x_fields=["SR", "H", "Search", "RF"], curve_field="Patch")
    plot_psnr_ssim_subplot_bw(df, x_fields=["SR", "H", "Search", "Patch"], curve_field="RF")
    plot_psnr_ssim_subplot_bw(df, x_fields=["SR", "Search", "Patch", "RF"], curve_field="H")


