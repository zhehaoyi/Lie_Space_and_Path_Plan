import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import seaborn as sns
from scipy.interpolate import interp1d

sns.set_theme(style="white", context="talk")
# Reset to defaults and set new style
plt.rcdefaults()
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif'],
    'font.size': 12,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 18,
    'text.usetex': False,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'axes.edgecolor': 'black',
    'axes.grid': True,  # Turn on grid by default
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})


# --------------------------
# read the data from the file
# --------------------------
def read_and_process_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    data_dict = {}
    current_key = None
    for line in lines:
        if 'Iteration' in line:
            current_key = int(line.split()[2])
            data_dict[current_key] = []
        elif line.strip() and not line.startswith('---'):
            try:
                data_dict[current_key].append(float(line.strip()))
            except:
                continue

    max_length = max(len(v) for v in data_dict.values())
    df = pd.DataFrame({key: values + [np.nan] * (max_length - len(values))
                       for key, values in data_dict.items()})

    return df.values


# -------
# fliter function
# -------
def wavelet_denoising(data, wavelet, level=1):
    coeff = pywt.wavedec(data, wavelet, mode='per')
    sigma = np.median(np.abs(coeff[-level]) / 0.6745)
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')


# --------------------------
# path analysis
# --------------------------
def analyze_path(dip, dt=5, window_length=21, polyorder=3):
    # dip_smooth = savgol_filter(dip, window_length=window_length, polyorder=polyorder)
    dip_smooth = dip
    velocities = [(dip_smooth[i + 1] - dip_smooth[i - 1]) / (2 * dt) for i in range(1, len(dip_smooth) - 1)]
    filtered_velocities = wavelet_denoising(velocities, wavelet='db4', level=2)
    accelerations = [(velocities[i + 1] - velocities[i - 1]) / (2 * dt) for i in
                     range(1, len(velocities) - 1)]

    avg_velocity = np.mean([abs(v) for v in velocities])
    avg_acceleration = np.mean([abs(a) for a in accelerations])
    kinetic_energy = np.mean([v ** 2 for v in velocities])
    path_length = sum(abs(dip[i + 1] - dip[i]) for i in range(len(dip) - 1))

    return {
        "avg_velocity": avg_velocity,
        "avg_acceleration": avg_acceleration,
        "kinetic_energy": kinetic_energy,
        "path_length": path_length,
        "velocities": velocities,
        "accelerations": accelerations
    }


# --------------------------
# plot velocity and acceleration
# --------------------------
def plot_single_iteration(iteration_data_list, labels, iteration_num, qubits, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')

    colors = ['#2E86AB', '#FB8500', '#52B788', '#FB8500']  # Professional color palette

    # Velocity subplot
    ax = axes[0]
    for i, data in enumerate(iteration_data_list):
        ax.plot(data["velocities"], label=f'{labels[i]}', linewidth=3, color=colors[i % len(colors)])
    ax.set_xlabel("time step", fontweight='bold', fontsize=18)
    ax.set_ylabel("velocity", fontweight='bold', fontsize=18)
    ax.legend(loc="upper right", fontsize=20)
    ax.set_ylim(-0.07, 0.07)
    ax.set_yticks(np.linspace(-0.07, 0.07, 5))
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    # Acceleration subplot
    ax = axes[1]
    for i, data in enumerate(iteration_data_list):
        ax.plot(data["accelerations"], label=f'{labels[i]}', linewidth=3, color=colors[i % len(colors)])
    ax.set_xlabel("time step", fontweight='bold', fontsize=18)
    ax.set_ylabel("acceleration", fontweight='bold', fontsize=18)
    ax.legend(loc="upper right", fontsize=20)
    ax.set_ylim(-0.005, 0.005)
    ax.set_yticks(np.linspace(-0.005, 0.005, 5))
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    base_name = f"Combined_parameter_{iteration_num}"
    output_path = os.path.join(output_dir, f"{base_name}_velocity_acceleration_{qubits}.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_scatter_with_trend_lines(data_list, labels, iteration_num, qubits, title="Scatter Plot with Trend Lines",
                                  method='cubic', num_points=500, output_dir="plots", step=25):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    colors_lines = ['#2E86AB', '#FB8500']
    colors_scatter = ['#A23B72', '#E85D04']
    for i, (y_data, label) in enumerate(zip(data_list, labels)):
        x = np.arange(len(y_data))
        f = interp1d(x, y_data, kind=method)
        x_new = np.linspace(0, len(x) - 1, num_points)
        y_new = f(x_new)

        # Subsample data for scatter points
        x_sparse = x[::step]
        y_sparse = y_data[::step]

        # Trend lines
        # ax.plot(x_new, y_new, linewidth=3, linestyle='-', color=colors_lines[i], label=f'{label} Trend Line', alpha=0.8)
        ax.plot(x_sparse, y_sparse, marker='o', linewidth=3, markersize=8,
                color=colors_lines[i], markerfacecolor=colors_scatter[i], markeredgecolor='black',
                markeredgewidth=1.5, alpha=0.8, label=f'{label} Trend Line')
        # Scatter points
        # ax.scatter(x_sparse, y_sparse, s=60, alpha=0.7, edgecolors='black', color=colors_scatter[i], label=f'{label} Points')
        for j, (x, y) in enumerate(zip(x_sparse, y_sparse)):
            ax.annotate(f'{y:.2f}', (x, y),
                        textcoords="offset points", xytext=(0, 12), ha='center',
                        fontweight='bold', fontsize=11, color=colors_lines[i],
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, edgecolor="none"))

    ax.set_xlabel("index", fontweight='bold', fontsize=20)
    ax.set_ylabel(r"exp($\theta$)", fontweight='bold', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=22)

    # Format Y axis to show powers of 10
    # ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y_axis))

    ax.legend(loc="upper right", fontsize=20)

    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    base_name = f"Combined_Iteration_{iteration_num}_{qubits}"
    output_path = os.path.join(output_dir, f"{base_name}_scatter_with_trend_lines.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def process_files(filenames, labels, qubits, output_suffix="results", window_length=21, polyorder=3):
    all_iterations_data = {label: [] for label in labels}
    all_iterations_raw = {label: [] for label in labels}

    input_base_names = "_".join([os.path.splitext(os.path.basename(fn))[0] for fn in filenames])
    output_file = f"analysis_{input_base_names}_{output_suffix}_{qubits}.csv"

    with open(output_file, 'w') as f:
        # 写表头
        f.write(
            "Filename, Parameters, Label, AvgVelocity, AvgAcceleration, KineticEnergy, PathLength, RsdVelocity, RsdAcceleration\n")

        for filename, label in zip(filenames, labels):
            print(f"\nProcessing {filename} ({label})...")
            data = read_and_process_data(filename)
            all_avg_velocity = []
            all_avg_acceleration = []
            all_kinetic_energy = []
            all_path_length = []

            for i in range(data.shape[0]):
                dip = data[i]
                if label == "Without NN":
                    data_min = min(dip)
                    data_max = max(dip)
                    lower = -1
                    upper = 1
                    temp = (dip - data_min) / (data_max - data_min)
                    dip = lower + (upper - lower) * temp

                analysis = analyze_path(dip, window_length=window_length, polyorder=polyorder)

                exp_dip = np.exp(dip)
                all_iterations_raw[label].append(exp_dip)

                avg_velocity = analysis["avg_velocity"]
                std_velocity = np.std(analysis["velocities"])
                rsd_velocity = std_velocity / avg_velocity
                avg_acceleration = analysis["avg_acceleration"]
                std_acceleration = np.std(analysis["accelerations"])
                rsd_acceleration = std_acceleration / avg_acceleration
                kinetic_energy = analysis["kinetic_energy"]
                path_length = analysis["path_length"]

                # write into csv
                f.write(f"{filename},Parameter_{i + 1},{label},{avg_velocity:.6f},{avg_acceleration:.6f},"
                        f"{kinetic_energy:.6f},{path_length:.6f}, {rsd_velocity:.2f}%, {rsd_acceleration:.2f}%\n")

                # 保存当前迭代数据，供后续绘图使用
                all_iterations_data[label].append({
                    "velocities": analysis["velocities"],
                    "accelerations": analysis["accelerations"]
                })

                all_avg_velocity.append(avg_velocity)
                all_avg_acceleration.append(avg_acceleration)
                all_kinetic_energy.append(kinetic_energy)
                all_path_length.append(path_length)

            # 计算整体统计量
            avg_velocity_total = np.mean(all_avg_velocity)
            std_velocity_total = np.std(all_avg_velocity)
            avg_acceleration_total = np.mean(all_avg_acceleration)
            std_acceleration_total = np.std(all_avg_acceleration)
            rsd_velocity_total = (std_velocity_total / avg_velocity_total) if avg_velocity_total != 0 else np.nan
            rsd_acceleration_total = (
                    std_acceleration_total / avg_acceleration_total) if avg_acceleration_total != 0 else np.nan
            kinetic_energy_total = np.mean(all_kinetic_energy)
            path_length_total = np.mean(all_path_length)

            # 写入汇总行
            f.write(f"{filename},Overall,{label},{avg_velocity_total:.6f},{avg_acceleration_total:.6f},"
                    f"{kinetic_energy_total:.6f},{path_length_total:.6f}\n")
            f.write(f",{label}_std,,{std_velocity_total:.6f},{std_acceleration_total:.6f},,\n")
            f.write(f",{label}_rsd,,{rsd_velocity_total:.2f}%,{rsd_acceleration_total:.2f}%,,\n\n")

    # 在所有文件处理完成后，为每一个 iteration 画图
    num_iterations = len(all_iterations_data[labels[0]])  # 假设所有文件有相同数量的迭代
    for i in range(num_iterations):
        iteration_data_list = [all_iterations_data[label][i] for label in labels]
        plot_single_iteration(iteration_data_list, labels, i + 1, qubits)

    num_iterations = len(all_iterations_raw[labels[0]])
    for i in range(num_iterations):
        combined_data = [all_iterations_raw[label][i] for label in labels]
        plot_scatter_with_trend_lines(combined_data, labels, i + 1, qubits)

    return all_iterations_data


if __name__ == "__main__":
    filenames = ["nn_generat_params_log1_9.txt", "sqc_generat_params_log1_9.txt"]  # 修改这个列表即可
    labels = ["With NN", "Without NN"]

    # analysis
    results = process_files(filenames, labels, 9, output_suffix="geodesic_analysis", window_length=21, polyorder=3)

    print("\nAnalysis Complete!")
    print("Store the images in 'plots/' ")
    print(
        f"All index are in the 'analysis_{'_'.join([os.path.splitext(os.path.basename(fn))[0] for fn in filenames])}_geodesic_analysis.csv'")
