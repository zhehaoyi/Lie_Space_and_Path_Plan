import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from filterpy.kalman import KalmanFilter
from scipy.signal import savgol_filter, butter, filtfilt


# --------------------------
# 数据读取函数
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
                continue  # 忽略无法解析的行

    max_length = max(len(v) for v in data_dict.values())
    df = pd.DataFrame({key: values + [np.nan] * (max_length - len(values))
                       for key, values in data_dict.items()})
    return df.T.values  # shape: (num_iterations, length_per_iter)


# -------
# 滤波函数
# -------
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def kalman_filter(data, dt=1.0):
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([data[0], 0])  # 初始状态 (位置和速度)
    kf.F = np.array([[1, dt],
                     [0, 1]])  # 状态转移矩阵
    kf.H = np.array([[1, 0]])  # 观测矩阵
    kf.P *= 1000  # 状态协方差矩阵
    kf.R = 0.01  # 观测噪声协方差
    kf.Q = np.array([[0.001, 0],
                     [0, 0.001]])  # 过程噪声协方差

    filtered_data = []
    for z in data:
        kf.predict()
        kf.update(z)
        filtered_data.append(kf.x[0])
    return np.array(filtered_data)


def wavelet_denoising(data, wavelet, level=1):
    # Perform the wavelet decomposition
    coeff = pywt.wavedec(data, wavelet, mode='per')

    # Calculate the threshold for noise reduction
    sigma = np.median(np.abs(coeff[-level]) / 0.6745)
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))

    # Apply the threshold to coefficients
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:])

    # Reconstruct the signal using the modified coefficients
    return pywt.waverec(coeff, wavelet, mode='per')


# --------------------------
# 路径分析函数
# --------------------------
# def analyze_path(dip, dt=10, window_length=21, polyorder=3):
#     dip_smooth = savgol_filter(dip, window_length=window_length, polyorder=polyorder)
#     velocities = [(dip_smooth[i + 1] - dip_smooth[i - 1]) / (2 * dt) for i in range(1, len(dip_smooth) - 1)]
#     accelerations = [(velocities[i + 1] - velocities[i - 1]) / (2 * dt) for i in range(1, len(velocities) - 1)]
#
#     avg_velocity = np.mean([abs(v) for v in velocities])
#     avg_acceleration = np.mean([abs(a) for a in accelerations])
#     kinetic_energy = np.mean([v ** 2 for v in velocities])
#     path_length = sum(abs(dip[i + 1] - dip[i]) for i in range(len(dip) - 1))
#
#     return {
#         "avg_velocity": avg_velocity,
#         "avg_acceleration": avg_acceleration,
#         "kinetic_energy": kinetic_energy,
#         "path_length": path_length,
#         "velocities": velocities,
#         "accelerations": accelerations
#     }


# --------------------------
# 绘图函数
# --------------------------
def analyze_path(dip, dt=10, window_length=21, polyorder=3, fs=30, cutoff=3.667):
    dip_smooth = savgol_filter(dip, window_length=window_length, polyorder=polyorder)
    velocities = [(dip_smooth[i + 1] - dip_smooth[i - 1]) / (2 * dt) for i in range(1, len(dip_smooth) - 1)]

    # 应用低通滤波器
    # filtered_velocities = butter_lowpass_filter(velocities, cutoff=cutoff, fs=20, order=3)
    # filtered_velocities = kalman_filter(velocities)
    filtered_velocities = wavelet_denoising(velocities, wavelet='db4', level=1)
    accelerations = [(filtered_velocities[i + 1] - filtered_velocities[i - 1]) / (2 * dt) for i in
                     range(1, len(filtered_velocities) - 1)]

    avg_velocity = np.mean([abs(v) for v in filtered_velocities])
    avg_acceleration = np.mean([abs(a) for a in accelerations])
    kinetic_energy = np.mean([v ** 2 for v in filtered_velocities])
    path_length = sum(abs(dip[i + 1] - dip[i]) for i in range(len(dip) - 1))

    return {
        "avg_velocity": avg_velocity,
        "avg_acceleration": avg_acceleration,
        "kinetic_energy": kinetic_energy,
        "path_length": path_length,
        "velocities": filtered_velocities,  # 使用过滤后的速度
        "accelerations": accelerations
    }


def plot_velocity_acceleration(results, labels, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    for i, res in enumerate(results):
        plt.figure(figsize=(12, 5))

        # 速度曲线
        plt.subplot(1, 2, 1)
        plt.plot(res["velocities"], label=f'{labels[i]} Velocity', linewidth=1.5)
        plt.title("Velocity Magnitude")
        plt.xlabel("Time Step")
        plt.ylabel("Velocity")
        plt.legend()
        plt.grid(True)
        plt.ylim(-0.015, 0.015)  # 例如只展示 [-0.02, 0.02] 的区间

        # 加速度曲线
        plt.subplot(1, 2, 2)
        plt.plot(res["accelerations"], label=f'{labels[i]} Acceleration', color='orange', linewidth=1.5)
        plt.title("Acceleration Magnitude")
        plt.xlabel("Time Step")
        plt.ylabel("Acceleration")
        plt.legend()
        plt.grid(True)
        plt.ylim(-0.015, 0.015)  # 例如只展示 [-0.02, 0.02] 的区间

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"path_{labels[i]}_{1}.png"))
        plt.close()


# 在主函数中调用plot_velocity_acceleration之前，确保数据已经被充分平滑


# --------------------------
# 主函数：批量处理多个文件
# --------------------------
def process_files(filenames, labels, output_file="analysis_results_{}.txt".format(1),
                  window_length=21, polyorder=3):
    results = []

    with open(output_file, 'w') as f:
        f.write(
            "Filename, Label, AvgVelocity, StdVelocity, AvgAcceleration, StdAcceleration, KineticEnergy, PathLength, RsdVelocity, RsdAcceleration\n")

        for filename, label in zip(filenames, labels):
            print(f"\nProcessing {filename} ({label})...")
            data = read_and_process_data(filename)

            all_avg_velocity = []
            all_avg_acceleration = []
            all_kinetic_energy = []
            all_path_length = []

            for i in range(data.shape[0]):
                dip = data[i]
                analysis = analyze_path(dip, window_length=window_length, polyorder=polyorder)

                all_avg_velocity.append(analysis["avg_velocity"])
                all_avg_acceleration.append(analysis["avg_acceleration"])
                all_kinetic_energy.append(analysis["kinetic_energy"])
                all_path_length.append(analysis["path_length"])

            # 取平均值作为整体路径质量
            avg_velocity = np.mean(all_avg_velocity)
            std_velocity = np.std(all_avg_velocity)
            avg_acceleration = np.mean(all_avg_acceleration)
            std_acceleration = np.std(all_avg_acceleration)
            rsd_velocity = avg_velocity / std_velocity
            rsd_acceleration = avg_acceleration / std_acceleration
            kinetic_energy = np.mean(all_kinetic_energy)
            path_length = np.mean(all_path_length)

            # 写入结果
            f.write(
                f"{filename},{label},{avg_velocity:.6f},{std_velocity:.6f},{avg_acceleration:.6f},{std_acceleration:.6f},"
                f"{kinetic_energy:.6f},{path_length:.6f},{rsd_velocity:.6f},{rsd_acceleration:.6f}\n")

            # 保存分析结果用于绘图
            results.append({
                "label": label,
                "avg_velocity": avg_velocity,
                "std_velocity": std_velocity,
                "avg_acceleration": avg_acceleration,
                "std_acceleration": std_acceleration,
                "kinetic_energy": kinetic_energy,
                "path_length": path_length,
                "velocities": analysis["velocities"],
                "accelerations": analysis["accelerations"]
            })

    return results


# --------------------------
# 主程序入口
# --------------------------
if __name__ == "__main__":
    # 设置参数
    filenames = ["nn_generat_params_log1.txt", "sqc_generat_params_log1.txt"]  # 替换为你的文件名
    labels = ["Path_A", "Path_B"]

    # 执行分析
    results = process_files(filenames, labels, window_length=21, polyorder=3)

    # 可视化
    plot_velocity_acceleration(results, labels)

    print("\n✅ 分析完成！")
    print("📊 图像已保存至 'plots/' 文件夹")
    print("📄 指标已输出至 'analysis_results.txt'")
