import numpy as np
import matplotlib.pyplot as plt
import os

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_average_iterations(qubits_list, model_list, max_num_interation):
    # compute the average of each model_idx
    averages = np.zeros((len(model_list), len(qubits_list)))

    for model_idx in range(len(model_list)):
        for qubit_idx in range(len(qubits_list)):
            iterations = max_num_interation[qubit_idx][model_idx]
            if iterations:
                averages[model_idx, qubit_idx] = np.mean(iterations)

    # create a file
    output_dir = 'result/average_interation'
    os.makedirs(output_dir, exist_ok=True)

    # define model properties
    model_properties = {
        'LieSpace': ('#D62728', '-', '#FF9896'),
        # 'Lie-NGQS': ('#FF7F0E', '-', '#FFBB78'),
        # 'Res-NGQS': ('#1F77B4', '-', '#AEC7E8'),
        # 'S-NGQS': ('#9467BD', '-', '#C5B0D5'),
        'StoVec': ('#25377F', '-', '#9EABD2'),
    }

    # plot the image
    for model_idx, model in enumerate(model_list):
        color, linestyle = model_properties.get(model, (None, None))
        if color is not None:
            plt.plot(qubits_list, averages[model_idx], marker='o', color=color, linestyle=linestyle, label=model,
                     linewidth=3)

    plt.xlabel('qubits')
    plt.ylabel('average Iterations')
    plt.xticks(qubits_list)  # 设置x轴刻度为qubits_list中的数字

    # Create a small graph to observe the convergence part
    inset_ax = inset_axes(plt.gca(), width="40%", height="30%", loc='center left', borderpad=0.5)
    for model_idx, model in enumerate(model_list):
        if model not in ['LieSpace']:
            continue

        color, linestyle = model_properties.get(model, (None, None))
        if color is None:
            continue

        inset_ax.plot(qubits_list[:3], averages[model_idx][:3], color=color,
                      linestyle=linestyle,
                      linewidth=1.5)
        inset_ax.set_xticks(qubits_list[:3])

    np.savetxt('result/average_interation/averages.txt', averages)
    np.savetxt('result/average_interation/qubit_list.txt', qubits_list)
    plt.savefig(os.path.join(output_dir, 'average_iterations.pdf'))
    plt.close()


def plot_average_iterations_from_file():
    # 读取数据
    output_dir = 'result/average_interation'
    averages = np.loadtxt(os.path.join(output_dir, 'averages.txt'))
    qubits_list = np.loadtxt(os.path.join(output_dir, 'qubit_list.txt'))
    model_list = ['LieSpace', 'StoVec']

    # 定义模型属性
    model_properties = {
        'LieSpace': ('#D62728', '-', '#FF9896'),
        # 'Lie-NGQS': ('#FF7F0E', '-', '#FFBB78'),
        # 'Res-NGQS': ('#1F77B4', '-', '#AEC7E8'),
        # 'S-NGQS': ('#9467BD', '-', '#C5B0D5'),
        'StoVec': ('#25377F', '-', '#9EABD2'),
    }

    # 绘制图表
    for model_idx, model in enumerate(model_list):
        color, linestyle = model_properties.get(model, (None, None))
        if color is not None:
            plt.plot(qubits_list, averages[model_idx], marker='o', color=color, linestyle=linestyle, label=model,
                     linewidth=3)
    plt.legend()
    plt.xlabel('qubits')
    plt.ylabel('average Iterations')
    plt.xticks(qubits_list)  # 设置x轴刻度为qubits_list中的数字

    # 创建一个小的图表来观察收敛部分
    inset_ax = inset_axes(plt.gca(), width="40%", height="30%", loc='center left', borderpad=3)
    for model_idx, model in enumerate(model_list):
        if model not in ['LieSpace']:
            continue

        color, linestyle = model_properties.get(model, (None, None))
        if color is None:
            continue

        inset_ax.plot(qubits_list[:3], averages[model_idx][:3], color=color,
                      linestyle=linestyle,
                      linewidth=1.5)
        inset_ax.set_xticks(qubits_list[:3])

    plt.savefig(os.path.join(output_dir, 'average_iterations_from_file.pdf'))
    plt.show()  # 显示图表


