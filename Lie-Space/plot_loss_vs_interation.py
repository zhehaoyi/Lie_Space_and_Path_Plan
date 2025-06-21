from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import compute_average_loss

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import numpy as np


def draw_avg_loss_vs_iteration(loss_value, num_iteration, num_qubits, model_list, depth_schema):
    # compute average, max, min iteration list
    avg_list, max_list, min_list, max_len = compute_average_loss.compute_average_loss(loss_value, num_iteration,
                                                                                      model_list)

    # select color
    model_properties = {
        'LieSpace': ('#D62728', '-', '#FF9896'),
        # 'Lie-NGQS': ('#FF7F0E', '-', '#FFBB78'),
        # 'Res-NGQS': ('#1F77B4', '-', '#AEC7E8'),
        # 'S-NGQS': ('#9467BD', '-', '#C5B0D5'),
        'StoVec': ('#25377F', '-', '#9EABD2'),
    }

    # draw main plot
    for model_idx, model in enumerate(model_list):
        color, linestyle, fill_color = model_properties.get(model, (None, None, None))
        if color is None:
            print(f"Warning: Model {model} not recognized.")
            continue

        # draw average line
        plt.plot(max_len[model_idx], avg_list[model_idx], label=model, color=color, linestyle=linestyle, linewidth=3)
        # plt.fill_between(max_len[model_idx], min_list[model_idx], max_list[model_idx], color=fill_color, alpha=0.1)

    plt.grid(axis='y', linestyle='--', color='grey', linewidth=0.5)

    # Adjust the display format of the y-axis scale
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 4))
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.legend(loc='upper right')

    plt.xlabel("iteration")
    plt.ylabel("average Loss")

    # Create a small graph to observe the convergence part
    inset_ax = inset_axes(plt.gca(), width="40%", height="30%", loc='center right', borderpad=0.5)
    for model_idx, model in enumerate(model_list):
        if model not in ['LieSpace']:
            continue

        color, linestyle, fill_color = model_properties.get(model, (None, None, None))
        if color is None:
            continue

        # Find the index of the first value less than 0.05
        threshold_idx = np.argmax(np.array(avg_list[model_idx]) < 0.01)

        if avg_list[model_idx][threshold_idx] >= 0.05:
            continue  # No values less than 0.05, skip this model

        # Plot the convergence part
        inset_ax.plot(max_len[model_idx][threshold_idx:], avg_list[model_idx][threshold_idx:], color=color,
                      linestyle=linestyle, linewidth=1.5)

    output_dir = 'result/Avg_loss_vs_epoch_and_norm_vs_epoch'
    file_path = os.path.join(output_dir, f'Avg_loss_vs_epoch_{num_qubits}qubits_{depth_schema}.pdf')
    os.makedirs(output_dir, exist_ok=True)  # create file
    plt.savefig(file_path)

    # Clear the cache
    plt.close()
