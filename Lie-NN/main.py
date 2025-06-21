from math import log, ceil

import numpy as np
from torch import optim
from tqdm import tqdm

import config
import cost_function
import network_model as nm
import plot_average_interation
import plot_loss_vs_interation
import random_circuit as rc
import record_total_itertaion_loss

# import config
problem = config.problem['VQA']
converge_difference = problem['converge_difference']
check_point = problem['check_point']
max_interation = problem['max_interation']
C = problem['C']
n = problem['n']
# num_qubits = problem['num_qubits']
model_list = problem['model_list']
qubits_list = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


def training():
    # loss_value = [[[] for _ in range(n)] for _ in range(len(model_list))]  # record loss value
    final_loss_value = [[[] for _ in range(n)] for _ in range(len(model_list))]  # record the final loss
    # num_interation = [[[] for _ in range(n)] for _ in range(len(model_list))]  # record interation value
    max_num_interation = [[[[] for _ in range(n)] for _ in range(len(model_list))] for _ in qubits_list]
    init_loss = problem['loss_last']
    j = 0  # index of max_num_interation
    for num_qubits in qubits_list:
        loss_value = [[[] for _ in range(n)] for _ in range(len(model_list))]  # record loss value
        num_interation = [[[] for _ in range(n)] for _ in range(len(model_list))]  # record interation value
        depth = ceil((num_qubits ** 2) * log(num_qubits))
        # if num_qubits <= 5:
        #     depth = max(10, 2 * num_qubits)
        # elif num_qubits >= 6:
        #     depth = max(20, int(0.5 * num_qubits ** 1.5))
        # circuit_param = torch.rand(3 * num_qubits + 4 * depth)

        for i in range(n):
            # circuit, para_group_idx = rc.random_circuit(num_qubits, depth)
            # circuit_param = torch.rand(num_qubits + para_group_idx * 4)
            # circuit, param_count = rc.random_circuit(num_qubits, depth)
            circuit, param_count = rc.barren_plateau_circuit(num_qubits, depth)
            circuit_param = np.random.uniform(0, 2 * np.pi, size=param_count)
            for model_idx, model_name in enumerate(model_list):
                # circuit = rc.random_circuit(num_qubits, depth)
                # select model
                if model_name == 'Lie-NGQS':
                    model = nm.Lie_FNN(num_qubits, circuit, circuit_param)
                if model_name == 'LieR-NGQS':
                    model = nm.Lie_RES_FNN(num_qubits, circuit, circuit_param)
                if model_name == 'Res-NGQS':
                    model = nm.NGQS_RES_FNN(num_qubits, circuit, circuit_param)
                if model_name == 'S-NGQS':
                    model = nm.NGQS_FNN(num_qubits, circuit, circuit_param)
                if model_name == 'RQS':
                    model = nm.RQS(num_qubits, circuit, circuit_param)

                # select optimizer
                optimizer = optim.Adam(model.parameters(), lr=0.01)
                # optimizer = optim.AdamW(model.parameters(), lr=0.01)
                # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

                # begin training
                with tqdm(desc="Training", dynamic_ncols=True) as pbar:
                    for interation in range(max_interation):
                        optimizer.zero_grad()
                        probs = model()
                        loss = cost_function.cost_function(probs, num_qubits)
                        loss.backward(retain_graph=True)
                        optimizer.step()
                        ex = loss.item()  # record loss value

                        # update pbar
                        pbar.set_description(
                            f"Qubits: {num_qubits}, Epoch: {interation}, Cost: {ex:.4f}, Model: {model_name}, N:{i + 1}")

                        # every 10 interation recode the interation and loss value
                        if interation % 10 == 0:
                            num_interation[model_idx][i].append(interation)
                            loss_value[model_idx][i].append(ex)

                        # if loss is less than C, break
                        if ex < C:
                            max_num_interation[j][model_idx][i].append(interation)
                            final_loss_value[model_idx][i].append(ex)
                            break

                        # if converged, break
                        if interation % check_point == 0:
                            distance = abs(init_loss - ex)
                            if distance < converge_difference:
                                max_num_interation[j][model_idx][i].append(interation)
                                final_loss_value[model_idx][i].append(ex)
                                break
                            else:
                                init_loss = ex

                        # if interation is greater than max_interation, break
                        if interation > max_interation:
                            max_num_interation[j][model_idx][i].append(interation)
                            final_loss_value[model_idx][i].append(ex)
                            break

            del model
        # record the result
        record_total_itertaion_loss.record_values(loss_value, num_qubits, 'loss', model_list, depth)
        record_total_itertaion_loss.record_values(num_interation, num_qubits, 'interation', model_list, depth)
        # plot the picture of loss vs. interation
        plot_loss_vs_interation.draw_avg_loss_vs_iteration(loss_value, num_interation, num_qubits, model_list,
                                                           'N^2logN')
        j += 1
        del loss_value
        del num_interation
    # plot average interation
    plot_average_interation.plot_average_iterations(qubits_list, model_list, max_num_interation)


if __name__ == '__main__':
    training()
