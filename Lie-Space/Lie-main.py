from math import ceil, log

from torch import optim
from tqdm import tqdm

import config
import cost_function as cf
import model as m
import quantum_circuit as qc
import random_state as rs
import plot_loss_vs_interation
import plot_average_interation
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
qubits_list = [4, 5, 6, 7, 8, 9, 10]


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

        for i in range(n):
            # circuit, para_group_idx = rc.random_circuit(num_qubits, depth)
            # circuit_param = torch.rand(num_qubits + para_group_idx * 4)
            # circuit, param_count = rc.random_circuit(num_qubits, depth)
            circuit = qc.circuit(num_qubits, depth)
            rand_input_state = rs.random_state(num_qubits)
            for model_idx, model_name in enumerate(model_list):
                init_loss = problem['loss_last']
                # select model
                if model_name == 'LieSpace':
                    model1 = m.LieSpace(num_qubits, circuit, depth, rand_input_state)
                if model_name == 'StoVec':
                    model2 = m.StoVec(num_qubits, circuit, depth, rand_input_state, i)
                    optimizer = optim.Adam(model2.parameters(), lr=0.01)
                if model_name == 'SQC':
                    model = m.SQC(num_qubits, circuit, depth, rand_input_state, i)
                    optimizer = optim.Adam(model.parameters(), lr=0.01)

                # select optimizer
                # optimizer = optim.Adam(model2.parameters(), lr=0.01)
                # optimizer = optim.AdamW(model.parameters(), lr=0.01)
                # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

                # begin training
                with tqdm(desc="Training", dynamic_ncols=True) as pbar:
                    for interation in range(max_interation):
                        if model_name == 'SQC':
                            optimizer.zero_grad()
                            probs = model()
                            loss = cf.cost_function(probs, num_qubits)
                            loss.backward(retain_graph=True)
                            optimizer.step()
                            ex = loss.item()  # record loss value
                        if model_name == 'StoVec':
                            optimizer.zero_grad()
                            probs = model2()
                            loss = cf.cost_function(probs, num_qubits)
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
                if model_name == 'LieSpace':
                    del model1
                elif model_name == 'StoVec':
                    del model2
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
