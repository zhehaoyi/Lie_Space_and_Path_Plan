import functools
import math

import numpy as np
import tensorcircuit as tc
import torch
import torch.nn as nn

tc.set_backend('pytorch')


# class LieSpace(nn.Module):
#     def __init__(self, num_qubits, circuit, circuit_depth, rand_input_state):
#         super(LieSpace, self).__init__()
#         self.num_qubits = num_qubits  # the number of qubits
#         self.circuit = circuit  # the quantum circuit
#         self.rand_input_state = rand_input_state  # input random quantum state
#         self.idx = 1
#
#         self.circuit_eval = tc.interfaces.torch_interface(self.circuit_eval_probs, jit=True)
#
#         self.gaussvec = nn.Parameter(torch.rand(3 * num_qubits + 5 * circuit_depth))
#
#         self.epsilon = 1e-5
#         self.gradients = []
#
#     def forward(self, loss):
#         if self.idx == 1:
#             probs = self.circuit_eval(self.gaussvec)
#             self.idx += 1
#             return probs
#         else:
#             # eval_grad = K.value_and_grad(loss)
#             # val_grad_jit = K.jit(eval_grad)
#             self.gaussvec = self.gaussvec - 0.01 * self.gradients
#             probs = self.circuit_eval(self.gaussvec)
#             return probs
#
#
#     def circuit_eval_probs(self, stovec):
#         eval_qc = tc.Circuit(self.num_qubits, inputs=self.rand_input_state)
#         idx = 0
#
#         gate_actions = {
#             'cnot': lambda qc, index, _: qc.cnot(index[0], index[1]),
#             'cz': lambda qc, index, _: qc.cz(index[0], index[1]),
#             'ry': lambda qc, index, v: qc.ry(index[0], theta=v),
#             'rz': lambda qc, index, v: qc.rz(index[0], theta=v),
#             'rx': lambda qc, index, v: qc.rx(index[0], theta=v),
#             'rxx': lambda qc, index, v: qc.rxx(index[0], index[1], theta=v)
#         }
#
#         for block in self.circuit.to_qir():
#             gate = block['name']
#             if gate in ['ry', 'rz', 'rx', 'rxx']:
#                 gate_actions[gate](eval_qc, block['index'], stovec[idx])
#                 idx += 1
#             else:
#                 gate_actions[gate](eval_qc, block['index'], None)
#
#         return eval_qc.probability()


class StoVec(nn.Module):
    def __init__(self, num_qubits, circuit, circuit_depth, rand_input_state, i):
        super(StoVec, self).__init__()
        self.num_qubits = num_qubits  # the number of qubits
        self.circuit = circuit  # the quantum circuit
        self.circuit_depth = circuit_depth  # the depth of quantum circuit
        self.rand_input_state = rand_input_state  # the input quantum state of a quantum circuit

        # generate a stochastic vector contains all the parameters the quantum circuit needed
        init_method = functools.partial(nn.init.uniform_, b=2 * math.pi)
        self.stovec = nn.Parameter(init_method(torch.Tensor(1, 4)),
                                   requires_grad=True)
        self.layer1 = nn.Linear(4, 10)
        self.layer2 = nn.Linear(10, 20)
        self.layer3 = nn.Linear(20, 3 * self.num_qubits + 4 * self.circuit_depth)
        self.circuit_eval = tc.interfaces.torch_interface(self.circuit_eval_probs, jit=True)
        self.act = nn.Tanh()

        # record the params
        self.filename = "nn_generat_params_log{}_{}.txt".format(i + 1, self.num_qubits)
        self.count = 0

    def forward(self):
        y = self.layer1(self.stovec)
        y = self.act(y)
        y = self.layer2(y)
        y = self.act(y)
        y = self.layer3(y)
        y = self.act(y)
        y = y.reshape((3 * self.num_qubits + 4 * self.circuit_depth))
        self.count += 1
        with open(self.filename, 'a') as f:
            f.write(f"--- Iteration {self.count} ---\n")
            np.savetxt(f, y[:, None].detach().numpy(), fmt='%.8f')  # 保存成单列
            f.write("\n")
        probs = self.circuit_eval(y)
        return probs

    def circuit_eval_probs(self, weight):
        qc = tc.Circuit(self.num_qubits, inputs=self.rand_input_state)
        blocks = self.circuit.to_qir()
        label = 0  # used to keep track of the number of 'cz' gates skipped

        for i, block in enumerate(blocks):
            gate = block['name']
            qubits = block['index']

            if gate in ['ry', 'rz']:
                # for the 'ry' and 'rz' gates, we need to adjust the index to skip the 'cz' gate
                adjusted_index = i - label
                theta = weight[adjusted_index]
                getattr(qc, gate)(qubits[0], theta=theta)  # use getattr to dynamically call gates

            elif gate == 'cz':
                # for the 'cz' gate, we apply the gate directly and increment the label count
                qc.cz(*qubits)
                label += 1

        return qc.probability()

class SQC(nn.Module):
    def __init__(self, num_qubits, circuit, circuit_depth, rand_input_state, i):
        super(SQC, self).__init__()
        self.num_qubits = num_qubits  # number of qubits
        self.circuit_depth = circuit_depth  # number of blocks
        self.circuit = circuit  # quantum circuit
        self.input_state = rand_input_state  # random input quantum state


        # connect tensorcircuit to torch and enable JIT compilation
        self.circuit_eval = tc.interfaces.torch_interface(self.circuit_eval_probs, jit=True)

        # initialize the parameters required by the circuit using uniform distribution
        init_method = functools.partial(torch.nn.init.uniform_, b=2 * math.pi)
        self.alpha = torch.nn.Parameter(init_method(torch.Tensor(3 * self.num_qubits + 4 * self.circuit_depth)),
                                        requires_grad=True)

        # record the params
        self.filename = "sqc_generat_params_log{}_{}.txt".format(i + 1, self.num_qubits)
        self.count = 0

    def forward(self):
        y = self.alpha.reshape((3 * self.num_qubits + 4 * self.circuit_depth))
        self.count += 1
        with open(self.filename, 'a') as f:
            f.write(f"--- Iteration {self.count} ---\n")
            np.savetxt(f, y[:, None].detach().numpy(), fmt='%.8f')  # 保存成单列
            f.write("\n")
        probs = self.circuit_eval(self.alpha)  # return quantum state probability
        return probs

    # evaluate probability
    def circuit_eval_probs(self, weight):
        qc = tc.Circuit(self.num_qubits, inputs=self.input_state)
        blocks = self.circuit.to_qir()
        label = 0  # used to keep track of the number of 'cz' gates skipped

        for i, block in enumerate(blocks):
            gate = block['name']
            qubits = block['index']

            if gate in ['ry', 'rz']:
                # for the 'ry' and 'rz' gates, we need to adjust the index to skip the 'cz' gate
                adjusted_index = i - label
                theta = weight[adjusted_index]
                getattr(qc, gate)(qubits[0], theta=theta)  # use getattr to dynamically call gates

            elif gate == 'cz':
                # for the 'cz' gate, we apply the gate directly and increment the label count
                qc.cz(*qubits)
                label += 1


        return qc.probability()