import tensorcircuit as tc
import torch.nn
import torch.nn as nn

import lie_model as lm

tc.set_backend('pytorch')


class Lie_RES_FNN(nn.Module):
    def __init__(self, num_qubits, circuit, circuit_param):
        super(Lie_RES_FNN, self).__init__()
        self.num_qubits = num_qubits
        self.circuit = circuit
        self.circuit_param = circuit_param
        self.init_alpha_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)  # trainable
        # self.init_alpha_bias = nn.Parameter(torch.tensor(0.01), requires_grad=True)  # trainable
        self.first_enhanced_vector = None

        # self.init_vector = torch.rand(3, requires_grad=True)  # to fit the dimension of so3
        self.init_vector = nn.Parameter(torch.rand(3), requires_grad=True)


        self.circuit_eval = tc.interfaces.torch_interface(self.circuit_eval_probs, jit=True)

        # initial lie model
        # model_lie = lm.liemodel(self.init_vector)
        # enhanced_lie_vector = model_lie.final_vector()
        # enhanced_vector = torch.abs(enhanced_lie_vector)

        # self.layer1 = nn.Linear(3, 24)
        self.layer1 = nn.Linear(3, 3)
        self.layer2 = nn.Linear(3, 3)
        self.layer3 = nn.Linear(3, 2 ** (self.num_qubits + 1))
        self.act = nn.Tanh()

    # new active function, which can keep the original data and stress the feature
    def enhance_data(self, x):
        norm = torch.norm(x, keepdim=True)
        x_norm = x / norm
        x_min = x_norm.min()
        x_max = x_norm.max()
        x_01 = (x_norm - x_min) / (x_max - x_min)
        return x + x_01

    def forward(self):
        # if self.first_enhanced_vector is None:
        model_lie = lm.liemodel(self.init_vector)
        enhanced_lie_vector = model_lie.final_vector()
        enhanced_vector = torch.abs(enhanced_lie_vector)

        x = self.layer1(enhanced_vector)
        x = self.act(x)
        x = x + enhanced_vector * self.init_alpha_weight
        a = x
        x = self.layer2(x)
        x = self.act(x)
        x = x + a * self.init_alpha_weight
        x = self.layer3(x)
        x = x.view(-1)
        x = x[:2 ** (self.num_qubits + 1)]

        # select real part and imag part
        real = x[: 2 ** self.num_qubits]
        imag = x[2 ** self.num_qubits:]

        # combine real and imag
        complex_output = real + 1j * imag
        # compute l2 norm
        norm = torch.norm(complex_output, p=2, dim=0, keepdim=True)
        # normalization
        input_state = complex_output / norm
        input_state = input_state.view(*([2] * self.num_qubits))
        probs = self.circuit_eval(input_state)
        return probs

    # def circuit_eval_probs(self, input_state):
    #     # initial circuit
    #     qc = tc.Circuit(self.num_qubits, inputs=input_state)
    #
    #     # transfer circuit to qir
    #     blocks_idx = self.circuit.to_qir()
    #     idx = 0
    #
    #     for i, blocks in enumerate(blocks_idx):
    #         index = blocks['index']
    #         gate = blocks['name']  # get the gate name
    #
    #         if gate == 'cnot':
    #             qc.cnot(index[0], index[1])
    #         elif gate == 'cz':
    #             qc.cz(index[0], index[1])
    #         elif gate == 'ry':
    #             qc.ry(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'rz':
    #             qc.rz(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'rx':
    #             qc.rx(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'h':
    #             qc.h(index[0])
    #
    #     # return circuit probs
    #     return qc.probability()
    # def circuit_eval_probs(self, input_state):
    #     eval_qc = tc.Circuit(self.num_qubits, inputs=input_state)
    #     idx = 0
    #
    #     # 转换为QIR处理
    #     blocks_idx = self.circuit.to_qir()
    #
    #     for block in blocks_idx:
    #         index = block['index']
    #         gate = block['name']
    #
    #         if gate == 'cnot':
    #             eval_qc.cnot(index[0], index[1])
    #         elif gate == 'cz':
    #             eval_qc.cz(index[0], index[1])
    #         elif gate == 'ry':
    #             eval_qc.ry(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'rz':
    #             eval_qc.rz(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'rx':
    #             eval_qc.rx(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'h':
    #             eval_qc.h(index[0])
    #         elif gate == 'cry':
    #             eval_qc.cry(index[0], index[1], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'rxx':
    #             eval_qc.rxx(index[0], index[1], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'ryy':
    #             eval_qc.ryy(index[0], index[1], theta=self.circuit_param[idx])
    #             idx += 1
    #
    #     return eval_qc.probability()
    def circuit_eval_probs(self, input_state):
        eval_qc = tc.Circuit(self.num_qubits, inputs=input_state)
        idx = 0

        blocks_idx = self.circuit.to_qir()

        for block in blocks_idx:
            index = block['index']
            gate = block['name']

            if gate == 'cnot':
                eval_qc.cnot(index[0], index[1])
            elif gate == 'cz':
                eval_qc.cz(index[0], index[1])
            elif gate == 'ry':
                eval_qc.ry(index[0], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'rz':
                eval_qc.rz(index[0], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'rx':
                eval_qc.rx(index[0], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'rxx':
                eval_qc.rxx(index[0], index[1], theta=self.circuit_param[idx])
                idx += 1

        return eval_qc.probability()

class Lie_FNN(nn.Module):
    def __init__(self, num_qubits, circuit, circuit_param):
        super(Lie_FNN, self).__init__()
        self.num_qubits = num_qubits
        self.circuit = circuit
        self.circuit_param = circuit_param
        self.init_alpha_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)  # trainable
        # self.init_alpha_bias = nn.Parameter(torch.tensor(0.01), requires_grad=True)  # trainable
        self.first_enhanced_vector = None

        # self.init_vector = torch.rand(3, requires_grad=True)  # to fit the dimension of so3
        self.init_vector = nn.Parameter(torch.rand(3), requires_grad=True)


        self.circuit_eval = tc.interfaces.torch_interface(self.circuit_eval_probs, jit=True)

        # initial lie model
        # model_lie = lm.liemodel(self.init_vector)
        # enhanced_lie_vector = model_lie.final_vector()
        # enhanced_vector = torch.abs(enhanced_lie_vector)

        # self.layer1 = nn.Linear(3, 24)
        self.layer1 = nn.Linear(3, 3)
        self.layer2 = nn.Linear(3, 3)
        self.layer3 = nn.Linear(3, 2 ** (self.num_qubits + 1))
        self.act = nn.Tanh()

    # new active function, which can keep the original data and stress the feature
    def enhance_data(self, x):
        norm = torch.norm(x, keepdim=True)
        x_norm = x / norm
        x_min = x_norm.min()
        x_max = x_norm.max()
        x_01 = (x_norm - x_min) / (x_max - x_min)
        return x + x_01

    def forward(self):
        # if self.first_enhanced_vector is None:
        model_lie = lm.liemodel(self.init_vector)
        enhanced_lie_vector = model_lie.final_vector()
        enhanced_vector = torch.abs(enhanced_lie_vector)

        x = self.layer1(enhanced_vector)
        x = self.act(x)
        x = self.layer2(x)
        x = self.act(x)
        x = self.layer3(x)
        x = x.view(-1)
        x = x[:2 ** (self.num_qubits + 1)]

        # select real part and imag part
        real = x[: 2 ** self.num_qubits]
        imag = x[2 ** self.num_qubits:]

        # combine real and imag
        complex_output = real + 1j * imag
        # compute l2 norm
        norm = torch.norm(complex_output, p=2, dim=0, keepdim=True)
        # normalization
        input_state = complex_output / norm
        input_state = input_state.view(*([2] * self.num_qubits))
        probs = self.circuit_eval(input_state)
        return probs

    # def circuit_eval_probs(self, input_state):
    #     # initial circuit
    #     qc = tc.Circuit(self.num_qubits, inputs=input_state)
    #
    #     # transfer circuit to qir
    #     blocks_idx = self.circuit.to_qir()
    #     idx = 0
    #
    #     for i, blocks in enumerate(blocks_idx):
    #         index = blocks['index']
    #         gate = blocks['name']  # get the gate name
    #
    #         if gate == 'cnot':
    #             qc.cnot(index[0], index[1])
    #         elif gate == 'cz':
    #             qc.cz(index[0], index[1])
    #         elif gate == 'ry':
    #             qc.ry(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'rz':
    #             qc.rz(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'rx':
    #             qc.rx(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'h':
    #             qc.h(index[0])
    #
    #     # return circuit probs
    #     return qc.probability()
    # def circuit_eval_probs(self, input_state):
    #     eval_qc = tc.Circuit(self.num_qubits, inputs=input_state)
    #     idx = 0
    #
    #     # 转换为QIR处理
    #     blocks_idx = self.circuit.to_qir()
    #
    #     for block in blocks_idx:
    #         index = block['index']
    #         gate = block['name']
    #
    #         if gate == 'cnot':
    #             eval_qc.cnot(index[0], index[1])
    #         elif gate == 'cz':
    #             eval_qc.cz(index[0], index[1])
    #         elif gate == 'ry':
    #             eval_qc.ry(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'rz':
    #             eval_qc.rz(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'rx':
    #             eval_qc.rx(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'h':
    #             eval_qc.h(index[0])
    #         elif gate == 'cry':
    #             eval_qc.cry(index[0], index[1], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'rxx':
    #             eval_qc.rxx(index[0], index[1], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'ryy':
    #             eval_qc.ryy(index[0], index[1], theta=self.circuit_param[idx])
    #             idx += 1
    #
    #     return eval_qc.probability()
    def circuit_eval_probs(self, input_state):
        """
        评估电路概率分布（支持Toffoli和Barrier）

        参数:
            input_state: 输入量子态

        返回:
            测量概率分布
        """
        eval_qc = tc.Circuit(self.num_qubits, inputs=input_state)
        idx = 0

        # 转换为QIR处理
        blocks_idx = self.circuit.to_qir()

        for block in blocks_idx:
            index = block['index']
            gate = block['name']

            if gate == 'cnot':
                eval_qc.cnot(index[0], index[1])
            elif gate == 'cz':
                eval_qc.cz(index[0], index[1])
            elif gate == 'ry':
                eval_qc.ry(index[0], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'rz':
                eval_qc.rz(index[0], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'rx':
                eval_qc.rx(index[0], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'h':
                eval_qc.h(index[0])
            elif gate == 's':  # 新增相位门S
                eval_qc.s(index[0])
            elif gate == 't':  # 新增相位门T
                eval_qc.t(index[0])
            elif gate == 'cry':
                eval_qc.cry(index[0], index[1], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'crz':  # 新增受控RZ门
                eval_qc.crz(index[0], index[1], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'rxx':
                eval_qc.rxx(index[0], index[1], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'ryy':
                eval_qc.ryy(index[0], index[1], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'rzz':
                eval_qc.rzz(index[0], index[1], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'swap':  # 新增SWAP门
                eval_qc.swap(index[0], index[1])
            elif gate == 'toffoli':  # 新增Toffoli门
                eval_qc.toffoli(index[0], index[1], index[2])
            elif gate == 'barrier':  # 处理Barrier（无实际操作）
                pass  # Barrier在模拟中不需要实现

        return eval_qc.probability()

class NGQS_RES_FNN(nn.Module):
    def __init__(self, num_qubits, circuit, circuit_param):
        super(NGQS_RES_FNN, self).__init__()
        self.num_qubits = num_qubits
        self.circuit = circuit
        self.circuit_param = circuit_param
        self.init_alpha_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)  # trainable

        # Generate random alpha
        self.alpha = nn.Parameter(torch.rand(3), requires_grad=True)

        # Create FNN
        self.l1 = nn.Linear(3, 3)
        self.l2 = nn.Linear(3, 3)
        self.l3 = nn.Linear(3, 2 ** (self.num_qubits + 1))
        self.act = nn.Tanh()

        # connect circuit to torch
        self.circuit_eval = tc.interfaces.torch_interface(self.circuit_eval_probs, jit=True)

    # def activation(self, x):
    #     scale = torch.sigmoid(self.init_alpha_weight * x + self.init_alpha_bias)
    #     return x * (1 + scale)

    def forward(self):
        y = self.l1(self.alpha)
        y = self.act(y)
        y = y + self.alpha * self.init_alpha_weight
        a = y
        y = self.l2(y)
        y = self.act(y)
        y = y + a * self.init_alpha_weight
        y = self.l3(y)
        y = y.view(-1)
        y = y[:2 ** (self.num_qubits + 1)]
        # select real part and imag part
        real = y[: 2 ** self.num_qubits]
        imag = y[2 ** self.num_qubits:]

        # combine real and imag
        complex_output = real + 1j * imag
        # compute l2 norm
        norm = torch.norm(complex_output, p=2, dim=0, keepdim=True)
        # normalization
        input_state = complex_output / norm
        input_state = input_state.view(*([2] * self.num_qubits))
        probs = self.circuit_eval(input_state)
        return probs

    # def circuit_eval_probs(self, input_state):
    #     # initial circuit
    #     qc = tc.Circuit(self.num_qubits, inputs=input_state)
    #
    #     # transfer circuit to qir
    #     blocks_idx = self.circuit.to_qir()
    #     idx = 0
    #
    #     for i, blocks in enumerate(blocks_idx):
    #         index = blocks['index']
    #         gate = blocks['name']  # get the gate name
    #
    #         if gate == 'cnot':
    #             qc.cnot(index[0], index[1])
    #         elif gate == 'cz':
    #             qc.cz(index[0], index[1])
    #         elif gate == 'ry':
    #             qc.ry(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'rz':
    #             qc.rz(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'rx':
    #             qc.rx(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'h':
    #             qc.h(index[0])
    #
    #     # return circuit probs
    #     return qc.probability()
    # def circuit_eval_probs(self, input_state):
    #     eval_qc = tc.Circuit(self.num_qubits, inputs=input_state)
    #     idx = 0
    #
    #     # 转换为QIR处理
    #     blocks_idx = self.circuit.to_qir()
    #
    #     for block in blocks_idx:
    #         index = block['index']
    #         gate = block['name']
    #
    #         if gate == 'cnot':
    #             eval_qc.cnot(index[0], index[1])
    #         elif gate == 'cz':
    #             eval_qc.cz(index[0], index[1])
    #         elif gate == 'ry':
    #             eval_qc.ry(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'rz':
    #             eval_qc.rz(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'rx':
    #             eval_qc.rx(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'h':
    #             eval_qc.h(index[0])
    #         elif gate == 'cry':
    #             eval_qc.cry(index[0], index[1], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'rxx':
    #             eval_qc.rxx(index[0], index[1], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'ryy':
    #             eval_qc.ryy(index[0], index[1], theta=self.circuit_param[idx])
    #             idx += 1
    #
    #     return eval_qc.probability()
    def circuit_eval_probs(self, input_state):
        """
        评估电路概率分布（支持Toffoli和Barrier）

        参数:
            input_state: 输入量子态

        返回:
            测量概率分布
        """
        eval_qc = tc.Circuit(self.num_qubits, inputs=input_state)
        idx = 0

        # 转换为QIR处理
        blocks_idx = self.circuit.to_qir()

        for block in blocks_idx:
            index = block['index']
            gate = block['name']

            if gate == 'cnot':
                eval_qc.cnot(index[0], index[1])
            elif gate == 'cz':
                eval_qc.cz(index[0], index[1])
            elif gate == 'ry':
                eval_qc.ry(index[0], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'rz':
                eval_qc.rz(index[0], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'rx':
                eval_qc.rx(index[0], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'h':
                eval_qc.h(index[0])
            elif gate == 's':  # 新增相位门S
                eval_qc.s(index[0])
            elif gate == 't':  # 新增相位门T
                eval_qc.t(index[0])
            elif gate == 'cry':
                eval_qc.cry(index[0], index[1], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'crz':  # 新增受控RZ门
                eval_qc.crz(index[0], index[1], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'rxx':
                eval_qc.rxx(index[0], index[1], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'ryy':
                eval_qc.ryy(index[0], index[1], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'rzz':
                eval_qc.rzz(index[0], index[1], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'swap':  # 新增SWAP门
                eval_qc.swap(index[0], index[1])
            elif gate == 'toffoli':  # 新增Toffoli门
                eval_qc.toffoli(index[0], index[1], index[2])
            elif gate == 'barrier':  # 处理Barrier（无实际操作）
                pass  # Barrier在模拟中不需要实现

        return eval_qc.probability()

class NGQS_FNN(nn.Module):
    def __init__(self, num_qubits, circuit, circuit_param):
        super(NGQS_FNN, self).__init__()
        self.num_qubits = num_qubits
        self.circuit = circuit
        self.circuit_param = circuit_param
        self.init_alpha_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)  # trainable

        # Generate random alpha
        self.alpha = nn.Parameter(torch.rand(3), requires_grad=True)

        # Create FNN
        self.l1 = nn.Linear(3, 3)
        self.l2 = nn.Linear(3, 3)
        self.l3 = nn.Linear(3, 2 ** (self.num_qubits + 1))
        self.act = nn.Tanh()

        # connect circuit to torch
        self.circuit_eval = tc.interfaces.torch_interface(self.circuit_eval_probs, jit=True)

    # def activation(self, x):
    #     scale = torch.sigmoid(self.init_alpha_weight * x + self.init_alpha_bias)
    #     return x * (1 + scale)

    def forward(self):
        y = self.l1(self.alpha)
        y = self.act(y)
        y = self.l2(y)
        y = self.act(y)
        y = self.l3(y)
        y = y.view(-1)
        y = y[:2 ** (self.num_qubits + 1)]
        # select real part and imag part
        real = y[: 2 ** self.num_qubits]
        imag = y[2 ** self.num_qubits:]

        # combine real and imag
        complex_output = real + 1j * imag
        # compute l2 norm
        norm = torch.norm(complex_output, p=2, dim=0, keepdim=True)
        # normalization
        input_state = complex_output / norm
        input_state = input_state.view(*([2] * self.num_qubits))
        probs = self.circuit_eval(input_state)
        return probs

    # def circuit_eval_probs(self, input_state):
    #     # initial circuit
    #     qc = tc.Circuit(self.num_qubits, inputs=input_state)
    #
    #     # transfer circuit to qir
    #     blocks_idx = self.circuit.to_qir()
    #     idx = 0
    #
    #     for i, blocks in enumerate(blocks_idx):
    #         index = blocks['index']
    #         gate = blocks['name']  # get the gate name
    #
    #         if gate == 'cnot':
    #             qc.cnot(index[0], index[1])
    #         elif gate == 'cz':
    #             qc.cz(index[0], index[1])
    #         elif gate == 'ry':
    #             qc.ry(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'rz':
    #             qc.rz(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'rx':
    #             qc.rx(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'h':
    #             qc.h(index[0])
    #
    #     # return circuit probs
    #     return qc.probability()
    # def circuit_eval_probs(self, input_state):
    #     eval_qc = tc.Circuit(self.num_qubits, inputs=input_state)
    #     idx = 0
    #
    #     # 转换为QIR处理
    #     blocks_idx = self.circuit.to_qir()
    #
    #     for block in blocks_idx:
    #         index = block['index']
    #         gate = block['name']
    #
    #         if gate == 'cnot':
    #             eval_qc.cnot(index[0], index[1])
    #         elif gate == 'cz':
    #             eval_qc.cz(index[0], index[1])
    #         elif gate == 'ry':
    #             eval_qc.ry(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'rz':
    #             eval_qc.rz(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'rx':
    #             eval_qc.rx(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'h':
    #             eval_qc.h(index[0])
    #         elif gate == 'cry':
    #             eval_qc.cry(index[0], index[1], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'rxx':
    #             eval_qc.rxx(index[0], index[1], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'ryy':
    #             eval_qc.ryy(index[0], index[1], theta=self.circuit_param[idx])
    #             idx += 1
    #
    #     return eval_qc.probability()
    def circuit_eval_probs(self, input_state):
        """
        评估电路概率分布（支持Toffoli和Barrier）

        参数:
            input_state: 输入量子态

        返回:
            测量概率分布
        """
        eval_qc = tc.Circuit(self.num_qubits, inputs=input_state)
        idx = 0

        # 转换为QIR处理
        blocks_idx = self.circuit.to_qir()

        for block in blocks_idx:
            index = block['index']
            gate = block['name']

            if gate == 'cnot':
                eval_qc.cnot(index[0], index[1])
            elif gate == 'cz':
                eval_qc.cz(index[0], index[1])
            elif gate == 'ry':
                eval_qc.ry(index[0], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'rz':
                eval_qc.rz(index[0], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'rx':
                eval_qc.rx(index[0], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'h':
                eval_qc.h(index[0])
            elif gate == 's':  # 新增相位门S
                eval_qc.s(index[0])
            elif gate == 't':  # 新增相位门T
                eval_qc.t(index[0])
            elif gate == 'cry':
                eval_qc.cry(index[0], index[1], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'crz':  # 新增受控RZ门
                eval_qc.crz(index[0], index[1], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'rxx':
                eval_qc.rxx(index[0], index[1], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'ryy':
                eval_qc.ryy(index[0], index[1], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'rzz':
                eval_qc.rzz(index[0], index[1], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'swap':  # 新增SWAP门
                eval_qc.swap(index[0], index[1])
            elif gate == 'toffoli':  # 新增Toffoli门
                eval_qc.toffoli(index[0], index[1], index[2])
            elif gate == 'barrier':  # 处理Barrier（无实际操作）
                pass  # Barrier在模拟中不需要实现

        return eval_qc.probability()

class RQS(nn.Module):
    def __init__(self, num_qubits, circuit, circuit_param):
        super(RQS, self).__init__()
        self.num_qubits = num_qubits
        self.circuit = circuit
        self.circuit_param = circuit_param

        # Generate random input state
        self.alpha = torch.nn.Parameter(torch.Tensor(2 ** (self.num_qubits + 1)), requires_grad=True)
        torch.nn.init.uniform_(self.alpha)
        # connect circuit to torch
        self.circuit_eval = tc.interfaces.torch_interface(self.circuit_eval_probs, jit=True)

    def forward(self):
        alpha = self.alpha.view(-1)
        real = alpha[:2 ** self.num_qubits]
        imag = alpha[2 ** self.num_qubits:]
        complex_output = real + 1j * imag
        norm = torch.norm(complex_output, p=2, dim=0, keepdim=True)
        input_state = complex_output / norm
        input_state = input_state.view(*([2] * self.num_qubits))
        probs = self.circuit_eval(input_state)
        return probs

    # def circuit_eval_probs(self, input_state):
    #     # initial circuit
    #     qc = tc.Circuit(self.num_qubits, inputs=input_state)
    #
    #     # transfer circuit to qir
    #     blocks_idx = self.circuit.to_qir()
    #     idx = 0
    #
    #     for i, blocks in enumerate(blocks_idx):
    #         index = blocks['index']
    #         gate = blocks['name']  # get the gate name
    #
    #         if gate == 'cnot':
    #             qc.cnot(index[0], index[1])
    #         elif gate == 'cz':
    #             qc.cz(index[0], index[1])
    #         elif gate == 'ry':
    #             qc.ry(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'rz':
    #             qc.rz(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'rx':
    #             qc.rx(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'h':
    #             qc.h(index[0])
    #         elif gate == 'cry':
    #             qc.cry(index[0], index[1], theta=self.circuit_param[idx])
    #             idx += 1
    #
    #     # return circuit probs
    #     return qc.probability()
    # def circuit_eval_probs(self, input_state):
    #     eval_qc = tc.Circuit(self.num_qubits, inputs=input_state)
    #     idx = 0
    #
    #     # 转换为QIR处理
    #     blocks_idx = self.circuit.to_qir()
    #
    #     for block in blocks_idx:
    #         index = block['index']
    #         gate = block['name']
    #
    #         if gate == 'cnot':
    #             eval_qc.cnot(index[0], index[1])
    #         elif gate == 'cz':
    #             eval_qc.cz(index[0], index[1])
    #         elif gate == 'ry':
    #             eval_qc.ry(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'rz':
    #             eval_qc.rz(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'rx':
    #             eval_qc.rx(index[0], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'h':
    #             eval_qc.h(index[0])
    #         elif gate == 'cry':
    #             eval_qc.cry(index[0], index[1], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'rxx':
    #             eval_qc.rxx(index[0], index[1], theta=self.circuit_param[idx])
    #             idx += 1
    #         elif gate == 'ryy':
    #             eval_qc.ryy(index[0], index[1], theta=self.circuit_param[idx])
    #             idx += 1
    #
    #     return eval_qc.probability()
    def circuit_eval_probs(self, input_state):
        """
        评估电路概率分布（支持Toffoli和Barrier）

        参数:
            input_state: 输入量子态

        返回:
            测量概率分布
        """
        eval_qc = tc.Circuit(self.num_qubits, inputs=input_state)
        idx = 0

        # 转换为QIR处理
        blocks_idx = self.circuit.to_qir()

        for block in blocks_idx:
            index = block['index']
            gate = block['name']

            if gate == 'cnot':
                eval_qc.cnot(index[0], index[1])
            elif gate == 'cz':
                eval_qc.cz(index[0], index[1])
            elif gate == 'ry':
                eval_qc.ry(index[0], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'rz':
                eval_qc.rz(index[0], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'rx':
                eval_qc.rx(index[0], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'h':
                eval_qc.h(index[0])
            elif gate == 's':  # 新增相位门S
                eval_qc.s(index[0])
            elif gate == 't':  # 新增相位门T
                eval_qc.t(index[0])
            elif gate == 'cry':
                eval_qc.cry(index[0], index[1], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'crz':  # 新增受控RZ门
                eval_qc.crz(index[0], index[1], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'rxx':
                eval_qc.rxx(index[0], index[1], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'ryy':
                eval_qc.ryy(index[0], index[1], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'rzz':
                eval_qc.rzz(index[0], index[1], theta=self.circuit_param[idx])
                idx += 1
            elif gate == 'swap':  # 新增SWAP门
                eval_qc.swap(index[0], index[1])
            elif gate == 'toffoli':  # 新增Toffoli门
                eval_qc.toffoli(index[0], index[1], index[2])
            elif gate == 'barrier':  # 处理Barrier（无实际操作）
                pass  # Barrier在模拟中不需要实现

        return eval_qc.probability()
