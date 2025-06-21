import random

import numpy as np
import tensorcircuit as tc
import torch
import torch.nn as nn
from qiskit.quantum_info import random_statevector
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

import cost_function as cf

tc.set_backend('pytorch')


def quantum_ciruit(num_qubits, num_layer):
    qc = tc.Circuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(i)

    for i in range(num_layer):
        for j in range(num_qubits):
            if j != (num_qubits - 1):
                qc.cnot(j, j + 1)
        for j in range(num_qubits):
            qc.ry(j)
    return qc, qc.to_qir()


qc, qc_toqir = quantum_ciruit(4, 30)
x = []
for gate in qc_toqir:
    if gate['name'] == 'ry':
        x.append([1])
    if gate['name'] == 'cnot':
        x.append([0])
        x.append([0])


def build_graph(num_idx, num_layers):
    edges = []

    # 初始节点设置
    start_node = 0
    idx_nodes = list(range(1, num_idx + 1))  # idx 节点 1~4
    current_id = num_idx + 1  # 下一个可用节点ID

    # 添加起始边：start -> idx nodes
    for idx in idx_nodes:
        edges.append((start_node, idx))

    # 存储每一层的节点列表
    layers = []

    # 构建第一层
    layer_nodes = list(range(current_id, current_id + 7))
    layers.append(layer_nodes)
    current_id += 7

    # idx -> layer1
    edges.append((idx_nodes[0], layer_nodes[0]))
    edges.append((idx_nodes[1], layer_nodes[0]))
    edges.append((idx_nodes[2], layer_nodes[1]))
    edges.append((idx_nodes[3], layer_nodes[2]))

    # layer1 内部连接
    a, b, c, d, e, f, g = layer_nodes
    edges.append((a, b))
    edges.append((a, d))
    edges.append((b, c))
    edges.append((b, e))
    edges.append((c, f))
    edges.append((c, g))

    # 构建后续层
    for _ in range(1, num_layers):
        prev_last_4 = layers[-1][-4:]  # 上一层的最后4个节点
        layer_nodes = list(range(current_id, current_id + 7))
        layers.append(layer_nodes)
        current_id += 7

        a, b, c, d, e, f, g = layer_nodes

        # 前四点连接
        edges.append((prev_last_4[0], a))
        edges.append((prev_last_4[1], a))
        edges.append((prev_last_4[2], b))
        edges.append((prev_last_4[3], c))

        # layer内部连接
        edges.append((a, b))
        edges.append((a, d))
        edges.append((b, c))
        edges.append((b, e))
        edges.append((c, f))
        edges.append((c, g))

    # 添加终点
    last_layer = layers[-1]
    end_node = current_id
    for node in [last_layer[3], last_layer[4], last_layer[5], last_layer[6]]:
        edges.append((node, end_node))

    # 总结点数
    total_nodes = end_node + 1

    # 转换为 edge_index 格式
    edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long).contiguous()

    return edge_index, layers


edge_index, layers = build_graph(num_idx=4, num_layers=30)

# def build_trainable_edge_weight(num_qubits, edge_index, layers):
#     """
#     构建可学习的 edge_weight，只对特定边初始化随机权重。
#     """
#     num_edges = edge_index.size(1)
#     edge_tuples = edge_index.t().tolist()
#
#     srcs = []
#     dsts = []
#
#     # Type A: 起始边 0 -> idx nodes (假设是 1~4)
#     for i in range(1, num_qubits + 1):  # 如果 num_idx 不固定，可以传入参数
#         srcs.append(0)
#         dsts.append(i)
#
#     # Type B: Layer 内部连接：前3个节点 → 后4个节点
#     for layer in layers:
#         a, b, c, d, e, f, g = layer
#         srcs += [a, b, c, c]
#         dsts += [d, e, f, g]
#
#     # 创建 mask
#     mask = torch.zeros(num_edges, dtype=torch.bool)
#     for i in range(num_edges):
#         u, v = edge_tuples[i]
#         if (u, v) in zip(srcs, dsts):
#             mask[i] = True
#
#     # 初始化 edge_weight
#     edge_weight = torch.zeros(num_edges)
#     edge_weight[mask] = torch.randn(mask.sum())  # 可学习部分
#     edge_weight = Parameter(edge_weight)
#
#     return edge_weight, mask
#
#
# edge_weight, mask = build_trainable_edge_weight(4, edge_index, layers)
data = Data(
    x=torch.tensor(x, dtype=torch.float),
    edge_index=edge_index,
)


class Net(nn.Module):
    def __init__(self, num_qubits, input_state, circuit, edge_index, num_layers, circuit_param,layer):
        super(Net, self).__init__()
        self.num_qubits = num_qubits
        self.input_state = input_state
        self.circuit = circuit
        self.num_layers = num_layers
        self.circuit_param = circuit_param

        self.edge_weight, self.mask = self.build_trainable_edge_weight(edge_index, layer)

        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 1)
        # self.linear1 = nn.Linear(len(self.alpha), 16)
        # self.linear2 = nn.Linear(16, len(self.alpha))
        # self.layer3 = nn.Linear(3, 2 ** (self.num_qubits + 1))

        self.act = nn.Tanh()
        self.circuit_eval = tc.interfaces.torch_interface(self.circuit_eval_probs, jit=True)

    def forward(self, data):
        learnable_edges = data.edge_index.t()[self.mask]
        dst_nodes = learnable_edges[:, 1]
        x = self.conv1(data.x, data.edge_index, self.edge_weight)
        x = self.act(x)
        x = self.conv2(x, data.edge_index, self.edge_weight)
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        trainable_weights = self.edge_weight[self.mask]
        trainable_weights = torch.where(torch.isnan(trainable_weights), torch.zeros_like(trainable_weights), trainable_weights)

        # 提取对应的特征
        dst_features = x[dst_nodes].squeeze() + trainable_weights
        layer1 = nn.Linear(len(dst_features), 2 ** (self.num_qubits + 1))
        dst_features = layer1(dst_features)
        dst_features = dst_features.view(-1)
        dst_features = dst_features[:2 ** (self.num_qubits + 1)]

        # select real part and imag part
        real = dst_features[: 2 ** self.num_qubits]
        imag = dst_features[2 ** self.num_qubits:]

        # combine real and imag
        complex_output = real + 1j * imag
        # compute l2 norm
        norm = torch.norm(complex_output, p=2, dim=0, keepdim=True)
        # normalization
        input_state = complex_output / norm
        input_state = input_state.view(*([2] * self.num_qubits))

        probs = self.circuit_eval(input_state)
        return probs

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

    def build_trainable_edge_weight(self, edge_index, layers):
        """
        构建可学习的 edge_weight，只对特定边初始化随机权重。
        """
        num_edges = edge_index.size(1)
        edge_tuples = edge_index.t().tolist()

        srcs = []
        dsts = []

        # Type A: 起始边 0 -> idx nodes (假设是 1~4)
        for i in range(1, self.num_qubits + 1):  # 如果 num_idx 不固定，可以传入参数
            srcs.append(0)
            dsts.append(i)

        # Type B: Layer 内部连接：前3个节点 → 后4个节点
        for layer in layers:
            a, b, c, d, e, f, g = layer
            srcs += [a, b, c, c]
            dsts += [d, e, f, g]

        # 创建 mask
        mask = torch.zeros(num_edges, dtype=torch.bool)
        for i in range(num_edges):
            u, v = edge_tuples[i]
            if (u, v) in zip(srcs, dsts):
                mask[i] = True

        # 初始化 edge_weight
        edge_weight = torch.rand(num_edges)
        edge_weight[mask] = torch.rand(mask.sum())  # 可学习部分
        edge_weight = Parameter(edge_weight)

        return edge_weight, mask


input_state = random_statevector(2 ** 4)
parmas = np.random.rand(4 + 4 * 30)
model = Net(4, input_state, qc, edge_index, 30, parmas,layers)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
for i in range(100000):
    model.train()
    optimizer.zero_grad()
    probs = model(data)
    loss = cf.cost_function(probs, 4)
    loss.backward()
    optimizer.step()
    if i % 50 == 0:
        print(i, loss.item())

    if loss.item() <= 1e-3:
        print('finished, loss %f' % loss.item())
