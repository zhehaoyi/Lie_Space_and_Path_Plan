import random
from typing import Tuple

import numpy as np
import tensorcircuit as tc
import matplotlib.pyplot as plt


def random_circuit(num_qubits, depth):
    qc = tc.Circuit(num_qubits)
    para_group_idx = 0

    for i in range(num_qubits):
        # add random rx, ry, rz gates first
        qc.rx(i)
        qc.ry(i)
        qc.rz(i)

    for i in range(depth):
        # select two random qubits
        # generate random target for different block
        # target_idx = random.randint(1, 2)
        q0, q1 = random.sample(range(num_qubits), 2)
        q0, q1 = min(q0, q1), max(q0, q1)  # q0 < q1，
        # if target_idx == 1:
        # qc.h(q0)
        # qc.h(q1)
        # qc.cz(q0, q1)
        # qc.ry(q0)
        # qc.ry(q1)
        # qc.cz(q0, q1)
        # qc.cnot(q0, q1)
        # qc.cnot(q1, q0)
        # qc.cz(q0, q1)
        # if target_idx == 2:
        # qc.cz(q0, q1)
        # qc.rx(q0)
        # qc.rz(q1)
        # qc.cz(q0, q1)
        qc.cz(q0, q1)
        qc.rx(q0)
        qc.rx(q1)
        qc.rz(q0)
        qc.rz(q1)
        qc.cz(q0, q1)
        para_group_idx += 1
        # qc.ry(q0)
        # qc.rz(q1)
        # qc.cnot(q0, q1)
        # qc.rx(q1)
        # qc.cnot(q1, q0)
        # qc.rz(q0)
        # qc.cry(q0, q1)
        # qc.cz(q0, q1)
        # qc.rxx(q0, q1, theta=np.pi / 4)
        # qc.ryy(q0, q1, theta=np.pi / 4)

    return qc, para_group_idx


# def random_circuit(num_qubits: int, depth: int) -> Tuple[tc.Circuit, int]:
#     qc = tc.Circuit(num_qubits)
#     para_group_idx = 0
#
#     # 可用门集合
#     single_gates = ['rx', 'ry', 'rz', 'h']
#     two_qubit_gates = ['cnot', 'cz', 'rxx', 'ryy', 'cry']
#
#     # 初始随机单比特门
#     for i in range(num_qubits):
#         gate = random.choice(single_gates)
#         if gate in ['rx', 'ry', 'rz']:
#             qc.__getattribute__(gate)(i)
#             para_group_idx += 1
#         else:
#             qc.__getattribute__(gate)(i)
#
#     # 深度循环
#     for _ in range(depth):
#         # 随机选择两个不同的量子比特
#         q0, q1 = random.sample(range(num_qubits), 2)
#         q0, q1 = min(q0, q1), max(q0, q1)  # 保持q0 < q1
#
#         # 随机选择门类型
#         gate_type = random.choice(two_qubit_gates)
#
#         if gate_type == 'rxx':
#             theta = np.random.uniform(0, 2 * np.pi)
#             qc.rxx(q0, q1, theta=theta)
#             para_group_idx += 1
#         elif gate_type == 'ryy':
#             theta = np.random.uniform(0, 2 * np.pi)
#             qc.ryy(q0, q1, theta=theta)
#             para_group_idx += 1
#         elif gate_type == 'cry':
#             theta = np.random.uniform(0, 2 * np.pi)
#             qc.cry(q0, q1, theta=theta)
#             para_group_idx += 1
#         else:  # cnot 或 cz
#             qc.__getattribute__(gate_type)(q0, q1)
#
#         # 有一定概率添加后续单比特门
#         if random.random() > 0.5:
#             for q in [q0, q1]:
#                 gate = random.choice(single_gates)
#                 if gate in ['rx', 'ry', 'rz']:
#                     qc.__getattribute__(gate)(q)
#                     para_group_idx += 1
#                 else:
#                     qc.__getattribute__(gate)(q)
#
#     return qc, para_group_idx
# def random_circuit(num_qubits: int, depth: int) -> Tuple[tc.Circuit, int]:
#     """
#     构建随机量子电路（适度复杂化版本）
#
#     改进点：
#     1. 增加更多单比特门类型（如相位门S/T）
#     2. 增加两比特门类型（如SWAP、CRZ）
#     3. 随机插入Barrier增加复杂度
#     4. 50%概率在纠缠层后添加单比特门
#     5. 10%概率添加三比特门（如Toffoli）
#
#     参数:
#         num_qubits: 量子比特数
#         depth: 电路深度
#
#     返回:
#         Tuple[电路对象, 参数组数]
#     """
#     qc = tc.Circuit(num_qubits)
#     para_group_idx = 0
#
#     # 扩展门集合
#     single_gates = ['rx', 'ry', 'rz', 'h']  # 新增S/T门
#     two_qubit_gates = ['cnot', 'cz', 'rxx', 'ryy', 'cry', 'crz', 'swap']  # 新增CRZ/SWAP
#
#     # 初始随机单比特门（增加多样性）
#     for i in range(num_qubits):
#         gate = random.choice(single_gates)
#         if gate in ['rx', 'ry', 'rz']:
#             qc.__getattribute__(gate)(i)
#             para_group_idx += 1
#         else:
#             qc.__getattribute__(gate)(i)
#
#         # # 30%概率插入Barrier增加复杂度
#         # if random.random() < 0.3:
#         #     qc.barrier(i)
#
#     # 深度循环
#     for _ in range(depth):
#         # 随机选择两个不同的量子比特（允许非相邻比特）
#         q0, q1 = random.sample(range(num_qubits), 2)
#         q0, q1 = min(q0, q1), max(q0, q1)  # 保持q0 < q1
#
#         # 随机选择门类型
#         gate_type = random.choice(two_qubit_gates)
#
#         # 处理两比特门
#         if gate_type in ['rxx', 'ryy', 'cry', 'crz']:
#             theta = np.random.uniform(0, 2 * np.pi)
#             qc.__getattribute__(gate_type)(q0, q1, theta=theta)
#             para_group_idx += 1
#         else:
#             qc.__getattribute__(gate_type)(q0, q1)
#
#         # 50%概率添加后续单比特门（两个比特都加）
#         if random.random() > 0.5:
#             for q in [q0, q1]:
#                 gate = random.choice(single_gates)
#                 if gate in ['rx', 'ry', 'rz']:
#                     qc.__getattribute__(gate)(q)
#                     para_group_idx += 1
#                 else:
#                     qc.__getattribute__(gate)(q)
#
#         # 10%概率尝试添加三比特门（需要至少3个量子比特）
#         if num_qubits >= 3 and random.random() < 0.1:
#             control1, control2, target = random.sample(range(num_qubits), 3)
#             qc.toffoli(control1, control2, target)
#
#     return qc, para_group_idx
def barren_plateau_circuit(num_qubits: int, depth: int) -> Tuple[tc.Circuit, int]:
    """
    生成具有贫瘠高原效应的复杂量子电路

    特性：
    1. 全局纠缠结构
    2. 随机参数化旋转门
    3. 交替的局部和全局耦合
    4. 深度与比特数成比例

    参数:
        num_qubits: 量子比特数 (≥4)
        depth: 电路深度 (≥10)

    返回:
        (电路对象, 参数数量)
    """
    qc = tc.Circuit(num_qubits)
    param_count = 0

    # 1. 初始强纠缠层
    for q in range(num_qubits):
        qc.h(q)
    for i in range(num_qubits - 1):
        qc.cnot(i, i + 1)
    qc.cnot(num_qubits - 1, 0)  # 环形耦合

    # 2. 重复的参数化纠缠块
    for d in range(depth):
        # 单比特随机旋转层
        for q in range(num_qubits):
            theta = np.random.uniform(0, 2 * np.pi)
            qc.rz(q)
            param_count += 1

            if (q + d) % 3 == 0:  # 非均匀选择
                phi = np.random.uniform(0, 2 * np.pi)
                qc.rx(q)
                param_count += 1

        # 全局纠缠模式（交替层）
        if d % 2 == 0:
            # 全连接RZZ耦合
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    theta = np.random.uniform(0, np.pi / 2)
                    qc.rzz(i, j, theta=theta)
                    param_count += 1
        else:
            # 随机配对CNOT
            pairs = np.random.permutation(num_qubits)
            # 确保只取偶数个比特进行配对
            for k in range(0, len(pairs) - len(pairs) % 2, 2):
                qc.cnot(pairs[k].item(), pairs[k + 1].item())

    return qc, param_count
