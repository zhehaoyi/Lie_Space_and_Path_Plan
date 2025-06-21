import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Statevector, Operator, Pauli

# 设置中文字体和LaTeX支持（可选）
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif'],
    'font.size': 20,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 18,
    'text.usetex': False,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'axes.edgecolor': 'black',
    'axes.grid': False
})


# 获取某个旋转门作用后状态的 Bloch 向量
def get_bloch_vector(theta, axis):
    if axis == 'x':
        gate = [[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                [-1j * np.sin(theta / 2), np.cos(theta / 2)]]
    elif axis == 'y':
        gate = [[np.cos(theta / 2), -np.sin(theta / 2)],
                [np.sin(theta / 2), np.cos(theta / 2)]]
    elif axis == 'z':
        gate = [[np.exp(-1j * theta / 2), 0],
                [0, np.exp(1j * theta / 2)]]
    else:
        raise ValueError("Axis must be 'x', 'y' or 'z'")

    if axis == 'x' or axis == 'y':
        state = Statevector([1, 0])  # 初始态 |0>
    elif axis == 'z':
        state = Statevector([1 / np.sqrt(2), 1 / np.sqrt(2)])  # 初始态 |+>
    op = Operator(gate)
    evolved_state = state.evolve(op)

    def expect(pauli):
        return evolved_state.expectation_value(Pauli(pauli)).real

    x = expect("X")
    y = expect("Y")
    z = expect("Z")

    return x, y, z


# 生成角度范围
angles = np.linspace(0, 2 * np.pi, 200)

# 计算各轴对应的 Bloch 向量路径
rx_path = np.array([get_bloch_vector(theta, 'x') for theta in angles])
ry_path = np.array([get_bloch_vector(theta, 'y') for theta in angles])
rz_path = np.array([get_bloch_vector(theta, 'z') for theta in angles])

zero_state = Statevector([1, 0])
one_state = Statevector([0, 1])
plus_state = Statevector([1 / np.sqrt(2), 1 / np.sqrt(2)])


def get_bloch_from_state(state):
    x = state.expectation_value(Pauli("X")).real
    y = state.expectation_value(Pauli("Y")).real
    z = state.expectation_value(Pauli("Z")).real
    return x, y, z


zero_bloch = get_bloch_from_state(zero_state)
one_bloch = get_bloch_from_state(one_state)
plus_bloch = get_bloch_from_state(plus_state)


# 创建图形
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')

# 绘制旋转路径
ax.plot(rx_path[:, 0], rx_path[:, 1], rx_path[:, 2], label=r'Rx($\theta$)', color='#2E86AB', linewidth=3)
ax.plot(ry_path[:, 0], ry_path[:, 1], ry_path[:, 2], label=r'Ry($\theta$)', color='#52B788', linewidth=3)
ax.plot(rz_path[:, 0], rz_path[:, 1], rz_path[:, 2], label=r'Rz($\theta$)', color='#FFB700', linewidth=3)
ax.scatter(*zero_bloch, color='#A23B72', s=100, label=r'$|0\rangle$', marker='o', edgecolors='black')
ax.scatter(*one_bloch, color='#FB8500', s=100, label=r'$|1\rangle$', marker='o', edgecolors='black')
ax.scatter(*plus_bloch, color='#E85D04', s=100, label=r'$|+\rangle$', marker='o', edgecolors='black')
ax.text(zero_bloch[0], zero_bloch[1], zero_bloch[2], r"$|0\rangle$", color="black", fontweight='bold', fontsize=24)
ax.text(plus_bloch[0], plus_bloch[1], plus_bloch[2], r"$|+\rangle$", color="black", fontweight='bold', fontsize=24)
ax.text(one_bloch[0], one_bloch[1], one_bloch[2], r"$|1\rangle$", color="black", fontweight='bold', fontsize=24)

# 绘制单位球面（辅助视觉）
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 50)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='gray', alpha=0.1, linewidth=0)

# 设置图形属性
# ax.set_title("Quantum Rotation Paths on the Bloch Sphere")
ax.set_xlabel("X", fontweight='bold', fontsize=24, labelpad=10)
ax.set_ylabel("Y", fontweight='bold', fontsize=24, labelpad=10)
ax.set_zlabel("Z", fontweight='bold', fontsize=24, labelpad=10)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

# 设置刻度间隔为 0.5
tick_values = np.arange(-1.0, 1.0 + 0.01, 0.5)
ax.set_xticks(tick_values)
ax.set_yticks(tick_values)
ax.set_zticks(tick_values)
ax.tick_params(axis='both', which='major', labelsize=22)
legend = ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.1), fontsize=22)
ax.set_box_aspect([1, 1, 1])  # 保持比例一致
plt.tight_layout()

# 保存为 PDF
plt.savefig('bloch_rotation_paths.pdf', dpi=300, bbox_inches='tight')
