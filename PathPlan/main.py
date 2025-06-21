import random
from collections import deque
from math import pi

import numpy as np
import torch
from qiskit.quantum_info import random_statevector
from torch import exp, nn, optim, complex128

state_size = 8

# 动作空间大小：前面定义的动作总数是141种
action_size = 14

# 超参数
batch_size = 16  # 每次训练用的经验数量
gamma = 0.95  # 折扣因子
epsilon_start = 1.0  # 初始探索率
epsilon_end = 0.01  # 最小探索率
epsilon_decay = 0.995  # 探索率衰减
learning_rate = 0.001  # 学习率
memory_size = 10000  # 经验回放缓冲区大小
episodes = 200  # 总共训练多少个 episode
max_steps = 100  # 每个 episode 的最大步数


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.net(x)
actions = []
for angle in [np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi),
              np.random.uniform(0, 2 * np.pi)]:
    actions.append(('rx', {'angle': angle}))
    actions.append(('ry', {'angle': angle}))
    actions.append(('rz', {'angle': angle}))
# two qubits gate
for from_obj, to_obj in [('Obj1', 'Obj2'), ('Obj2', 'Obj1')]:
    if from_obj or to_obj == 1:
        actions.append(('cnot', {
            'from': from_obj,
            'to': to_obj,
            'cond_bit': 1,
            'flip_bit': 1
        }))

target_state = input_state = random_statevector(2 ** 4)


def flatten_state(state):
    flat = []
    for obj in state:
        flat.extend(obj)
    return flat


def choose_action(model, state, epsilon):
    if np.random.rand() < epsilon:
        return random.randint(0, action_size - 1)  # 随机选择动作
    else:
        with torch.no_grad():
            q_values = model(state)
            return int(torch.argmax(q_values).item())  # 选择最优动作


def compute_fidelity_reward(current_state_vector, target_state_vector):
    fidelity = np.abs(np.dot(current_state_vector.conj(), target_state_vector)) ** 2
    return float(fidelity)
def step_1(current_state, action_idx):
    action_type, params = actions[action_idx]
    new_state = apply_operation(current_state, action_type, params)
    # 转换为张量积态
    vector_state = state_to_tensor_product(new_state)
    # 计算 fidelity
    fidelity = compute_fidelity_reward(vector_state, target_state)
    # reward 设为负的 fidelity，这样最大化 reward 就等于最小化 fidelity
    reward = -fidelity
    return new_state, reward
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # 使用双端队列存储经验
    def add(self, state, action, reward, next_state, done):
        """添加一条经验"""
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        """随机采样一批经验"""
        return random.sample(self.buffer, batch_size)
    def size(self):
        """返回当前缓冲区中的经验数量"""
        return len(self.buffer)
def train(model, target_model, optimizer, buffer, batch_size, gamma):
    """
    从经验回放缓冲区中采样并训练模型
    :param model: 主网络
    :param target_model: 目标网络
    :param optimizer: 优化器
    :param buffer: 缓冲区
    :param batch_size: 批量大小
    :param gamma: 折扣因子
    """
    a = buffer.size()
    if a < batch_size:
        return

    # 采样一批经验
    batch = buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    # 转换为张量
    states = torch.tensor(np.array(states), dtype=torch.complex128)
    actions = torch.tensor(actions, dtype=torch.complex128).unsqueeze(-1)
    rewards = torch.tensor(rewards, dtype=torch.complex128)
    next_states = torch.tensor(np.array(next_states), dtype=torch.complex128)
    dones = torch.tensor(dones, dtype=torch.complex128)

    # 获取当前动作的 Q 值
    current_q_values = model(states).gather(1, actions).squeeze()

    # 使用目标网络预测下一状态的最大 Q 值
    with torch.no_grad():
        next_q_values = target_model(next_states).max(1).values
        expected_q_values = rewards + (1 - dones) * gamma * next_q_values

    # 计算损失
    loss = nn.MSELoss()(current_q_values, expected_q_values)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()

if __name__ == '__main__':
    # 初始状态：全部基态
    initial_state = [
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0]
    ]

    # 初始化模型和目标网络
    model = QNetwork(state_size, action_size)
    target_model = QNetwork(state_size, action_size)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    buffer = ReplayBuffer(memory_size)

    epsilon = epsilon_start

    best_fidelity = float('inf')
    best_episode = 0
    best_actions = []

    print("开始训练...")

    for episode in range(episodes):
        state = initial_state.copy()
        total_reward = 0
        done = False
        action_history = []  # 记录本 episode 所有动作

        for step_num in range(max_steps):
            flat_state = flatten_state(state)
            state_tensor = torch.tensor(flat_state, dtype=torch.complex128).unsqueeze(0)

            action_idx = choose_action(model, state_tensor, epsilon)
            action_history.append(action_idx)

            next_state, reward = step_1(state, action_idx)
            done = False

            flat_next_state = flatten_state(next_state)

            buffer.add(flat_state, action_idx, reward, flat_next_state, done)

            state = next_state
            total_reward += reward

            train(model, target_model, optimizer, buffer, batch_size, gamma)

            if done:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())

        # 每次结束后，计算当前 fidelity
        final_vector = state_to_tensor_product(state)
        fidelity = compute_fidelity_reward(final_vector, target_state)

        if fidelity < best_fidelity:
            best_fidelity = fidelity
            best_episode = episode
            best_actions = action_history.copy()

        if 1 - fidelity < 0.0001:
            print("finished")
            break

        print(f"Episode {episode + 1}, Fidelity: {fidelity:.6f}, Epsilon: {epsilon:.2f}")

    # 输出最佳结果
    print("\n\n🎉 最佳结果出现在 Episode", best_episode + 1)
    print("Fidelity:", best_fidelity)
    print("使用的动作序列（编号）:")
    print(best_actions)
    print("对应的动作为:")
    for idx in best_actions:
        print(f"{idx}: {actions[idx]}")
