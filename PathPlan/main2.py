import itertools

import numpy as np
import pandas as pd
import tensorcircuit as tc
import torch
import torch.nn as nn
from autoray.lazy.draw import span
from matplotlib import pyplot as plt
from qiskit.quantum_info import random_statevector
from torch import optim
from torch.distributions import Categorical

# 设置默认后端为 numpy，避免自动微分干扰
tc.set_backend("numpy")


# 自定义环境类
class rl_quantum_circuit:
    def __init__(self, num_qubits, input_state):
        self.num_qubits = num_qubits
        self.target_state = random_statevector(2 ** self.num_qubits)
        self.input_state = input_state
        self.quantum_circuit = self.reset()


        # 所有可能的单/双量子比特门组合
        self.single_gates = ['rx', 'ry', 'rz']
        self.two_gates = ['cnot', 'cz']
        self.single_actions = [(g, i) for g in self.single_gates for i in range(num_qubits)]
        self.two_actions = [(g, i, j) for g in self.two_gates for i, j in itertools.permutations(range(num_qubits), 2)]
        self.gate_actions = self.single_actions + self.two_actions

    def step(self, action_type, gate_idx, target_indices):
        if action_type == 0:  # 单量子比特门
            gate_name = self.single_gates[gate_idx]
            qubit = target_indices[0]
            if gate_name == 'rx':
                self.quantum_circuit.rx(qubit, theta=np.random.uniform(0, 1))
            elif gate_name == 'ry':
                self.quantum_circuit.ry(qubit, theta=np.random.uniform(0, 1))
            elif gate_name == 'rz':
                self.quantum_circuit.rz(qubit, theta=np.random.uniform(0, 1))
        else:  # 双量子比特门
            gate_name = self.two_gates[gate_idx]
            i, j = target_indices
            if gate_name == 'cnot':
                self.quantum_circuit.cnot(i, j)
            elif gate_name == 'cz':
                self.quantum_circuit.cz(i, j)

        next_state = self.quantum_circuit.state().ravel()
        fidelity = self.compute_fidelity(next_state)
        reward = fidelity
        done = fidelity > 0.8
        return next_state, reward, done

    def compute_fidelity(self, circuit_state):
        fidelity = np.abs(np.dot(circuit_state.conj(), self.target_state)) ** 2
        return fidelity

    def reset(self):
        quantum_circuit = tc.Circuit(self.num_qubits, inputs=self.input_state)
        # initial quantum circuit
        for i in range(self.num_qubits):
            quantum_circuit.ry(i, theta=np.random.uniform(0, 1))
        for i in range(self.num_qubits):
            quantum_circuit.cnot(i, i + 1 if i < self.num_qubits - 1 else 0)
        return quantum_circuit


# manager for a imaginary target
class Manager(nn.Module):
    def __init__(self, state_dim, goal_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, goal_dim)
        )

    def forward(self, x):
        return self.net(x)


# woker
class Worker(nn.Module):
    def __init__(self, state_dim, goal_dim, num_qubits):
        super().__init__()
        self.num_qubits = num_qubits
        self.shared = nn.Sequential(
            nn.Linear(state_dim + goal_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh()
        )
        # Gate type: single or two-qubit (2 classes)
        self.gate_type_head = nn.Linear(32, 2)

        # Single gate index: rx, ry, rz
        self.single_gate_index = nn.Linear(32, 3)

        # Single qubit index: 0 ~ num_qubits – 1
        self.single_qubit_index = nn.Linear(32, num_qubits)

        # Two-qubit gate type: cnot or cz
        self.two_gate_index = nn.Linear(32, 2)

        # Two-qubit pair indices: 所有 i != j 的排列组合
        self.two_qubit_pair_num = num_qubits * (num_qubits - 1)
        self.two_qubit_pair = nn.Linear(32, self.two_qubit_pair_num)

    def forward(self, x, goal):
        combined = torch.cat([x, goal], dim=-1)
        shared_out = self.shared(combined)
        return {
            "gate_type": self.gate_type_head(shared_out),
            "single_gate": self.single_gate_index(shared_out),
            "single_qubit": self.single_qubit_index(shared_out),
            "two_gate": self.two_gate_index(shared_out),
            "two_qubit": self.two_qubit_pair(shared_out),
        }


# Critic 网络（用于估计状态价值）
class Critic_worker(nn.Module):
    def __init__(self, state_dim, goal_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + goal_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x, goal):
        combined = torch.cat([x, goal], dim=-1)
        return self.net(combined)


class Critic_manager(nn.Module):
    def __init__(self, state_dim, goal_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + goal_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x, goal):
        combined = torch.cat([x, goal], dim=-1)
        return self.net(combined)


def compute_gae(rewards, values, dones, gamma=0.95, lam=0.9):
    """
    Compute Generalized Advantage Estimate (GAE)
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] * (not dones[t]) - values[t]
        gae = delta + gamma * lam * (not dones[t]) * gae
        advantages[t] = gae

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = advantages + values[:T]

    return advantages, returns


def plot_smoothed_fidelity(
        all_episode_fidelities,
        target_fidelity,
        episode,
        num_qubits,
        window_size=50,  # 平滑窗口大小
        downsample_step=10,  # 下采样步长
        save_dir=".",  # 图像保存路径
):
    """
    绘制保真度曲线，包含原始点 + 平滑曲线。
    """
    # 将列表转换为 numpy 数组，确保兼容性
    fidelities = np.array(all_episode_fidelities)

    # 1️⃣ 使用 rolling mean 平滑数据，保持长度一致
    smoothed = pd.Series(fidelities).rolling(window=window_size, center=False).max().fillna(method='bfill').fillna(method='ffill').values

    # 2️⃣ 下采样：每隔 downsample_step 步取一个点
    indices = np.arange(len(fidelities))[::downsample_step]
    raw_downsampled = fidelities[indices]
    smoothed_downsampled = smoothed[indices]

    # 3️⃣ 绘图设置
    plt.figure(figsize=(10, 6))

    # 原始 fidelity 点（下采样显示）
    plt.scatter(indices, raw_downsampled,
                color='blue', alpha=0.4, s=10, label="Raw Fidelity (downsampled)")

    # 平滑 fidelity 曲线（完整）
    plt.plot(smoothed, color='darkblue', linewidth=2, label="Smoothed Fidelity")

    # 平滑 fidelity 点（下采样显示），辅助对比
    plt.scatter(indices, smoothed_downsampled,
                color='green', alpha=0.8, s=15, label="Smoothed Fidelity (downsampled)")

    # 添加目标线
    plt.axhline(y=target_fidelity, color='red', linestyle='--', linewidth=2,
                label=f"Target Fidelity ({target_fidelity:.4f})")

    # 标签与样式
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Fidelity", fontsize=12)
    plt.title("Fidelity Curve During Training", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    # 文件名
    filename = f"{save_dir}/all_fidelity_curve_ep{episode + 1}_qubits{num_qubits}.pdf"
    plt.savefig(filename, format='pdf')
    plt.close()


def train_quantum_policy(num_qubits,
                         alpha,
                         episodes=200,
                         max_steps=50,
                         gamma=0.95,
                         clip_param=0.1,
                         K_epochs=6,
                         target_fidelity=0.8):
    print(f"\n🚀 Begin Training | num_qubits={num_qubits}, episodes={episodes}, max_steps={max_steps}")
    state = random_statevector(2 ** num_qubits)
    state_dim = 2 ** num_qubits * 2  # real + imag parts
    goal_dim = 2 ** num_qubits * 2
    manager = Manager(state_dim, goal_dim)
    worker = Worker(state_dim, goal_dim, num_qubits=num_qubits)
    critic_worker = Critic_worker(state_dim, goal_dim)
    critic_manager = Critic_manager(state_dim, goal_dim)
    optimizer1 = optim.AdamW(manager.parameters(), lr=0.001)
    optimizer2 = optim.AdamW(worker.parameters(), lr=0.001)
    optimizer3 = optim.AdamW(critic_worker.parameters(), lr=0.0005)
    optimizer4 = optim.AdamW(critic_manager.parameters(), lr=0.0005)
    all_episode_fidelities = []

    for episode in range(episodes):
        env = rl_quantum_circuit(num_qubits, state)

        episode_fidelities = []
        episode_reward = []
        episode_hillinger_distance = []
        state = state.data if episode == 0 else np.array(state.data)
        state = np.concatenate([state.real, state.imag])
        done = False

        # 每个 episode 开始时生成新的 goal
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            current_goal = manager(state_tensor)

        total_reward = 0
        states, actions, old_log_probs, rewards, dones = [], [], [], [], []
        worker_values, manager_values = [], []

        for step in range(max_steps):
            if step != 0:
                state = np.concatenate([state.real, state.imag])
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            outputs = worker(state_tensor, current_goal.unsqueeze(0))
            value_w = critic_worker(state_tensor, current_goal.unsqueeze(0))
            value_m = critic_manager(state_tensor, current_goal.unsqueeze(0))

            gate_type_logits = outputs["gate_type"]
            gate_type_dist = Categorical(logits=gate_type_logits)
            gate_type = gate_type_dist.sample()
            log_prob = gate_type_dist.log_prob(gate_type)

            if gate_type.item() == 0:
                sg_dist = Categorical(logits=outputs["single_gate"])
                sq_dist = Categorical(logits=outputs["single_qubit"])
                sg = sg_dist.sample()
                sq = sq_dist.sample()
                action = (0, sg.item(), [sq.item()])
                log_prob += sg_dist.log_prob(sg) + sq_dist.log_prob(sq)
            else:
                tg_dist = Categorical(logits=outputs["two_gate"])
                tq_dist = Categorical(logits=outputs["two_qubit"])
                tg = tg_dist.sample()
                tq = tq_dist.sample()

                pair_list = [(i, j) for i in range(num_qubits) for j in range(num_qubits) if i != j]
                action = (1, tg.item(), pair_list[tq.item()])
                log_prob += tg_dist.log_prob(tg) + tq_dist.log_prob(tq)

            next_state, reward, done = env.step(*action)
            fidelity = reward
            bonus = target_fidelity - fidelity
            penalty = step * 0.01
            hillinger_distance = torch.sqrt(1 - torch.tensor(fidelity))
            reward = alpha * reward - (1 - alpha) * hillinger_distance - penalty
            episode_hillinger_distance.append(hillinger_distance.item())
            total_reward += reward

            episode_fidelities.append(fidelity.item())
            print(f"  ➤ Step {step + 1:3d} | Fidelity: {fidelity:.6f}", end="")
            if done:
                print(" 🔚")
            else:
                print()

            states.append(state_tensor)
            actions.append(action)
            old_log_probs.append(log_prob.detach())
            worker_values.append(value_w.detach())
            manager_values.append(value_m.detach())
            rewards.append(reward.item())
            dones.append(done)

            state = next_state

            if fidelity >= target_fidelity:
                print(f"🎉 At Episode {episode + 1}, Step {step + 1} Achieve Fidelity ({target_fidelity})！")
                break


        all_episode_fidelities.extend(episode_fidelities)
        avg_fidelity = sum(episode_fidelities) / len(episode_fidelities)

        if fidelity >= target_fidelity:
            best_episode = episode + 1
            best_step = step + 1
            best_circuit = env.quantum_circuit.copy()

            if len(all_episode_fidelities) > max_steps:
                    plot_smoothed_fidelity(all_episode_fidelities, target_fidelity, episode,num_qubits)

            plt.figure(figsize=(8, 5))
            plt.plot(episode_fidelities, label="fidelity", color='blue')
            plt.axhline(y=target_fidelity, color='r', linestyle='--', label=f"target Fidelity ({target_fidelity})")
            plt.xlabel("step")
            plt.ylabel("fidelity")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            filename = f"fidelity_curve_ep{best_episode}_step{best_step}_qubits{num_qubits}.pdf"
            plt.savefig(filename, format='pdf')
            plt.close()

            plt.figure(figsize=(8, 5))
            ax1 = plt.subplot()
            line1, = ax1.plot(rewards, label="reward", color='blue')
            ax1.set_xlabel("step")
            ax1.set_ylabel("reward", color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.grid(True)
            # Hellinger Distance
            ax2 = ax1.twinx()
            line2, = ax2.plot(episode_hillinger_distance, label="hellinger distance", color='red')
            ax2.set_ylabel("hellinger distance", color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            lines = [line1, line2]
            labels = [line.get_label() for line in lines]
            ax1.legend(lines, labels, bbox_to_anchor=(0.5, -0.1), ncol=2)
            plt.tight_layout()
            filename = f"reward_each_step_ep{best_episode}_step{best_step}_qubits{num_qubits}.pdf"
            plt.savefig(filename, format='pdf', bbox_inches='tight')
            plt.close()

            print("\n📈 The best circuit structure：")
            print(best_circuit.draw(output='text'))

            circuit_filename = f"best_circuit_ep{best_episode}_step{best_step}_qubits{num_qubits}.txt"
            with open(circuit_filename, "w") as f:
                f.write(str(best_circuit.draw()))
            print(f"Circuit stored at {circuit_filename}")
            print(f"\n🏁 Best Fidelity: {fidelity:.6f} @ Episode {best_episode}, Step {best_step}")
            return

        # 将 value 转换为 tensor
        worker_values_tensor = torch.cat(worker_values).squeeze()
        next_state = np.array(next_state)
        final_value = critic_worker(
            torch.tensor(np.concatenate([next_state.real, state.imag]), dtype=torch.float32).unsqueeze(0),
            current_goal.unsqueeze(0)).detach()

        # 构造完整的 value 序列用于 GAE
        full_worker_values = torch.cat([worker_values_tensor, final_value.view(1)])

        # Convert to tensors
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.bool)

        # 使用 GAE 计算 advantage
        advantages, returns = compute_gae(
            rewards=rewards_tensor,
            values=full_worker_values,
            dones=dones_tensor,
            gamma=0.99,
            lam=0.9
        )

        # 转换为 tensor
        old_log_probs_tensor = torch.stack(old_log_probs)
        states_tensor = torch.cat(states)
        # PPO 更新逻辑
        for _ in range(K_epochs):
            for idx in range(len(states)):
                s = states[idx]
                a = actions[idx]
                old_log_p = old_log_probs_tensor[idx]
                adv = advantages[idx]

                outputs = worker(s, current_goal.unsqueeze(0))

                gate_type_logits = outputs["gate_type"]
                gate_type_dist = Categorical(logits=gate_type_logits)
                gate_type = torch.tensor([a[0]], dtype=torch.long)

                if a[0] == 0:
                    sg = torch.tensor([a[1]], dtype=torch.long)
                    sq = torch.tensor(a[2], dtype=torch.long)
                    log_prob = gate_type_dist.log_prob(gate_type) \
                               + Categorical(logits=outputs["single_gate"]).log_prob(sg) \
                               + Categorical(logits=outputs["single_qubit"]).log_prob(sq)

                    entropy = gate_type_dist.entropy() \
                              + Categorical(logits=outputs["single_gate"]).entropy() \
                              + Categorical(logits=outputs["single_qubit"]).entropy()
                else:
                    tg = torch.tensor([a[1]], dtype=torch.long)
                    tq = torch.tensor([a[2][1]], dtype=torch.long)
                    log_prob = gate_type_dist.log_prob(gate_type) \
                               + Categorical(logits=outputs["two_gate"]).log_prob(tg) \
                               + Categorical(logits=outputs["two_qubit"]).log_prob(tq)

                    entropy = gate_type_dist.entropy() \
                              + Categorical(logits=outputs["two_gate"]).entropy() \
                              + Categorical(logits=outputs["two_qubit"]).entropy()

                value_w = critic_worker(s, current_goal.unsqueeze(0))
                value_m = critic_manager(s, current_goal.unsqueeze(0))

                ratio = torch.exp(log_prob - old_log_p.detach())
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * adv
                # loss_actor = -torch.min(surr1, surr2).mean()
                loss_actor = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

                target_r = returns[idx].detach()
                loss_critic_worker = (value_w - target_r).pow(2)

                final_reward = torch.tensor([rewards[-1]], dtype=torch.float32)
                loss_critic_manager = (value_m - final_reward).pow(2)

                loss = loss_actor + 0.8 * loss_critic_worker + 0.2 * loss_critic_manager

                optimizer1.zero_grad()
                optimizer2.zero_grad()
                optimizer3.zero_grad()
                optimizer4.zero_grad()
                loss.backward()
                optimizer1.step()
                optimizer2.step()
                optimizer3.step()
                optimizer4.step()

        print(f"Episode {episode + 1:3d}; Avg Fidelity: {avg_fidelity:.6f}")

    print("Can't get target fidelity in limit episodes")


# 启动训练
if __name__ == "__main__":
    train_quantum_policy(num_qubits=2, alpha=0.3, episodes=50, max_steps=100, target_fidelity=0.8)
