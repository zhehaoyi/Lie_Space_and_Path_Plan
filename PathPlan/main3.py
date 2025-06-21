import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorcircuit as tc
import torch
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from qiskit.quantum_info import random_statevector
from torch.distributions import Categorical

# 设置设备
tc.set_backend("pytorch")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# reinforcement learning evn
class rl_quantum_circuit:
    def __init__(self, num_qubits, input_state):
        self.num_qubits = num_qubits
        self.target_state = random_statevector(2 ** self.num_qubits)
        self.input_state = input_state.data
        self.quantum_circuit = self.reset()

        # useable gates
        self.single_gates = ['rx', 'ry', 'rz']
        self.two_gates = ['cnot', 'cz']
        self.single_actions = [(g, i) for g in self.single_gates for i in range(num_qubits)]
        self.two_actions = [(g, i, j) for g in self.two_gates for i, j in itertools.permutations(range(num_qubits), 2)]
        self.gate_actions = self.single_actions + self.two_actions

    def step(self, action_type, gate_idx, target_indices):
        if action_type == 0:  # single qubit gate
            gate_name = self.single_gates[gate_idx]
            qubit = target_indices[0]
            if gate_name == 'rx':
                self.quantum_circuit.rx(qubit, theta=np.random.uniform(0, 1))
            elif gate_name == 'ry':
                self.quantum_circuit.ry(qubit, theta=np.random.uniform(0, 1))
            elif gate_name == 'rz':
                self.quantum_circuit.rz(qubit, theta=np.random.uniform(0, 1))
        else:  # two qubit gate
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
        fidelity = np.abs(np.dot(circuit_state.resolve_conj().numpy(), self.target_state)) ** 2
        return fidelity

    def reset(self):
        quantum_circuit = tc.Circuit(self.num_qubits, inputs=torch.tensor(self.input_state))
        # initial random gate
        for i in range(self.num_qubits):
            quantum_circuit.ry(i, theta=np.random.uniform(0, 1))
        for i in range(self.num_qubits - 1):
            quantum_circuit.cnot(i, i + 1)
        return quantum_circuit


# Manager and Worker
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
        self.gate_type_head = nn.Linear(32, 2)
        self.single_gate_index = nn.Linear(32, 3)
        self.single_qubit_index = nn.Linear(32, num_qubits)
        self.two_gate_index = nn.Linear(32, 2)
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


# gae plot
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.9):
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] * (not dones[t]) - values[t]
        gae = delta + gamma * lam * (not dones[t]) * gae
        advantages[t] = gae

    # normalization
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = advantages + values[:T]

    return advantages, returns


def plot_smoothed_fidelity(fidelities, target_fidelity=0.8, run_id=0, num_qubits=2, window_size=10):
    plt.rcdefaults()

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif'],
        'font.size': 12,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 18,
        'text.usetex': False,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'axes.edgecolor': 'black',
        'axes.grid': True
    })

    x_values = np.arange(len(fidelities))
    y_values = fidelities

    # smoothe the data
    smoothed = pd.Series(y_values).rolling(window=window_size, center=False).max().fillna(method='bfill').fillna(
        method='ffill').values

    fig, ax = plt.subplots(figsize=(12, 4), facecolor='white')

    line = ax.plot(x_values, smoothed, linewidth=3, markersize=8,
                   color='#2E86AB', markerfacecolor='#A23B72', markeredgecolor='black',
                   markeredgewidth=1.5, alpha=0.8, label="Fidelity")

    # add target line
    ax.axhline(y=target_fidelity, color='red', linestyle='--', linewidth=2, label="Target Fidelity")

    ax.set_xlabel('step', fontweight='bold')
    ax.set_ylabel('fidelity', fontweight='bold')

    ax.set_xlim(0, len(fidelities) + 1)
    ax.set_ylim(0, 1.05)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    # step_interval = max(1, len(x_values) // 20)
    # for i in range(0, len(x_values), step_interval):
    #     ax.annotate(f"{smoothed[i]:.2f}", (x_values[i], smoothed[i]),
    #                 textcoords="offset points", xytext=(0, 10), ha='center',
    #                 fontsize=12, fontweight='bold', color='#2E86AB')

    # ax.text(0.6, 0.1, 'Smoothed using rolling maximum (window size: 50)', transform=ax.transAxes,
    #         fontsize=12, style='italic', alpha=0.7)

    plt.tight_layout(pad=2.0)

    # save the image
    filename = f"fidelity_run{run_id}.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close()


def plot_all_smoothed_fidelities(all_results, target_fidelity=0.8, window_size=10, save_dir="."):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif'],
        'font.size': 12,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 18,
        'text.usetex': False,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'axes.edgecolor': 'black',
        # 'axes.grid': True
    })

    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['#2E86AB', '#52B788', '#FB8500', '#003049', '#7209B7',
              '#4CC9F0', '#F72585', '#3A0CA3', '#72EFDD', '#FF9F1C']

    max_len = max(len(f) for f in all_results)

    x = np.arange(max_len)
    y_spacing = 50
    y = np.arange(0, len(all_results) * y_spacing, y_spacing)

    ax.set_box_aspect([3, 4, 1.5])

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')

    ax.xaxis._axinfo["grid"].update({"visible": False})
    ax.yaxis._axinfo["grid"].update({"visible": False})
    ax.zaxis._axinfo["grid"].update({"visible": True, "linewidth": 0.5, "alpha": 0.1})

    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X, dtype=float)

    # fill Z
    for run_id, fidelities in enumerate(all_results):
        if len(fidelities) < max_len:
            padded = np.pad(fidelities, (0, max_len - len(fidelities)), mode='edge')
        else:
            padded = fidelities[:max_len]

        smoothed = pd.Series(padded).rolling(window=window_size, center=False).max().fillna(
            method='bfill').fillna(method='ffill').values
        Z[run_id, :] = smoothed

        last_x = x[-1]
        last_y = y[run_id]
        last_z = smoothed[-1]
        ax.scatter(last_x, last_y, last_z, c='red', s=50, edgecolors='black')
        ax.text(last_x + 4, last_y, last_z, f"{last_z:.2f}", color="black", fontsize=8, fontweight='bold',
                verticalalignment='bottom', horizontalalignment='left')


        ax.plot(x, [y[run_id]] * len(x), smoothed, color=colors[run_id % len(colors)], linewidth=1.8, alpha=0.8,
                label=f'Round {run_id + 1}')
        poly = Poly3DCollection([list(zip(x, [y[run_id]] * len(x), smoothed)) + list(
            zip(x[::-1], [y[run_id]] * len(x[::-1]), [0] * len(x)))])
        poly.set_color(colors[run_id % len(colors)])
        poly.set_alpha(0.05)
        ax.add_collection3d(poly)


    ax.set_yticks(y)
    ax.set_yticklabels([str(i + 1) for i in range(len(all_results))])  # y轴标签仍为1,2,3...

    ax.xaxis.labelpad = 6
    ax.yaxis.labelpad = 6
    ax.zaxis.labelpad = 0.5

    ax.set_xlabel("Step", fontsize=14, fontweight='bold')
    ax.set_ylabel("Round", fontsize=14, fontweight='bold')
    ax.set_zlabel("Fidelity", fontsize=14, fontweight='bold')
    ax.set_zlim(0, 1.05)

    handles, labels = ax.get_legend_handles_labels()
    n_legends = len(labels)
    ncol = min(5, n_legends)
    ax.legend(
        handles, labels,
        frameon=True,
        bbox_to_anchor=(1.05, 1),
        ncol=ncol,  # 动态列数
        loc='lower right',
        fontsize=10
    )

    ax.view_init(elev=55., azim=-30)
    ax.set_ylim(-y_spacing // 2, y[-1] + y_spacing // 2)

    plt.tight_layout()

    # save the figure
    filename = os.path.join(save_dir, "all_fidelities_3d_waterfall_optimized.pdf")
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close()


def save_to_txt(data, filename, mode='a'):
    with open(filename, mode) as f:
        if isinstance(data[0], (list, np.ndarray)):  # save all_episode_fidelities_list）
            for run_id, fids in enumerate(data):
                f.write(f"Run_{run_id + 1}: " + ", ".join([f"{x:.4f}" for x in fids]) + "\n")
        else:  # save fidelities
            f.write(f"Step, Fidelity\n")
            for step, fid in enumerate(data):
                f.write(f"{step}, {fid:.4f}\n")


# single episode
def train_episode(input_state, run_id):
    env = rl_quantum_circuit(num_qubits, input_state)
    episode_fidelities = []
    done = False

    state_np = np.concatenate([input_state.data.real, input_state.data.imag])
    state_tensor = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        current_goal = manager(state_tensor)

    states, actions, old_log_probs, rewards, dones = [], [], [], [], []

    while not done:
        outputs = worker(state_tensor, current_goal)
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

        next_state, fidelity, done = env.step(*action)
        reward = fidelity
        next_state_np = np.concatenate([next_state.real, next_state.imag])

        episode_fidelities.append(fidelity)
        states.append(state_tensor)
        actions.append(action)
        old_log_probs.append(log_prob.detach())
        rewards.append(reward)
        dones.append(done)

        state_tensor = torch.tensor(next_state_np, dtype=torch.float32).unsqueeze(0)

        if fidelity >= 0.8:
            print(f"Run {run_id + 1} Episode: Achieved Target Fidelity!")
            break

    # PPO
    if len(states) > 0:
        states_tensor = torch.cat(states)
        actions_tensor = actions
        old_log_probs_tensor = torch.stack(old_log_probs)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.bool)

        with torch.no_grad():
            values = critic_worker(states_tensor, current_goal.expand_as(states_tensor[:, :current_goal.shape[-1]]))
            final_value = critic_worker(torch.tensor(next_state_np, dtype=torch.float32).unsqueeze(0),
                                        current_goal).item()
            full_values = torch.cat([values.squeeze(), torch.tensor([final_value])])

        advantages, returns = compute_gae(rewards_tensor, full_values, dones_tensor)

        for idx in range(len(states)):
            s = states[idx]
            a = actions[idx]
            old_log_p = old_log_probs_tensor[idx]
            adv = advantages[idx]
            ret = returns[idx]

            outputs = worker(s.unsqueeze(0), current_goal.unsqueeze(0))

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

            value_w = critic_worker(s.unsqueeze(0), current_goal.unsqueeze(0))
            value_m = critic_manager(s.unsqueeze(0), current_goal.unsqueeze(0))

            ratio = torch.exp(log_prob - old_log_p.detach())
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 0.8, 1.2) * adv
            loss_actor = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()
            loss_critic_worker = (value_w - ret.detach()).pow(2).mean()
            loss_critic_manager = (value_m - ret.detach()).pow(2).mean()

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            optimizer4.zero_grad()
            loss_actor.backward()
            loss_critic_worker.backward()
            loss_critic_manager.backward()
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()

    return {
        "fidelities": episode_fidelities,
        "achieved": fidelity >= 0.8,
        "steps": len(episode_fidelities),
        "best_circuit": env.quantum_circuit.draw(output='text') if fidelity >= 0.8 else None
    }


# main training part
def main():
    global num_qubits
    num_qubits = 2
    initial_state = random_statevector(2 ** num_qubits)

    all_episode_fidelities_list = []
    all_episode_fidelities = []
    success_rates = []
    steps_to_solution = []
    best_circuits = []

    for run_id in range(10):
        print(f"\nRunning Experiment {run_id + 1}/10")
        result = train_episode(initial_state, run_id)
        achieved = result['achieved']
        steps = result['steps']
        fidelities = result['fidelities']

        all_episode_fidelities.extend(fidelities)
        success_rates.append(1 if achieved else 0)

        if achieved:
            all_episode_fidelities_list.append(fidelities)
            best_circuits.append(result['best_circuit'])

            # 保存电路
            with open(f"best_circuit_run{run_id}.txt", "w") as f:
                f.write(str(result['best_circuit']))

        if achieved:
            steps_to_solution.append(steps)

        if achieved:
            plot_smoothed_fidelity(fidelities, 0.8, run_id, num_qubits)
            save_to_txt(fidelities, f"fidelity_run_{run_id + 1}_{num_qubits}.txt", mode='w')

    if len(all_episode_fidelities_list) > 0:
        plot_all_smoothed_fidelities(all_episode_fidelities_list)
        save_to_txt(all_episode_fidelities_list, "all_fidelities_summary.txt", mode='w')

    print("\nFinal Results:")
    print(f"Total Runs: 10")
    print(f"Success Count: {sum(success_rates)}")
    print(f"Success Rate: {sum(success_rates) / 10:.2%}")


if __name__ == "__main__":
    # initial state dim goal dim and model
    state_dim = 2 ** 2 * 2
    goal_dim = 2 ** 2 * 2

    worker = Worker(state_dim, goal_dim, num_qubits=2).to(device)
    critic_worker = Critic_worker(state_dim, goal_dim).to(device)
    manager = Manager(state_dim, goal_dim).to(device)
    critic_manager = Critic_manager(state_dim, goal_dim).to(device)

    optimizer1 = optim.Adam(worker.parameters(), lr=3e-4)
    optimizer2 = optim.Adam(critic_worker.parameters(), lr=1e-3)
    optimizer3 = optim.Adam(manager.parameters(), lr=3e-4)
    optimizer4 = optim.Adam(critic_manager.parameters(), lr=1e-3)

    main()
