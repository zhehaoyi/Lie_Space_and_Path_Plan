from qiskit.quantum_info import random_statevector


def random_state(num_qubits):
    # generate a random input quantum state
    input_state = random_statevector(2 ** num_qubits)
    return input_state
