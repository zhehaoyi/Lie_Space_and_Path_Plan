import random

import numpy as np
import tensorcircuit as tc
from matplotlib import pyplot as plt
from qiskit.transpiler.passes.analysis import num_qubits

def circuit(num_qubits, depth):
    # create a quantum circuit with a specified number of qubits
    circuit = tc.Circuit(num_qubits)

    # apply rz, ry, rz for each qubits
    for i in range(num_qubits):
        circuit.rz(i)
        circuit.ry(i)
        circuit.rz(i)

    # add a specified number of random blocks
    for i in range(depth):
        # randomly select two different qubit
        q0, q1 = random.sample(range(num_qubits), 2)
        q0, q1 = min(q0, q1), max(q0, q1)  # make sure q0 < q1 to facilitate adding parameters to each gate later
        circuit.cz(q0, q1)  # add a CZ gate between the selected two qubits
        circuit.ry(q0)
        circuit.ry(q1)
        circuit.rz(q0)
        circuit.rz(q1)

    return circuit

