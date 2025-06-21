problem = {
    'VQA': {
        'num_qubits': 4,  # number of qubits
        'max_interation': 500000,  # max interation
        # model list within model name
        'model_list': ['StoVec', 'SQC'],
        'converge_difference': 0.0001,
        'n': 10,  # ten times training
        'C': 0.001, # target cost function value
        'check_point': 10,  # check whether converge per 100 interation
        'loss_last': 1e10,
    },
}