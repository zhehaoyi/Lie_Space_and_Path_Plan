# save the loss value and iterations
import os


def record_values(loss_or_iteration, num_qubits, value_type, model_list, depth):
    # create a file
    output_dir = 'result/value_for_avg_plot'
    os.makedirs(output_dir, exist_ok=True)
    # save the values (iterations or losses) at convergence to a file
    filename = f'result/value_for_avg_plot/{value_type}_value_for_avg_plot_{num_qubits}qubits_{depth}.txt'
    with open(filename, 'w') as f:
        for i, model_name in enumerate(model_list):
            f.write(f"Model: {model_name}\n")
            # write the value data corresponding to this model
            line = "[" + ", ".join(map(str, loss_or_iteration[i])) + "]"
            f.write(line + "\n")
