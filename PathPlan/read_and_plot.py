from main3 import plot_all_smoothed_fidelities


def read_fidelities_from_txt(filename="all_fidelities_summary.txt"):
    all_results = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():  # skip the empty line
                data_str = line.split(":")[1].strip()
                # turn string to float
                fidelities = list(map(float, data_str.split(",")))
                all_results.append(fidelities)
    return all_results


data = read_fidelities_from_txt("all_fidelities_summary.txt")

if len(data) > 0:
    print(f"success load {len(data)} data")

    plot_all_smoothed_fidelities(
        all_results=data,
        target_fidelity=0.8,
        window_size=10,
        save_dir="."
    )
else:
    print("Warning: no data")
