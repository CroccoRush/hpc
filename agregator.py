import csv
import os
import subprocess
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


WITH_HUGE = False
DEFAULT_STEP = 64
DEFAULT_START = 4096
DEFAULT_STOP = 4096
HUGE_STEP = 1024
HUGE_START = DEFAULT_STOP + HUGE_STEP
HUGE_STOP = 4096
CPU_COUNT = 6
THREADS_START = 1
THREADS_STOP = 18
RUNS_COUNT = 3  # The number of runs for averaging

matrix_sizes = list(range(DEFAULT_START, DEFAULT_STOP + 1, DEFAULT_STEP))
if WITH_HUGE:
    matrix_sizes.extend(range(HUGE_START, HUGE_STOP + 1, HUGE_STEP))

threads_counts = range(THREADS_START, THREADS_STOP + 1)
calculation_types = ["simple"]
print_mode = False

results = {}
for calculation_type in calculation_types:
    for matrix_size in matrix_sizes:
        for threads_count in threads_counts:
            print(f"ms: {matrix_size}, tc: {threads_count}, ct: {calculation_type}")
            # Forming a command to start calculations
            command = [
                "./hpc_task1",
                str(matrix_size),
                str(threads_count),
                calculation_type,
                str(print_mode).lower(),
            ]

            # Calculations
            run_results = []
            for _ in range(RUNS_COUNT):
                result = subprocess.run(command, capture_output=True, text=True)
                try:
                    result_value = float(result.stdout.strip())
                    run_results.append(result_value)
                except ValueError:
                    pass

            # Calculate the average value and variance
            if run_results:
                avg_result = sum(run_results) / len(run_results)
                variance = np.var(run_results)
            else:
                avg_result = None
                variance = None

            # Saving the results
            key = (matrix_size, threads_count, calculation_type)
            results[key] = (avg_result, variance)
            print(
                f"ms: {matrix_size}, tc: {threads_count}, ct: {calculation_type}, "
                f"avg_result: {avg_result}, variance: {variance}"
            )

# Creating a folder to save images
if not os.path.exists("./imgs"):
    os.makedirs("./imgs")

# Plotting and saving graphs
fig, axs = plt.subplots(
    2,
    len(calculation_types),
    figsize=(30, 60),
    height_ratios=[45, 15],
    gridspec_kw={"wspace": 0.1, "hspace": 0.1},
)

for i, calculation_type in enumerate(calculation_types):
    ax1: plt.Axes = axs[0]
    ax2: plt.Axes = axs[1]

    ax1.set_title(f"Calculation Type: {calculation_type}")
    ax1.set_xlabel("Threads Count")
    ax1.set_ylabel("Average Result")
    ax1.set_yscale("log")

    ax2.set_title(f"Speedup for Calculation Type: {calculation_type}")
    ax2.set_xlabel("Threads Count")
    ax2.set_ylabel("Speedup")
    ax2.set_aspect("equal")

    x = []
    z = []
    speedup = []
    for matrix_size in matrix_sizes:
        x_line = []
        z_line = []
        speedup_line = []
        for threads_count in threads_counts:
            key = (matrix_size, threads_count, calculation_type)
            avg_result, _ = results.get(key, (None, None))
            if avg_result is not None:
                x_line.append(threads_count)
                z_line.append(avg_result)
                # Calculating the speedup relative to the result with threads_count=1
                key_single_thread = (matrix_size, THREADS_START, calculation_type)
                avg_result_single_thread, _ = results.get(
                    key_single_thread, (None, None)
                )
                if avg_result_single_thread is not None:
                    speedup_line.append(avg_result_single_thread / avg_result)
        x.append(x_line)
        z.append(z_line)
        speedup.append(speedup_line)

    # Plotting the average result
    for x_line, z_line, matrix_size in zip(x, z, matrix_sizes):
        ax1.plot(x_line, z_line, label=f"Matrix Size: {matrix_size}")

    # Adding vertical lines to the graph of the average result
    for x_val in range(CPU_COUNT, THREADS_STOP + 1, CPU_COUNT):
        ax1.axvline(x=x_val, color="red", linestyle="--", linewidth=1)

    # Plotting the speedup graph
    for x_line, speedup_line, matrix_size in zip(x, speedup, matrix_sizes):
        ax2.plot(x_line, speedup_line, label=f"Matrix Size: {matrix_size}")

    # Adding vertical lines to the speedup graph
    for x_val in range(CPU_COUNT, THREADS_STOP + 1, CPU_COUNT):
        ax2.axvline(x=x_val, color="red", linestyle="--", linewidth=1)

    # Adding the y = x line to the speedup graph
    ax2.plot(
        [1, CPU_COUNT * 2],
        [1, CPU_COUNT * 2],
        color="green",
        linestyle="dashed",
        linewidth=1,
        label="Amdahl's law",
    )

    ax1.legend()
    ax2.legend()

    # Saving graphs to a file
    filename = f"./imgs/calculation_type_{calculation_type}.png"
    plt.savefig(filename)
    print(f"Saved plot to {filename}")

plt.close(fig)

# Saving the results to a CSV file
csv_filename = f"./results/data-{datetime.now().timestamp()}.csv"
with open(csv_filename, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "Matrix Size",
            "Threads Count",
            "Calculation Type",
            "Average Result",
            "Variance",
        ]
    )
    for key, (avg_result, variance) in results.items():
        matrix_size, threads_count, calculation_type = key
        writer.writerow(
            [matrix_size, threads_count, calculation_type, avg_result, variance]
        )

print(f"Saved results to {csv_filename}")
