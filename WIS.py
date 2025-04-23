import random
import time
import tracemalloc
import matplotlib.pyplot as plt


class Interval:
    def __init__(self, start, end, weight):
        self.start = start
        self.end = end
        self.weight = weight


def generate_intervals(n, max_time=1000, max_weight=100):
    intervals = []
    for _ in range(n):
        start = random.randint(0, max_time - 1)
        end = random.randint(start + 1, max_time)
        weight = random.randint(1, max_weight)
        intervals.append(Interval(start, end, weight))
    return intervals


def find_compatible_previous(intervals):    
    # Time complexity: O(n^2)
    # Find the index of the last compatible interval for each interval. Compatible meaning the previous interval's end time is <= current interval's start time
    compatible = []
    for j in range(len(intervals)):
        last_compatible = -1  # no compatible interval
        
        for i in range(j):
            if intervals[i].end <= intervals[j].start:
                last_compatible = i
        compatible.append(last_compatible)
    return compatible      # Outputs array of indexes of last compatible interval for each interval


def solve_weighted_intervals(intervals):
    # Sort intervals by end time to ensure we process them in order
    intervals.sort(key=lambda x: x.end)
    n = len(intervals)
    compatible = find_compatible_previous(intervals)

    # Initialize the optimal value array for memoization
    optimal_values = [0] * (n + 1)
    
    optimal_values[0] = 0 # Base case

    # Recurrence equation: for each interval j, decide whether to include it or not
    for j in range(1, n + 1):
        # Option 1: Include the current interval (j-1) and add its weight to the optimal solution of the last compatible interval
        include_weight = intervals[j - 1].weight + optimal_values[compatible[j - 1] + 1]
        # Option 2: Exclude the current interval, take the optimal solution up to the previous interval
        exclude_weight = optimal_values[j - 1]
        # Optimal function: choose the maximum of including or excluding the current interval
        optimal_values[j] = max(include_weight, exclude_weight)

    return optimal_values, compatible


def brute_force_intervals(intervals, ind, last_end):
    if ind == len(intervals):
        return 0  # Base case: no more intervals

    # Exclude the current interval
    exclude = brute_force_intervals(intervals, ind + 1, last_end)

    # Include the current interval if compatible
    include = 0
    if intervals[ind].start >= last_end:
        include = intervals[ind].weight + brute_force_intervals(intervals, ind + 1, intervals[ind].end)

    return max(include, exclude)


def reconstruct_optimal_solution(intervals, compatible, optimal_values):
    solution = []
    j = len(intervals)
    
    # Backtracking: reconstruct the solution by checking which choice was made at each step
    while j > 0:
        # If including the interval at j-1 gave a better result, add it to the solution
        if intervals[j - 1].weight + optimal_values[compatible[j - 1] + 1] > optimal_values[j - 1]:
            solution.append(intervals[j - 1])
            j = compatible[j - 1] + 1
        else:
            j -= 1

    solution.reverse()  # Reverse to get intervals in increasing order
    return solution


def performance_analysis(dp_sizes, compare_sizes):
    dp_runtimes = []
    compare_dp_runtimes = []
    brute_runtimes = []
    memories = []

    # Dynamic programming performance
    for n in dp_sizes:
        intervals = generate_intervals(n)
        tracemalloc.start()
        start_time = time.perf_counter()
        optimal_values, compatible = solve_weighted_intervals(intervals)
        _ = reconstruct_optimal_solution(intervals, compatible, optimal_values)
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        dp_runtimes.append(end_time - start_time)
        memories.append(peak / 1024)  # Convert to KB

    # Dynamic programming and brute force performance (smaller sizes for comparison)
    for n in compare_sizes:
        # Dynamic programming
        intervals = generate_intervals(n)
        start_time = time.perf_counter()
        optimal_values, compatible = solve_weighted_intervals(intervals)
        _ = reconstruct_optimal_solution(intervals, compatible, optimal_values)
        end_time = time.perf_counter()
        compare_dp_runtimes.append(end_time - start_time)

        # Brute force
        intervals = generate_intervals(n)
        intervals.sort(key=lambda x: x.end)
        start_time = time.perf_counter()
        _ = brute_force_intervals(intervals, 0, 0)
        end_time = time.perf_counter()
        brute_runtimes.append(end_time - start_time)

    return dp_runtimes, compare_dp_runtimes, brute_runtimes, memories


def plot_results(dp_sizes, compare_sizes, dp_runtimes, compare_dp_runtimes, brute_runtimes, memories):
    # Plot 1: Dynamic programming runtime (large sizes)
    fig1, ax1 = plt.subplots()
    ax1.plot(dp_sizes, dp_runtimes, marker='o', color='tab:blue', label='Dynamic Programming Runtime (O(n²))')
    ax1.set_xlabel('Input Size (Number of Intervals)')
    ax1.set_ylabel('Runtime (seconds)')
    ax1.set_title('Dynamic Programming Runtime (O(n²))')
    ax1.legend()
    ax1.grid(True)
    plt.tight_layout()
    plt.savefig("dp_runtime_plot.png")
    plt.close()

    # Plot 2: Memory usage of dynamic programming
    fig2, ax2 = plt.subplots()
    ax2.plot(dp_sizes, memories, marker='s', color='tab:red', label='Memory Usage')
    ax2.set_xlabel('Input Size (Number of Intervals)')
    ax2.set_ylabel('Memory Usage (KB)')
    ax2.set_title('Dynamic Programming Memory Usage')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig("memory_plot.png")
    plt.close()

    # Plot 3: Runtime comparison (dynamic programming vs. brute force, smaller sizes)
    fig3, ax3 = plt.subplots()
    ax3.plot(compare_sizes, compare_dp_runtimes, marker='o', color='tab:blue', label='Dynamic Programming (O(n²))')
    ax3.plot(compare_sizes, brute_runtimes, marker='^', color='tab:orange', label='Brute Force (O(2^n))')
    ax3.set_xlabel('Input Size (Number of Intervals)')
    ax3.set_ylabel('Runtime (seconds)')
    ax3.set_title('Runtime Comparison: Dynamic Programming vs. Brute Force')
    ax3.legend()
    ax3.grid(True)
    plt.tight_layout()
    plt.savefig("runtime_comparison_plot.png")
    plt.close()

def visualize_optimal_intervals(intervals, optimal_intervals):
    fig, ax = plt.subplots(figsize=(10, len(intervals) * 0.4))

    # Plot each interval as a horizontal bar
    for i, interval in enumerate(intervals):
        # Determine if this interval is in the optimal solution
        is_optimal = interval in optimal_intervals
        color = 'green' if is_optimal else 'gray'
        alpha = 1.0 if is_optimal else 0.3
        
        # Draw the interval as a horizontal bar
        ax.barh(i, interval.end - interval.start, left=interval.start, height=0.4, color=color, alpha=alpha)
        ax.text((interval.start + interval.end) / 2, i, f'w={interval.weight}', 
                ha='center', va='center', color='white' if is_optimal else 'black', fontsize=8)

    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Intervals')
    ax.set_yticks(range(len(intervals)))
    ax.set_yticklabels([f'[{interval.start}, {interval.end}]' for interval in intervals])
    ax.set_title('Weighted Interval Scheduling: Optimal Intervals')
    
    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig('optimal_intervals_plot.png')
    plt.close()


# -------------------------------------
#  Running algorithm and visualizations
# -------------------------------------

if __name__ == "__main__":
    # Performance analysis
    dp_sizes = [100, 1000, 2000, 4000, 6000, 8000]
    compare_sizes = [10, 15, 20, 25, 30, 35, 40]  # Sizes for DP vs. brute force comparison
    dp_runtimes, compare_dp_runtimes, brute_runtimes, memories = performance_analysis(dp_sizes, compare_sizes)
    plot_results(dp_sizes, compare_sizes, dp_runtimes, compare_dp_runtimes, brute_runtimes, memories)

    # Visualize optimal intervals for 15 intervals
    intervals = generate_intervals(15, max_time=100, max_weight=50)
    optimal_values, compatible = solve_weighted_intervals(intervals)
    optimal_intervals = reconstruct_optimal_solution(intervals, compatible, optimal_values)
    visualize_optimal_intervals(intervals, optimal_intervals)