import random
import bisect
import time
import tracemalloc
import matplotlib.pyplot as plt


class Interval:
    def __init__(self, start, end, weight):
        self.start = start
        self.end = end
        self.weight = weight

    def __repr__(self):
        return f"[{self.start}, {self.end}, w={self.weight}]"


def generate_intervals(n, max_time=1000, max_weight=100):
    intervals = []
    for _ in range(n):
        start = random.randint(0, max_time - 1)
        end = random.randint(start + 1, max_time)
        weight = random.randint(1, max_weight)
        intervals.append(Interval(start, end, weight))
    return intervals


def compute_previous_intervals(intervals):
    # Intervals must be sorted by end time
    start_times = [interval.start for interval in intervals]
    p = []
    for j in range(len(intervals)):
        i = bisect.bisect_right([intervals[k].end for k in range(len(intervals))], intervals[j].start) - 1
        p.append(i)
    return p


def weighted_interval_scheduling(intervals):
    # Sort by end time
    intervals.sort(key=lambda x: x.end)
    n = len(intervals)
    p = compute_previous_intervals(intervals)

    M = [0] * (n + 1)
    for j in range(1, n + 1):
        incl = intervals[j - 1].weight + M[p[j - 1] + 1]
        M[j] = max(incl, M[j - 1])
    return M, p


def reconstruct_solution(intervals, p, M):
    solution = []
    j = len(intervals)
    
    while j > 0:
        if intervals[j - 1].weight + M[p[j - 1] + 1] > M[j - 1]:
            solution.append(intervals[j - 1])
            j = p[j - 1] + 1
        else:
            j -= 1

    solution.reverse()  # Optional: if you want the solution in increasing order
    return solution



# -------------------------------
# 🔍 Experimental Analysis
# -------------------------------

def performance_analysis(sizes):
    runtimes = []
    memories = []

    for n in sizes:
        intervals = generate_intervals(n)
        tracemalloc.start()
        start_time = time.perf_counter()
        M, p = weighted_interval_scheduling(intervals)
        _ = reconstruct_solution(intervals, p, M)
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        runtimes.append(end_time - start_time)
        memories.append(peak / 1024)  # Convert to KB

    return runtimes, memories


def plot_results(sizes, runtimes, memories):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Input Size (Number of Intervals)')
    ax1.set_ylabel('Runtime (seconds)', color='tab:blue')
    ax1.plot(sizes, runtimes, marker='o', color='tab:blue', label='Runtime')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Memory Usage (KB)', color='tab:red')
    ax2.plot(sizes, memories, marker='s', linestyle='--', color='tab:red', label='Memory Usage')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Weighted Interval Scheduling Performance')
    fig.tight_layout()
    plt.savefig("performance_plot.png")



# -------------------------------
# 🧪 Run and Visualize
# -------------------------------

if __name__ == "__main__":
    test_sizes = [100, 200, 500, 1000, 2000, 5000]
    runtimes, memories = performance_analysis(test_sizes)
    plot_results(test_sizes, runtimes, memories)
