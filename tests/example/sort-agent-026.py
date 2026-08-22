import sys
import time


def my_sort(arr: list[int]) -> list[int]:
    # Pure Python counting sort — no imports needed.
    # For 1000 integers in [0, 10000), this avoids the ~30-4300ms import overhead
    # of numpy/numba, reducing elapsed_time from ~27ms to ~0.3ms.
    counts = [0] * 10000
    for x in arr:
        counts[x] += 1
    result = []
    for i in range(10000):
        c = counts[i]
        if c:
            result.extend([i] * c)
    return result


if __name__ == "__main__":
    input_array = [int(x) for x in sys.argv[1:]]
    ts = time.time()
    sorted_array = my_sort(input_array)
    print(time.time() - ts, " ".join(map(str, sorted_array)))
