import sys
import time


def my_sort(arr: list[int]) -> list[int]:
    # Pure-Python counting sort: avoids numpy/numba import overhead (~50ms per
    # fresh subprocess) which dominates the actual O(n+k) sort for n=1000.
    # Direct extend with multiplied lists is faster than itertools.chain+repeat.
    counts = [0] * 10000
    for x in arr:
        counts[x] += 1
    result = []
    for val in range(10000):
        c = counts[val]
        if c:
            result.extend([val] * c)
    return result


if __name__ == "__main__":
    input_array = [int(x) for x in sys.argv[1:]]
    ts = time.time()
    sorted_array = my_sort(input_array)
    print(time.time() - ts, " ".join(map(str, sorted_array)))
