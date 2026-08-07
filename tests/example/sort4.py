import sys
import time


def my_sort(arr: list[int]) -> list[int]:
    # Counting sort: O(n + k) where k=10000 (bounded integer range 0-9999)
    try:
        import numpy as np
        a = np.asarray(arr, dtype=np.intp)
        counts = np.bincount(a, minlength=10000)
        return np.repeat(np.arange(10000, dtype=np.intp), counts).tolist()
    except ImportError:
        # Pure Python fallback
        counts = [0] * 10000
        for x in arr:
            counts[x] += 1
        result = []
        for val, cnt in enumerate(counts):
            if cnt:
                result.extend([val] * cnt)
        return result


if __name__ == "__main__":
    input_array = [int(x) for x in sys.argv[1:]]
    ts = time.time()
    sorted_array = my_sort(input_array)
    print(time.time() - ts, " ".join(map(str, sorted_array)))
