import sys
import time


def my_sort(arr: list[int]) -> list[int]:
    # Use numpy's optimized C-level introsort/radix sort
    try:
        import numpy as np
        # uint16 fits 0-9999 and uses half the memory of int32, enabling
        # numpy's radix sort path for small integer types → faster for n=1000
        a = np.array(arr, dtype=np.uint16)
        return np.sort(a).tolist()
    except ImportError:
        # Pure Python fallback: counting sort O(n + k)
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
