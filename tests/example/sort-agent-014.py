import sys
import time


def my_sort(arr: list[int]) -> list[int]:
    import numpy as np
    a = np.asarray(arr, dtype=np.int32)
    counts = np.bincount(a, minlength=10000)
    return np.repeat(np.arange(len(counts), dtype=np.int32), counts).tolist()


if __name__ == "__main__":
    input_array = [int(x) for x in sys.argv[1:]]
    ts = time.time()
    sorted_array = my_sort(input_array)
    print(time.time() - ts, " ".join(map(str, sorted_array)))
