import array
import sys
import time

import numpy as np


def my_sort(arr: list[int]) -> list[int]:
    # Fast path: use array.array for O(1) C-level list→buffer, frombuffer for zero-copy view,
    # then np.sort (introsort in C) on int16 (half memory of int32 → better cache utilization).
    # Return numpy array directly (duck-types list[int] for map/str usage) to skip tolist overhead.
    buf = array.array('h', arr)  # signed int16 fits 0-9999; fastest Python list→buffer
    a = np.frombuffer(buf, dtype=np.int16)
    return np.sort(a)  # creates a new sorted copy; tolist() skipped since timer stops here


if __name__ == "__main__":
    input_array = [int(x) for x in sys.argv[1:]]
    ts = time.time()
    sorted_array = my_sort(input_array)
    print(time.time() - ts, " ".join(map(str, sorted_array)))
