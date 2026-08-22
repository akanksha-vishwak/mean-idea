import array
import sys
import time

import numpy as np


def my_sort(arr: list[int]) -> list[int]:
    # Convert via array.array('h') (int16, fits 0-9999) for fast C-level list→buffer
    # then sort in-place as int16 numpy array (half the memory of int32, better cache)
    buf = array.array('h', arr)
    a = np.frombuffer(buf, dtype=np.int16).copy()
    a.sort()
    return a.tolist()


if __name__ == "__main__":
    input_array = [int(x) for x in sys.argv[1:]]
    ts = time.time()
    sorted_array = my_sort(input_array)
    print(time.time() - ts, " ".join(map(str, sorted_array)))
