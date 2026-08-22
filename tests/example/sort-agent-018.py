import sys
import time


def my_sort(arr: list[int]) -> list[int]:
    import array as _array
    import numpy as _np
    # Convert via array.array('h') (int16, fits 0-32767) then frombuffer for minimal overhead
    buf = _array.array('h', arr)
    return _np.sort(_np.frombuffer(buf, dtype=_np.int16))


if __name__ == "__main__":
    input_array = [int(x) for x in sys.argv[1:]]
    ts = time.time()
    sorted_array = my_sort(input_array)
    print(time.time() - ts, " ".join(map(str, sorted_array)))
