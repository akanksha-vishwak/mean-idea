import sys
import time


def my_sort(arr: list[int]) -> list[int]:
    # Counting sort with Numba JIT compilation for near-C speed.
    # cache=True persists compiled LLVM bitcode to disk so subsequent
    # process invocations skip recompilation entirely.
    try:
        import numpy as np
        import numba as nb

        @nb.njit(cache=True)
        def _counting_sort_jit(a):
            counts = np.zeros(10000, dtype=np.int64)
            for i in range(len(a)):
                counts[a[i]] += 1
            result = np.empty(len(a), dtype=np.int64)
            pos = 0
            for v in range(10000):
                c = counts[v]
                for _ in range(c):
                    result[pos] = v
                    pos += 1
            return result

        a = np.asarray(arr, dtype=np.int64)
        return _counting_sort_jit(a).tolist()

    except ImportError:
        # Fall back to NumPy vectorised counting sort (still O(n+k), no pure-Python loops)
        try:
            import numpy as np
            a = np.asarray(arr, dtype=np.intp)
            counts = np.bincount(a, minlength=10000)
            return np.repeat(np.arange(10000, dtype=np.intp), counts).tolist()
        except ImportError:
            # Pure Python last resort
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
