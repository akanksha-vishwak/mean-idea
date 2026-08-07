import sys
import time


def my_sort(arr: list[int]) -> list[int]:
    try:
        import numpy as np
        # LSD radix sort, base-256, 2 passes for values 0-9999 (max = 0x270F)
        a = np.array(arr, dtype=np.uint16)
        # Pass 1: stable sort by lower byte (0-255); numpy uses counting sort for uint8
        idx = np.argsort(a.astype(np.uint8), kind='stable')
        a = a[idx]
        # Pass 2: stable sort by upper byte (0-39 for range 0-9999)
        idx = np.argsort((a >> 8).astype(np.uint8), kind='stable')
        a = a[idx]
        return a.tolist()
    except ImportError:
        # Pure-Python LSD radix sort, base-100, 2 passes: O(2*(n+100))
        b = [[] for _ in range(100)]
        for x in arr:
            b[x % 100].append(x)
        mid = []
        mid_extend = mid.extend
        for bucket in b:
            mid_extend(bucket)
        b2 = [[] for _ in range(100)]
        for x in mid:
            b2[x // 100].append(x)
        out = []
        out_extend = out.extend
        for bucket in b2:
            out_extend(bucket)
        return out


if __name__ == "__main__":
    input_array = [int(x) for x in sys.argv[1:]]
    ts = time.time()
    sorted_array = my_sort(input_array)
    print(time.time() - ts, " ".join(map(str, sorted_array)))
