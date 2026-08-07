import sys
import time


def my_sort(arr: list[int]) -> list[int]:
    # LSD Radix sort, base-256, 2 passes for integers in [0, 9999].
    # Pass 1 sorts by low byte (bits 0-7, values 0-255).
    # Pass 2 sorts by high byte (bits 8-13, values 0-38 for max 9999).
    # Avoids numpy import overhead (~50-200ms in a fresh process) and
    # outperforms counting sort over k=10000 with only 296 bucket slots total.
    from itertools import chain
    b1 = [[] for _ in range(256)]
    b2 = [[] for _ in range(40)]
    for x in arr:
        b1[x & 0xFF].append(x)
    for bl in b1:
        for x in bl:
            b2[x >> 8].append(x)
    return list(chain.from_iterable(b2))


if __name__ == "__main__":
    input_array = [int(x) for x in sys.argv[1:]]
    ts = time.time()
    sorted_array = my_sort(input_array)
    print(time.time() - ts, " ".join(map(str, sorted_array)))
