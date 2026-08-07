import sys
import time


def my_sort(arr: list[int]) -> list[int]:
    # LSD radix sort implemented as counting sort: O(n + range) for bounded integers.
    # For n=1000 integers in [0, 9999], this is far faster than O(n²) bubble sort.
    if not arr:
        return arr

    # Two-pass LSD radix sort with base 256 (handles values 0..65535 in 2 passes)
    # Pass 1: sort by lower 8 bits; Pass 2: sort by upper 8 bits
    n = len(arr)
    buf = [0] * n

    # Pass over lower 8 bits
    counts = [0] * 256
    for x in arr:
        counts[x & 0xFF] += 1
    # Build prefix sums (starting offsets for each bucket)
    pos = [0] * 256
    acc = 0
    for i in range(256):
        pos[i] = acc
        acc += counts[i]
    # Scatter elements into buffer
    for x in arr:
        d = x & 0xFF
        buf[pos[d]] = x
        pos[d] += 1

    # Pass over upper 8 bits (bits 8-15)
    counts = [0] * 256
    for x in buf:
        counts[(x >> 8) & 0xFF] += 1
    pos = [0] * 256
    acc = 0
    for i in range(256):
        pos[i] = acc
        acc += counts[i]
    result = [0] * n
    for x in buf:
        d = (x >> 8) & 0xFF
        result[pos[d]] = x
        pos[d] += 1

    return result


if __name__ == "__main__":
    input_array = [int(x) for x in sys.argv[1:]]
    ts = time.time()
    sorted_array = my_sort(input_array)
    print(time.time() - ts, " ".join(map(str, sorted_array)))
