import sys
import time


def my_sort(arr: list[int]) -> list[int]:
    # LSD radix sort, base 100, 2 passes for values 0-9999
    # Pass 1: least significant 2 digits
    BASE = 100
    buckets = [[] for _ in range(BASE)]
    for x in arr:
        buckets[x % BASE].append(x)
    arr = [x for b in buckets for x in b]
    # Pass 2: most significant 2 digits
    for b in buckets:
        b.clear()
    for x in arr:
        buckets[x // BASE].append(x)
    return [x for b in buckets for x in b]


if __name__ == "__main__":
    input_array = [int(x) for x in sys.argv[1:]]
    ts = time.time()
    sorted_array = my_sort(input_array)
    print(time.time() - ts, " ".join(map(str, sorted_array)))
