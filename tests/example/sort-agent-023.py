import sys
import time


def my_sort(arr: list[int]) -> list[int]:
    # Use Python's sorted() built-in (C-implemented Timsort), distinct from list.sort()
    return sorted(arr)


if __name__ == "__main__":
    input_array = [int(x) for x in sys.argv[1:]]
    ts = time.time()
    sorted_array = my_sort(input_array)
    print(time.time() - ts, " ".join(map(str, sorted_array)))
