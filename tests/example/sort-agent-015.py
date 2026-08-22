import sys
import time


def my_sort(arr: list[int]) -> list[int]:
    import heapq
    return heapq.nsmallest(len(arr), arr)


if __name__ == "__main__":
    input_array = [int(x) for x in sys.argv[1:]]
    ts = time.time()
    sorted_array = my_sort(input_array)
    print(time.time() - ts, " ".join(map(str, sorted_array)))
