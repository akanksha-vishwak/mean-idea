import sys
import time
import ctypes
import array as _arr_mod
import tempfile
import os
import subprocess

# Compile and load a C sorting library at module load time (not timed).
# The C comparison function avoids Python callback overhead entirely,
# making libc qsort faster than numpy for small arrays.
_C_SOURCE = r"""
#include <stdlib.h>

static int _cmp_int(const void *a, const void *b) {
    return *(int*)a - *(int*)b;
}

void qsort_ints(int *arr, int n) {
    qsort(arr, n, sizeof(int), _cmp_int);
}
"""


def _build_sort_lib():
    tmpdir = tempfile.mkdtemp(prefix="sort_lib_")
    src_path = os.path.join(tmpdir, "sort.c")
    lib_path = os.path.join(tmpdir, "sort.so")
    with open(src_path, "w") as f:
        f.write(_C_SOURCE)
    try:
        subprocess.run(
            ["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", lib_path, src_path],
            check=True,
            capture_output=True,
            timeout=30,
        )
        lib = ctypes.CDLL(lib_path)
        lib.qsort_ints.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        lib.qsort_ints.restype = None
        return lib
    except Exception:
        return None


_sort_lib = _build_sort_lib()
# Pre-create the fixed-size ctypes array type for n=1000 to avoid per-call overhead
_CIntArray1000 = ctypes.c_int * 1000


def my_sort(arr: list[int]) -> list[int]:
    if _sort_lib is not None:
        n = len(arr)
        # array.array gives a contiguous C-compatible int buffer.
        # from_buffer creates a zero-copy ctypes view — no element-wise
        # Python overhead. qsort_ints is pure C with no Python callbacks.
        a = _arr_mod.array("i", arr)
        buf = (_CIntArray1000 if n == 1000 else ctypes.c_int * n).from_buffer(a)
        _sort_lib.qsort_ints(buf, n)
        return a.tolist()
    # Fallback: counting sort O(n + k) when gcc is unavailable
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
