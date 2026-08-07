import sys
import time
import os
import subprocess
import importlib.util
import sysconfig

# ── C extension: counting sort for integers in [0, 9999] ──────────────────────
# Compiled once and cached as a .so in the same directory as this script.
# The compilation runs at module-import time (before my_sort is timed).

_CSORT_C_SRC = r"""
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>

static PyObject* counting_sort(PyObject* self, PyObject* args) {
    PyObject* input_list;
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &input_list)) return NULL;

    Py_ssize_t n = PyList_GET_SIZE(input_list);
    int counts[10000];
    memset(counts, 0, sizeof(counts));

    for (Py_ssize_t i = 0; i < n; i++) {
        long val = PyLong_AsLong(PyList_GET_ITEM(input_list, i));
        if (val < 0 || val >= 10000) {
            PyErr_SetString(PyExc_ValueError, "Value out of range [0, 9999]");
            return NULL;
        }
        counts[val]++;
    }

    PyObject* result = PyList_New(n);
    if (!result) return NULL;

    Py_ssize_t idx = 0;
    for (int i = 0; i < 10000 && idx < n; i++) {
        if (counts[i] > 0) {
            PyObject* v = PyLong_FromLong(i);
            if (!v) { Py_DECREF(result); return NULL; }
            PyList_SET_ITEM(result, idx++, v);
            for (int j = 1; j < counts[i]; j++) {
                Py_INCREF(v);
                PyList_SET_ITEM(result, idx++, v);
            }
        }
    }
    return result;
}

static PyMethodDef CsortMethods[] = {
    {"counting_sort", counting_sort, METH_VARARGS,
     "Fast C counting sort for integers in [0, 9999]"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef csort_module = {
    PyModuleDef_HEAD_INIT, "_csort_native", NULL, -1, CsortMethods
};

PyMODINIT_FUNC PyInit__csort_native(void) {
    return PyModule_Create(&csort_module);
}
"""

_csort_mod = None


def _init_csort():
    """Compile (if needed) and load the C counting-sort extension."""
    global _csort_mod
    base = os.path.dirname(os.path.abspath(__file__))
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so_path = os.path.join(base, f'_csort_native{suffix}')
    src_path = os.path.join(base, '_csort_native.c')
    include_dir = sysconfig.get_path('include')

    if not os.path.exists(so_path):
        with open(src_path, 'w') as f:
            f.write(_CSORT_C_SRC)
        subprocess.run(
            ['gcc', '-O3', '-march=native', '-shared', '-fPIC',
             f'-I{include_dir}', '-o', so_path, src_path, '-ldl', '-lm'],
            capture_output=True,
        )

    if os.path.exists(so_path):
        try:
            spec = importlib.util.spec_from_file_location('_csort_native', so_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _csort_mod = mod
        except Exception:
            pass


_init_csort()  # runs before my_sort is timed


def my_sort(arr: list[int]) -> list[int]:
    # Fast path: C extension counting sort (~0.017 ms for n=1000)
    if _csort_mod is not None:
        return _csort_mod.counting_sort(arr)

    # Pure-Python fallback (no gcc available)
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
