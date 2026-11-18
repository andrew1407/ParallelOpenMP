// Lab 3: the lab's multiply() and cleanMatrix() with SIZE made a runtime value (gen.py),
// plus the share of multiplyParallel()'s collapse(3) loop that one OpenMP thread runs.
#include <algorithm>
#include <emscripten.h>
#define main lab_main
#include "gen/lab3.cpp"
#undef main

static BASE_TYPE **A = nullptr, **B = nullptr, **C = nullptr;
static int allocated = 0;

static void release() {
  for (int i = 0; i < allocated; i++) { delete[] A[i]; delete[] B[i]; delete[] C[i]; }
  delete[] A; delete[] B; delete[] C;
  allocated = 0;
}

extern "C" {
// Allocate and fill A, B and C the way the lab's main() does, from a given seed so that
// every worker builds the same matrices (the lab seeds with time(NULL)).
EMSCRIPTEN_KEEPALIVE void lab3_setup(int size, unsigned seed) {
  release();
  SIZE = size;
  srand(seed);
  A = new BASE_TYPE*[SIZE]; B = new BASE_TYPE*[SIZE]; C = new BASE_TYPE*[SIZE];
  for (int i = 0; i < SIZE; i++) {
    A[i] = new BASE_TYPE[SIZE]; B[i] = new BASE_TYPE[SIZE]; C[i] = new BASE_TYPE[SIZE];
    for (int u = 0; u < SIZE; u++) {
      A[i][u] = (BASE_TYPE) rand() % 100;
      B[i][u] = (BASE_TYPE) rand() % 100;
      C[i][u] = 0;
    }
  }
  allocated = SIZE;
}

// Sequential run: the lab's own multiply().
EMSCRIPTEN_KEEPALIVE double lab3_seq() { cleanMatrix(C); return multiply(A, B, C); }

// Thread `tid` of `n` in multiplyParallel(): `parallel for collapse(3)` with the default
// static schedule gives each thread one contiguous block of the SIZE³ (i, u, t) space,
// the first SIZE³ % n threads one iteration more. The atomic += becomes a private C that
// the page adds up after the join.
EMSCRIPTEN_KEEPALIVE double lab3_chunk(int tid, int n) {
  cleanMatrix(C);
  const double start = omp_get_wtime();
  const long long total = (long long)SIZE * SIZE * SIZE, base = total / n, extra = total % n;
  const long long lo = tid * base + std::min<long long>(tid, extra), hi = lo + base + (tid < extra);
  // walk the block as the lab's nested loops, entering and leaving it part-way through a row
  for (long long k = lo; k < hi;) {
    const int i = k / ((long long)SIZE * SIZE), u = (k / SIZE) % SIZE, t0 = k % SIZE;
    const int t1 = (int)std::min<long long>(SIZE, t0 + (hi - k));
    for (int t = t0; t < t1; t++)
      C[i][u] += A[i][t] * B[t][u];
    k += t1 - t0;
  }
  return omp_get_wtime() - start;
}

// Copy C row by row into a flat buffer for the page.
EMSCRIPTEN_KEEPALIVE void lab3_result(BASE_TYPE* out) {
  for (int i = 0; i < SIZE; i++) std::copy(C[i], C[i] + SIZE, out + (long long)i * SIZE);
}
}
