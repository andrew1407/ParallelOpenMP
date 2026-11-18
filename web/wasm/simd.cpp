// Labs 5 and 6: the lab's calcSumsSeq() and calcSumsPar() (`#pragma omp simd reduction`),
// built several times with different flags. -fopenmp-simd makes clang honour the simd
// pragma without needing an OpenMP runtime; -msimd128 lets it emit WebAssembly SIMD.
#include <emscripten.h>
#define main lab_main
#include LAB_SOURCE
#undef main

#include <emscripten.h>
// The lab's main() under a C name the page can call.
extern "C" EMSCRIPTEN_KEEPALIVE int run_main() { return lab_main(); }

static sums S;

extern "C" {
EMSCRIPTEN_KEEPALIVE double simd_seq() { S.clear(); return calcSumsSeq(&S); }
EMSCRIPTEN_KEEPALIVE double simd_par() { S.clear(); return calcSumsPar(&S); }
EMSCRIPTEN_KEEPALIVE const float* simd_sums() { return &S.sum1; }
EMSCRIPTEN_KEEPALIVE int simd_enabled() {
#ifdef __wasm_simd128__
  return 1;
#else
  return 0;
#endif
}

// 6-lab's main() loop with `reps` calls (the lab does RANGE * 10), keeping each result
// in a checksum as the lab does.
EMSCRIPTEN_KEEPALIVE double simd_bench(int reps, int parallel) {
  volatile float checksum = 0;
  const double start = omp_get_wtime();
  for (int i = 0; i < reps; i++) {
    sums s;
    parallel ? calcSumsPar(&s) : calcSumsSeq(&s);
    checksum = checksum + s.sum1 + s.sum2 + s.sum3_1 + s.sum3_2 + s.sum4_1 + s.sum4_2;
  }
  return omp_get_wtime() - start;
}

}
