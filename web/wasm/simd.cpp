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
static volatile float sink;

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

// 6-lab's main(): `reps` calls (the lab does RANGE * 10) whose sums are thrown away.
EMSCRIPTEN_KEEPALIVE double simd_bench(int reps, int parallel) {
  const double start = omp_get_wtime();
  if (parallel) for (int i = 0; i < reps; i++) { sums s; calcSumsPar(&s); }
  else for (int i = 0; i < reps; i++) { sums s; calcSumsSeq(&s); }
  return omp_get_wtime() - start;
}
// The same loop but keeping each result, so the optimizer can't drop the calls.
EMSCRIPTEN_KEEPALIVE double simd_bench_kept(int reps, int parallel) {
  const double start = omp_get_wtime();
  for (int i = 0; i < reps; i++) { sums s; parallel ? calcSumsPar(&s) : calcSumsSeq(&s); sink = s.sum1 + s.sum2 + s.sum3_1 + s.sum3_2 + s.sum4_1 + s.sum4_2; }
  return omp_get_wtime() - start;
}

// Not the lab's code: calcSumsPar() rewritten without the pointer select, so the
// reductions are plain += and the loop can be vectorized. For comparison only.
EMSCRIPTEN_KEEPALIVE double simd_rewrite() {
  float sum1 = 0, sum2 = 0, sum3_1 = 0, sum3_2 = 0, sum4_1 = 0, sum4_2 = 0;
  const double start = omp_get_wtime();
  #pragma omp simd reduction(+: sum1, sum2, sum3_1, sum3_2, sum4_1, sum4_2)
  for (int i = 0; i < RANGE; i++) {
    const float el = 1.0 / (i + 1);
    const float elSub = 1.0 / (RANGE - i);
    const bool isEven = (i + 1) % 2 == 0;
    sum1 += el;
    sum2 += elSub;
    sum3_1 += isEven ? elSub : 0.0f;
    sum3_2 += isEven ? 0.0f : elSub;
    sum4_1 += isEven ? el : 0.0f;
    sum4_2 += isEven ? 0.0f : el;
  }
  const double end = omp_get_wtime();
  S.sum1 = sum1; S.sum2 = sum2; S.sum3_1 = sum3_1; S.sum3_2 = sum3_2; S.sum4_1 = sum4_1; S.sum4_2 = sum4_2;
  return end - start;
}
EMSCRIPTEN_KEEPALIVE double simd_rewrite_bench(int reps) {
  const double start = omp_get_wtime();
  for (int i = 0; i < reps; i++) { simd_rewrite(); sink = S.sum1; }
  return omp_get_wtime() - start;
}
}
