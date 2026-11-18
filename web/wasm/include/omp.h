/* Minimal omp.h for the browser build. Emscripten ships no OpenMP runtime, so the
   labs compile without -fopenmp (the pragmas are ignored) and these calls are
   provided by omp_stub.c. Thread parallelism comes from Web Workers instead: each
   worker sets its OpenMP thread number and team size before running its share. */
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
double omp_get_wtime(void);
int omp_get_thread_num(void);
int omp_get_num_threads(void);
int omp_get_max_threads(void);
void omp_set_num_threads(int n);
#ifdef __cplusplus
}
#endif
