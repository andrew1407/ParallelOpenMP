// Lab 1: the lab's main() unchanged. Each Web Worker runs it once, as one thread of
// the parallel region, and the page prints the lines in the order they arrive.
#define main lab_main
#include "../../1-lab/Parallel_OpenMP.cpp"
#undef main

#include <emscripten.h>
// The lab's main() under a C name the page can call.
extern "C" EMSCRIPTEN_KEEPALIVE int run_main() { return lab_main(); }
