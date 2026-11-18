// Lab 2: the lab's main() with threads, aLen and the arrays made adjustable (gen.py).
// Worker k runs it as OpenMP thread k of `threads`, so it fills and smooths only its
// own chunk, and the page keeps the report block that main() prints for thread k.
#include <vector>
#include <emscripten.h>
int lab2_threads = 8, lab2_len = 16000;
#define main lab_main
#include "gen/lab2.cpp"
#undef main

#include <emscripten.h>
// The lab's main() under a C name the page can call.
extern "C" EMSCRIPTEN_KEEPALIVE int run_main() { return lab_main(); }

extern "C" EMSCRIPTEN_KEEPALIVE void lab2_set(int threads, int len) { lab2_threads = threads; lab2_len = len; }
