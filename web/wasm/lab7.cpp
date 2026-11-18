// Lab 7: the lab's main() unchanged, linked against mpi_stub.cpp. Two workers run it
// as ranks 0 and 1.
#define main lab_main
#include "../../7-lab-mpi/Parallel_OpenMP.cpp"
#undef main

#include <emscripten.h>
// The lab's main() under a C name the page can call (asynchronously, through Asyncify).
extern "C" EMSCRIPTEN_KEEPALIVE int run_main() { return lab_main(0, nullptr); }
