// omp_* runtime calls for the browser build (see include/omp.h).
#include <emscripten.h>
#include "omp.h"

static int team_thread = 0, team_size = 1, max_threads = 1;

double omp_get_wtime(void) { return emscripten_get_now() / 1000.0; }
int omp_get_thread_num(void) { return team_thread; }
int omp_get_num_threads(void) { return team_size; }
int omp_get_max_threads(void) { return max_threads; }
void omp_set_num_threads(int n) { max_threads = n; }

// Called by the worker before it runs its share: "this worker is thread `tid` of `n`".
EMSCRIPTEN_KEEPALIVE void omp_stub_team(int tid, int n) { team_thread = tid; team_size = n; }
