// MPI calls for the browser build of 7-lab-mpi. The worker running a rank sets
// Module.mpiRank, Module.mpiSend (posts the buffer to the page, which forwards it to
// the destination rank) and Module.mpiRecv (a promise for the matching message).
// MPI_Recv blocks the rank's main() with Asyncify until that promise resolves.
#include <emscripten.h>
#include "mpi.h"

EM_JS(int, mpi_rank_js, (), { return Module.mpiRank; });
EM_JS(int, mpi_size_js, (), { return Module.mpiSize; });
EM_JS(void, mpi_send_js, (const void* buf, int bytes, int dest, int tag), {
  Module.mpiSend(dest, tag, HEAPU8.slice(buf, buf + bytes));
});
EM_ASYNC_JS(int, mpi_recv_js, (void* buf, int bytes, int source, int tag), {
  const data = await Module.mpiRecv(source, tag);
  const n = Math.min(bytes, data.length);
  HEAPU8.set(data.subarray(0, n), buf);
  return n;
});
EM_JS(double, mpi_wtime_js, (), { return performance.now() / 1000; });

extern "C" {
int MPI_Init(int*, char***) { return MPI_SUCCESS; }
int MPI_Finalize(void) { return MPI_SUCCESS; }
int MPI_Comm_rank(MPI_Comm, int* rank) { *rank = mpi_rank_js(); return MPI_SUCCESS; }
int MPI_Comm_size(MPI_Comm, int* size) { *size = mpi_size_js(); return MPI_SUCCESS; }
int MPI_Send(const void* buf, int count, MPI_Datatype type, int dest, int tag, MPI_Comm) {
  mpi_send_js(buf, count * type, dest, tag);
  return MPI_SUCCESS;
}
int MPI_Recv(void* buf, int count, MPI_Datatype type, int source, int tag, MPI_Comm, MPI_Status* status) {
  const int got = mpi_recv_js(buf, count * type, source, tag);
  if (status) { status->MPI_SOURCE = source; status->MPI_TAG = tag; status->MPI_ERROR = MPI_SUCCESS; status->count = got / type; }
  return MPI_SUCCESS;
}
double MPI_Wtime(void) { return mpi_wtime_js(); }
}
