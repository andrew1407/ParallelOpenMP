/* Minimal mpi.h for the browser build of 7-lab-mpi. Each rank is a Web Worker running
   the lab's main(); MPI_Send/MPI_Recv go through postMessage (see mpi_stub.cpp). */
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
typedef int MPI_Datatype;
typedef int MPI_Comm;
typedef struct { int MPI_SOURCE, MPI_TAG, MPI_ERROR, count; } MPI_Status;
#define MPI_INT ((MPI_Datatype)4)
#define MPI_COMM_WORLD ((MPI_Comm)0)
#define MPI_SUCCESS 0
int MPI_Init(int* argc, char*** argv);
int MPI_Finalize(void);
int MPI_Comm_rank(MPI_Comm comm, int* rank);
int MPI_Comm_size(MPI_Comm comm, int* size);
int MPI_Send(const void* buf, int count, MPI_Datatype type, int dest, int tag, MPI_Comm comm);
int MPI_Recv(void* buf, int count, MPI_Datatype type, int source, int tag, MPI_Comm comm, MPI_Status* status);
double MPI_Wtime(void);
#ifdef __cplusplus
}
#endif
