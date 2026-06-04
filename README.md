# ParallelOpenMP

A set of C++ parallel-computing labs built with OpenMP and MPI. Each lab is a self-contained program that compares a sequential implementation against a parallel one and reports execution times, covering multithreading, SIMD vectorization, recursive task parallelism, and message passing between processes.

## Labs

| Lab | Topic | Demonstrates |
| --- | --- | --- |
| [`1-lab`](1-lab/Parallel_OpenMP.cpp) | Hello world | Basic `#pragma omp parallel` region with a `critical` section, thread count taken from `std::thread::hardware_concurrency()`. |
| [`2-lab`](2-lab/Parallel_OpenMP.cpp) | Uneven workload split | Manual chunking of an array across 8 threads using `master` and `barrier`; per-thread fill and stencil-calculation timing. |
| [`3-lab`](3-lab/Parallel_OpenMP.cpp) | Matrix multiplication | `parallel for collapse(3)` with `atomic` accumulation over a 500x500 matrix; sequential vs. 1/2/4/8-thread comparison. |
| [`4-lab`](4-lab/Parallel_OpenMP.cpp) | Recursive fractals | Koch snowflake and Towers of Hanoi parallelized with `#pragma omp task ... if(...)`; Koch output rendered to an SVG file. |
| [`5-lab`](5-lab/Parallel_OpenMP.cpp) | SIMD reduction | Series summation vectorized with `#pragma omp simd reduction(...)`, sequential vs. parallel timing. |
| [`6-lab`](6-lab/Parallel_OpenMP.cpp) | SIMD benchmarking | Same SIMD reduction kernel as lab 5 run in a repeated benchmark loop; ships with a Linux compile script and Intel `icc` optimization-flag examples. |
| [`7-lab-mpi`](7-lab-mpi/Parallel_OpenMP.cpp) | MPI message passing | Two ranks generate array halves and exchange them with `MPI_Send`/`MPI_Recv`, then each sums its portion and reports `MPI_Wtime`. |
| [`control-task`](control-task/Parallel_OpenMP.cpp) | Layered network pass | Multi-layer feed-forward pass using `parallel for reduction` and a `tanh`-style activation, with `if`-toggled parallelism. |

## Tech stack

- C++ with OpenMP (`#pragma omp` directives, `omp_*` runtime calls).
- MPI for the message-passing lab (`mpi.h`; the Visual Studio project links `msmpi.lib`).
- Visual Studio solutions/projects (`.sln`, `.vcxproj`) per lab, with OpenMP support enabled (`<OpenMPSupport>true</OpenMPSupport>`).
- Timing via `omp_get_wtime()` and `MPI_Wtime()`.

## Build and run

### Visual Studio (Windows)

Open the `Parallel_OpenMP.sln` inside any lab folder and build/run it. OpenMP support is already enabled in the project settings; the MPI lab additionally requires MS-MPI (`msmpi.lib`).

### Command line (Linux/macOS)

Most labs compile directly with an OpenMP-capable compiler:

```sh
c++ -o out -fopenmp Parallel_OpenMP.cpp
./out
```

[`6-lab`](6-lab) includes a helper script that uses the system compiler by default and Intel `icc` when arguments are passed:

```sh
cd 6-lab
./compile.sh             # c++ -o out -fopenmp Parallel_OpenMP.cpp
./compile.sh -O1 -ipo    # icc -O1 -ipo -fopenmp Parallel_OpenMP.cpp -o out
```

See [`6-lab/commands.txt`](6-lab/commands.txt) for further `icc` optimization-flag examples (`-O0`, `-ipo`, `-prof-gen`/`-prof-use`, `-parallel`).

The MPI lab is compiled and launched with the MPI toolchain (built for two ranks):

```sh
cd 7-lab-mpi
mpic++ -o out Parallel_OpenMP.cpp
mpirun -np 2 ./out
```

## Project structure

```
.
├── 1-lab/            # OpenMP hello world (parallel + critical)
├── 2-lab/            # Manual array chunking across threads
├── 3-lab/            # Matrix multiplication (collapse + atomic)
├── 4-lab/            # Koch snowflake & Hanoi towers (omp task)
│   ├── koch.cpp
│   ├── hanoi.cpp
│   └── Parallel_OpenMP.cpp
├── 5-lab/            # SIMD reduction
├── 6-lab/            # SIMD reduction benchmark + compile.sh / commands.txt
├── 7-lab-mpi/        # MPI send/recv between two ranks
└── control-task/     # Layered feed-forward pass
```

Each lab folder contains a [`Parallel_OpenMP.cpp`](1-lab/Parallel_OpenMP.cpp) source file alongside its Visual Studio solution and project files. Lab 4 additionally splits its workloads into [`koch.cpp`](4-lab/koch.cpp) and [`hanoi.cpp`](4-lab/hanoi.cpp).
