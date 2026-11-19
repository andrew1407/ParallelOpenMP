# ParallelOpenMP

<p align="center">
  <a href="https://andrew1407.github.io/ParallelOpenMP/" title="Open the labs in the browser">
    <img src="web/icon.svg" alt="Parallel OpenMP Labs" width="140" height="140">
  </a>
</p>

[![pages deploy badge](https://github.com/andrew1407/ParallelOpenMP/actions/workflows/pages.yml/badge.svg)](https://andrew1407.github.io/ParallelOpenMP/)

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

## Showcase

**[Open the labs in the browser](https://andrew1407.github.io/ParallelOpenMP/)**. The labs' own C++ is compiled to WebAssembly with emscripten, with adjustable parameters, timing and speedup charts, the Koch curve SVG, and buttons to copy each chart (PNG or CSV), drawing (SVG or PNG), table, program output and set of results.

- **Threads are Web Workers.** GitHub Pages can't send the COOP/COEP headers that wasm pthreads need, and emscripten has no OpenMP runtime. So the labs compile without `-fopenmp` against a small [`omp.h`](web/wasm/include/omp.h). The page then forks each parallel region over N workers running the same module, gives each one the share OpenMP's static schedule would give that thread, and joins on their replies. Timings are wall clock, sequential against 1, 2, 4 and 8 workers.
- **Native lab code:** labs 1, 2 and 7 and the control task run their unchanged `main()`. Lab 3 uses `multiply()` and `cleanMatrix()`, lab 4 the `KochSnowflake` and `HanoiTower` classes (including `drawSVG`), and labs 5–6 `calcSumsSeq`/`calcSumsPar`, built with `-fopenmp-simd` and with and without `-msimd128`. In lab 7, `MPI_Send`/`MPI_Recv` go through `postMessage` ([`mpi.h`](web/wasm/include/mpi.h), with Asyncify for the blocking receive).
- **GPU mode (labs 3, 5 and 6, optional):** a port of the lab's computation, not the lab's own code, in f32 on the GPU (int for lab 3): [WebGPU](web/gpu/) compute shaders (a tiled matrix multiply in workgroup memory, and a two-level parallel reduction for `calcSumsPar`), with a WebGL2 fragment-shader fallback. The page shows the backend that ran, its time beside the CPU runs and its error against the CPU result. Without either API the option is disabled.
- **Ported:** only the per-thread shares: lab 3's block of the `collapse(3)` loop, the control task's per-layer chunk, and the split of lab 4's task trees. [`web/wasm/gen.py`](web/wasm/gen.py) makes lab 2's array size and lab 3's `SIZE` adjustable at build time, without touching the sources.

Run it locally (needs [emsdk](https://emscripten.org/docs/getting_started/downloads.html)):

```sh
./web/wasm/build.sh              # builds web/build/*.mjs + .wasm
python -m http.server            # then open http://localhost:8000/web/
```

The [pages workflow](.github/workflows/pages.yml) builds and deploys it on every push to `main`.

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
├── control-task/     # Layered feed-forward pass
└── web/              # Browser showcase (GitHub Pages)
    ├── index.html, app.js, worker.js, theme.js, icon.svg
    └── wasm/         # emscripten build: build.sh, gen.py, lab wrappers, omp.h/mpi.h stubs
```

Each lab folder contains a [`Parallel_OpenMP.cpp`](1-lab/Parallel_OpenMP.cpp) source file alongside its Visual Studio solution and project files. Lab 4 additionally splits its workloads into [`koch.cpp`](4-lab/koch.cpp) and [`hanoi.cpp`](4-lab/hanoi.cpp).
