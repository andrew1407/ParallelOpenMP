#!/usr/bin/env bash
# Compile the labs to WebAssembly modules in web/build/ (needs emsdk: source emsdk_env.sh).
set -euo pipefail
cd "$(dirname "$0")"
OUT=../build
mkdir -p "$OUT"
python3 gen.py

COMMON=(-Iinclude -sMODULARIZE -sEXPORT_ES6 -sENVIRONMENT=web,worker -sALLOW_MEMORY_GROWTH
  -sSTACK_SIZE=1MB -sEXPORTED_RUNTIME_METHODS=ccall,FS,HEAPU8,HEAP32,HEAPF32,HEAPF64
  -sEXPORTED_FUNCTIONS=_malloc,_free -Wno-unknown-pragmas -Wno-vla-cxx-extension)

build() { # name, flags..., sources...
  local name=$1; shift
  echo "build/$name.mjs"
  em++ "$@" "${COMMON[@]}" -o "$OUT/$name.mjs"
}

emcc -O3 -Iinclude -c omp_stub.c -o "$OUT/omp_stub.o"
STUB="$OUT/omp_stub.o"

build lab1 -O3 lab1.cpp "$STUB"
build lab2 -O3 lab2.cpp "$STUB"
build lab3 -O3 lab3.cpp "$STUB"
build lab4 -O3 lab4.cpp "$STUB" -sINITIAL_MEMORY=64MB -sMAXIMUM_MEMORY=2GB
build control -O3 control.cpp "$STUB"
build lab7 -O3 lab7.cpp mpi_stub.cpp -sASYNCIFY -sASYNCIFY_STACK_SIZE=65536

# Labs 5 and 6 in several flavours, the way 6-lab/commands.txt compares compiler flags.
LAB5=-DLAB_SOURCE='"../../5-lab/Parallel_OpenMP.cpp"'
LAB6=-DLAB_SOURCE='"../../6-lab/Parallel_OpenMP.cpp"'
build lab5-scalar  -O3 -fopenmp-simd            "$LAB5" simd.cpp "$STUB"
build lab5-simd128 -O3 -fopenmp-simd -msimd128  "$LAB5" simd.cpp "$STUB"
build lab6-O0      -O0 -fopenmp-simd            "$LAB6" simd.cpp "$STUB"
build lab6-O1      -O1 -fopenmp-simd            "$LAB6" simd.cpp "$STUB"
build lab6-O2      -O2 -fopenmp-simd            "$LAB6" simd.cpp "$STUB"
build lab6-O3      -O3 -fopenmp-simd            "$LAB6" simd.cpp "$STUB"
build lab6-O3-lto  -O3 -fopenmp-simd -flto      "$LAB6" simd.cpp "$STUB"
build lab6-O3-simd128 -O3 -fopenmp-simd -msimd128 "$LAB6" simd.cpp "$STUB"
build lab6-O3-simd128-fast -O3 -fopenmp-simd -msimd128 -ffast-math "$LAB6" simd.cpp "$STUB"
rm -f "$OUT/omp_stub.o"
ls -la "$OUT"
