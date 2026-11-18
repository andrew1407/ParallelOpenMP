// Control task: the lab's main(), calcSum() and fillArrRand(), plus the layered pass with
// any layer sizes, sequentially or as one thread's static-schedule share of each layer.
#include <vector>
#include <emscripten.h>
#define main lab_main
#include "../../control-task/Parallel_OpenMP.cpp"
#undef main

#include <emscripten.h>
// The lab's main() under a C name the page can call.
extern "C" EMSCRIPTEN_KEEPALIVE int run_main() { return lab_main(); }

static std::random_device rd;
static std::mt19937 generator(rd());
static std::uniform_real_distribution<double> distribution(0.0, 1.0);
static std::vector<double> chunk, next, sums;

// One output neuron, the body of the lab's inner `parallel for`.
static inline double neuron(const double sum) {
  const double weight = distribution(generator);
  const double a = weight * sum + 1.0;
  const double expPlus = exp(a);
  const double expMinus = exp(-a);
  return (expPlus - expMinus) / (expPlus + expMinus);
}

extern "C" {
// The lab's pass over `count` layer sizes. Returns the duration; sums per layer and the
// output go to ctrl_sums()/ctrl_data().
EMSCRIPTEN_KEEPALIVE double ctrl_seq(const int* layers, int count) {
  double* arr = new double[layers[0]];
  fillArrRand(arr, layers[0]);
  double* arrIterable = arr;
  sums.clear();
  const double start = omp_get_wtime();
  for (int i = 1; i < count; i++) {
    const int inputLength = layers[i - 1];
    const int outputLength = layers[i];
    const double sum = calcSum(arrIterable, inputLength);
    sums.push_back(sum);
    double* output = new double[outputLength];
    for (int u = 0; u < outputLength; u++) output[u] = neuron(sum);
    delete[] arrIterable;
    arrIterable = output;
  }
  const double end = omp_get_wtime();
  chunk.assign(arrIterable, arrIterable + layers[count - 1]);
  delete[] arrIterable;
  return end - start;
}

// Worker side: fill this thread's chunk of the input and return its partial sum.
EMSCRIPTEN_KEEPALIVE double ctrl_init(int length) {
  chunk.assign(length, 0.0);
  if (length) fillArrRand(chunk.data(), length);
  return length ? calcSum(chunk.data(), length) : 0.0;
}
// Worker side: this thread's chunk of the next layer from the reduced sum, and its partial sum.
EMSCRIPTEN_KEEPALIVE double ctrl_layer(double sum, int length) {
  next.resize(length);
  for (int u = 0; u < length; u++) next[u] = neuron(sum);
  chunk.swap(next);
  return length ? calcSum(chunk.data(), length) : 0.0;
}
EMSCRIPTEN_KEEPALIVE const double* ctrl_data() { return chunk.data(); }
EMSCRIPTEN_KEEPALIVE int ctrl_length() { return chunk.size(); }
EMSCRIPTEN_KEEPALIVE const double* ctrl_sums() { return sums.data(); }
}
