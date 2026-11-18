// Lab 4: the lab's KochSnowflake and HanoiTower classes. Their recursive helpers are
// private, so they're opened up here to hand subtrees of the task tree to workers.
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <emscripten.h>
#define private public
#include "../../4-lab/koch.cpp"
#include "../../4-lab/hanoi.cpp"
#undef private

// The lab's constants (4-lab/Parallel_OpenMP.cpp).
static KochSnowflake ks(1500, 500);
static HanoiTower ht;
static std::vector<line> kochLines;

struct HanoiSub { int height; char in, out, mid; };
static std::vector<HanoiSub> hanoiSubs;

// The recursion of HanoiTower::calculate(), stopped `levels` calls deep.
static void hanoiSplit(int h, char in, char out, char mid, int levels) {
  if (!levels || h < 2) { hanoiSubs.push_back({h, in, out, mid}); return; }
  hanoiSplit(h - 1, in, mid, out, levels - 1);
  hanoiSplit(h - 1, mid, out, in, levels - 1);
}

extern "C" {
// Sequential: calcFractalPoints(depth) as the lab calls it.
EMSCRIPTEN_KEEPALIVE double koch_seq(int depth) { return ks.calcFractalPoints(depth); }
// The lab's drawSVG() writes to the in-memory file system; the page reads the file back.
EMSCRIPTEN_KEEPALIVE void koch_svg(int depth) { ks.calcFractalPoints(depth); ks.drawSVG("/koch.svg"); }
EMSCRIPTEN_KEEPALIVE int koch_count() { return ks.points.size(); }
EMSCRIPTEN_KEEPALIVE const double* koch_data() { return &ks.points[0].first; }

// One worker's share: run fractalize() to `levels` to get the 4^levels task lines, then
// fractalize the lines [first, first + count) the rest of the way down.
EMSCRIPTEN_KEEPALIVE double koch_part(int depth, int levels, int first, int count) {
  const double start = omp_get_wtime();
  ks.points.clear();
  if (!depth) { if (!first) ks.calcFractalPoints(0); return omp_get_wtime() - start; }
  ks.fractalize(levels, ks.getInitialPoints());
  kochLines.clear();
  for (size_t i = 0; i + 1 < ks.points.size(); i += 2) kochLines.push_back(make_pair(ks.points[i], ks.points[i + 1]));
  ks.points.clear();
  for (int j = first; j < first + count && j < (int)kochLines.size(); j++) ks.fractalize(depth - levels, kochLines[j]);
  return omp_get_wtime() - start;
}

// Hanoi: the lab's example with move logging on (printed through stdout).
EMSCRIPTEN_KEEPALIVE double hanoi_example(int height) {
  ht.setParallel(true); ht.setShowLogs(true);
  return ht.runCycle(height, 'A', 'B', 'C');
}
EMSCRIPTEN_KEEPALIVE double hanoi_seq(int height) {
  ht.setParallel(false); ht.setShowLogs(false);
  return ht.runCycle(height, 'A', 'B', 'C');
}
// One worker's share: the subproblems [first, first + count) of calculate() split `levels` deep.
EMSCRIPTEN_KEEPALIVE double hanoi_part(int height, int levels, int first, int count) {
  ht.setShowLogs(false);
  const double start = omp_get_wtime();
  hanoiSubs.clear();
  hanoiSplit(height, 'A', 'B', 'C', levels);
  for (int j = first; j < first + count && j < (int)hanoiSubs.size(); j++) {
    const HanoiSub& s = hanoiSubs[j];
    if (s.height) ht.calculate(s.height, s.in, s.out, s.mid);
  }
  return omp_get_wtime() - start;
}
}
