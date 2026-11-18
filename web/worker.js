// One showcase "thread": a Web Worker that loads the labs' WebAssembly modules (build/*.mjs)
// on demand and runs the task the page sends it. The page forks by posting a task to N of
// these workers and joins by waiting for all N replies.

const modules = new Map();
let out = [];
const inbox = [];      // MPI messages that arrived before the rank asked for them
const waiting = [];    // pending MPI_Recv calls: { from, tag, resolve }
const now = () => performance.timeOrigin + performance.now(); // comparable across workers

function load(name) {
  if (!modules.has(name)) {
    modules.set(name, import(`./build/${name}.mjs`).then((m) => m.default({
      print: (s) => out.push(s),
      printErr: (s) => out.push(s),
    })));
  }
  return modules.get(name);
}

// Release all threads of a fork at the same moment, like the threads of a parallel region.
function startAt(t) { while (now() < t) { /* spin */ } }

const f64 = (M, ptr, n) => M.HEAPF64.slice(ptr >> 3, (ptr >> 3) + n);

const TASKS = {
  load: () => ({}),

  // Any lab's main() with its stdout.
  main(M) {
    out = [];
    const t0 = now();
    M._run_main();
    return { lines: out, ms: now() - t0 };
  },

  // Lab 1: the lab's main() as thread `tid` of `n`.
  lab1(M, { tid, n, at }) {
    M._omp_stub_team(tid, n);
    startAt(at);
    out = [];
    const t0 = now();
    M._run_main();
    return { lines: out, t0, t1: now() };
  },

  // Lab 2: the lab's main() as thread `tid` of `n`, over an array of `len`.
  lab2(M, { tid, n, len, at }) {
    M._lab2_set(n, len);
    M._omp_stub_team(tid, n);
    startAt(at);
    out = [];
    const t0 = now();
    M._run_main();
    return { lines: out, t0, t1: now() };
  },

  lab3setup(M, { size, seed }) { M._lab3_setup(size, seed); return {}; },
  lab3seq(M, { size }) { const sec = M._lab3_seq(); return lab3C(M, size, sec); },
  lab3chunk(M, { tid, n, size }) { const sec = M._lab3_chunk(tid, n); return lab3C(M, size, sec); },

  kochSvg(M, { depth }) {
    M._koch_svg(depth);
    return { svg: M.FS.readFile('/koch.svg', { encoding: 'utf8' }), points: M._koch_count() };
  },
  kochSeq(M, { depth, keep }) {
    const sec = M._koch_seq(depth), count = M._koch_count();
    const points = keep ? f64(M, M._koch_data(), count * 2) : null;
    return { sec, count, points, transfer: points ? [points.buffer] : [] };
  },
  kochPart(M, { depth, levels, first, count }) {
    const sec = M._koch_part(depth, levels, first, count), n = M._koch_count();
    const points = n ? f64(M, M._koch_data(), n * 2) : new Float64Array(0);
    return { sec, count: n, points, transfer: [points.buffer] };
  },

  hanoiExample(M, { height }) { out = []; const sec = M._hanoi_example(height); return { sec, lines: out }; },
  hanoiSeq(M, { height }) { return { sec: M._hanoi_seq(height) }; },
  hanoiPart(M, { height, levels, first, count }) { return { sec: M._hanoi_part(height, levels, first, count) }; },

  // Labs 5 and 6.
  simd(M, { reps }) {
    // one call takes tens of microseconds, below the worker's timer resolution, so time `reps` calls and average
    const best = (f) => { f(); const t0 = performance.now(); for (let i = 0; i < reps; i++) f(); return (performance.now() - t0) / reps; };
    const sums = () => Array.from(M.HEAPF32.subarray(M._simd_sums() >> 2, (M._simd_sums() >> 2) + 6));
    const seq = best(() => M._simd_seq()); const seqSums = sums();
    const par = best(() => M._simd_par()); const parSums = sums();
    return { simd: !!M._simd_enabled(), seq, par, seqSums, parSums };
  },
  simdBench(M, { reps }) {
    return {
      simd: !!M._simd_enabled(),
      seq: M._simd_bench(reps, 0) * 1e3, par: M._simd_bench(reps, 1) * 1e3,
    };
  },

  // Lab 7: the lab's main() as MPI rank `rank` of `size`.
  async mpi(M, { rank, size }) {
    M.mpiRank = rank;
    M.mpiSize = size;
    M.mpiSend = (dest, tag, bytes) => postMessage({ type: 'mpi', from: rank, dest, tag, data: bytes, at: now() }, [bytes.buffer]);
    M.mpiRecv = (from, tag) => new Promise((resolve) => {
      const i = inbox.findIndex((m) => m.from === from && m.tag === tag);
      if (i >= 0) { const [m] = inbox.splice(i, 1); postMessage({ type: 'mpi-recv', rank, from, tag, at: now() }); resolve(m.data); }
      else { waiting.push({ from, tag, resolve }); postMessage({ type: 'mpi-wait', rank, from, tag, at: now() }); }
    });
    out = [];
    const t0 = now();
    await M.ccall('run_main', 'number', [], [], { async: true });
    return { lines: out, t0, t1: now() };
  },

  // Control task.
  ctrlSeq(M, { layers }) {
    const p = M._malloc(layers.length * 4);
    M.HEAP32.set(layers, p >> 2);
    const sec = M._ctrl_seq(p, layers.length);
    M._free(p);
    return { sec, sums: Array.from(f64(M, M._ctrl_sums(), layers.length - 1)), output: Array.from(f64(M, M._ctrl_data(), M._ctrl_length())) };
  },
  ctrlInit(M, { length }) { return { partial: M._ctrl_init(length) }; },
  ctrlLayer(M, { sum, length, last }) {
    const partial = M._ctrl_layer(sum, length);
    return { partial, output: last ? Array.from(f64(M, M._ctrl_data(), M._ctrl_length())) : null };
  },
};

function lab3C(M, size, sec) {
  const p = M._malloc(size * size * 4);
  M._lab3_result(p);
  const C = M.HEAP32.slice(p >> 2, (p >> 2) + size * size);
  M._free(p);
  return { sec, C, transfer: [C.buffer] };
}

self.onmessage = async ({ data }) => {
  if (data.type === 'mpi-deliver') {
    const i = waiting.findIndex((w) => w.from === data.from && w.tag === data.tag);
    if (i >= 0) {
      const [w] = waiting.splice(i, 1);
      postMessage({ type: 'mpi-recv', rank: data.rank, from: data.from, tag: data.tag, at: now() });
      w.resolve(data.data);
    } else inbox.push(data);
    return;
  }
  const { id, mod, task, args } = data;
  try {
    const M = await load(mod);
    const t0 = now();
    const res = await TASKS[task](M, args || {});
    const transfer = res.transfer || [];
    delete res.transfer;
    postMessage({ id, ok: true, res, start: t0, end: now() }, transfer);
  } catch (e) {
    postMessage({ id, ok: false, error: String(e && e.stack || e) });
  }
};
