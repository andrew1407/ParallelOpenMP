// Parallel OpenMP Labs showcase: lab panels, SVG charts, and the worker "threads".
// The labs' C++ is compiled to WebAssembly (wasm/build.sh -> build/*.mjs). A parallel
// region is a fork over N Web Workers (worker.js), each running the same module on the
// share OpenMP would give that thread; the join is waiting for all N replies.

const REPO = 'https://github.com/Andrew1407/ParallelOpenMP';
const SRC = `${REPO}/blob/main`;
const $ = (s, root = document) => root.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmt = (v, d = 4) => {
  if (typeof v !== 'number' || !Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e6 || a < 1e-3)) return v.toExponential(2);
  return String(+v.toFixed(d));
};
const tickFmt = (v) => { const a = Math.abs(v); return a >= 1e6 ? `${+(v / 1e6).toPrecision(3)}M` : a >= 1e4 ? `${+(v / 1e3).toPrecision(3)}k` : fmt(v, 6); };
const msFmt = (v) => `${fmt(v, v < 10 ? 3 : 1)} ms`;
const range = (n, f = (i) => i) => Array.from({ length: n }, (_, i) => f(i));
const pause = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
const stat = (value, label, cls = '') => `<div class="stat ${cls}"><b>${value}</b><span>${label}</span></div>`;
const color = (i) => `--c${(i % 8) + 1}`;
const CORES = navigator.hardwareConcurrency || 4;

// ----------------------------------------------------------------- workers --
const pool = [];
const pending = new Map();
let callSeq = 0;
let mpiHook = null;

function worker(i) {
  while (pool.length <= i) {
    const idx = pool.length;
    const w = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    w.onmessage = (e) => onWorkerMessage(idx, e.data);
    w.onerror = (e) => console.error(`worker ${idx}:`, e.message);
    pool.push(w);
  }
  return pool[i];
}

function onWorkerMessage(i, d) {
  if (d.type === 'mpi') { // MPI_Send from a rank: hand it to the destination rank
    mpiHook?.({ ...d, bytes: d.data.byteLength });
    worker(d.dest).postMessage({ type: 'mpi-deliver', rank: d.dest, from: d.from, tag: d.tag, data: d.data }, [d.data.buffer]);
    return;
  }
  if (d.type === 'mpi-recv' || d.type === 'mpi-wait') { mpiHook?.(d); return; }
  const p = pending.get(d.id);
  pending.delete(d.id);
  if (d.ok) p.resolve(d); else p.reject(new Error(d.error));
}

function call(i, mod, task, args = {}, transfer = []) {
  const id = ++callSeq;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker(i).postMessage({ id, mod, task, args }, transfer);
  });
}
// fork n threads, join on all replies; `ms` is the page's wall-clock time for the whole region
async function fork(n, mod, task, argsOf = () => ({})) {
  const t0 = performance.now();
  const r = await Promise.all(range(n, (tid) => call(tid, mod, task, argsOf(tid))));
  return { ms: performance.now() - t0, r };
}
const warm = (n, mod) => fork(n, mod, 'load');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
// V8 runs a wasm function's first calls in its baseline compiler and swaps in optimized
// code from a later call on. So before timing, every worker runs the kernels a few times
// on a small input, then waits for the optimizing compiler to finish.
async function warmUp(n, steps) {
  for (let k = 0; k < 2; k++) for (const [mod, task, argsOf] of steps) await fork(n, mod, task, argsOf);
  await sleep(150);
}
const startSoon = () => performance.timeOrigin + performance.now() + 40;
// OpenMP's default static schedule: one contiguous block per thread, the first total % n one longer
const staticRange = (total, n, tid) => {
  const base = Math.floor(total / n), extra = total % n, lo = tid * base + Math.min(tid, extra);
  return [lo, lo + base + (tid < extra ? 1 : 0)];
};

// ------------------------------------------------------------------ charts --
const charts = new Map();
let chartSeq = 0;

function niceStep(span, count) {
  const raw = span / Math.max(1, count);
  const p = 10 ** Math.floor(Math.log10(raw || 1));
  const m = raw / p;
  return (m >= 5 ? 10 : m >= 2 ? 5 : m >= 1 ? 2 : 1) * p;
}
function ticks(min, max, count) {
  if (min === max) { min -= 1; max += 1; }
  const step = niceStep(max - min, count);
  const lo = Math.floor(min / step) * step, hi = Math.ceil(max / step) * step;
  const out = [];
  for (let v = lo; v <= hi + step / 2; v += step) out.push(+v.toPrecision(12));
  return { lo, hi, ticks: out };
}

/** spec: { title, x, y, series: [{ name, xs, ys, color, dash }], height, logY, kind: 'bar' with cats } */
function chart(spec) {
  const id = `c${++chartSeq}`;
  charts.set(id, spec);
  const keys = spec.series.length > 1
    ? `<span class="keys">${spec.series.map((s) => `<span><i style="background:var(${s.color})${s.dash ? ';opacity:.6' : ''}"></i>${esc(s.name)}</span>`).join('')}</span>` : '';
  return `<figure class="chart" data-chart="${id}" data-copyable data-title="${esc(spec.title)}"><figcaption><span>${esc(spec.title)}</span>${keys}${actions('png', 'csv')}</figcaption><div class="plot"></div></figure>`;
}

function drawCharts(root = document) {
  root.querySelectorAll('figure.chart').forEach((fig) => {
    const spec = charts.get(fig.dataset.chart);
    const plot = $('.plot', fig);
    if (!spec || !plot.offsetParent) return;
    (spec.kind === 'bar' ? drawBars : drawChart)(plot, spec);
  });
}
let resizeTimer;
window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { drawCharts(); redrawCustom(); }, 120); });

function yScale(spec, values, H, M) {
  if (spec.logY) {
    const pos = values.filter((v) => v > 0);
    const lo = Math.floor(Math.log10(Math.min(...pos))), hi = Math.max(lo + 1, Math.ceil(Math.log10(Math.max(...pos))));
    const t = range(hi - lo + 1, (i) => 10 ** (lo + i));
    const Y = (v) => H - M.b - ((Math.log10(Math.max(v, 10 ** lo)) - lo) / (hi - lo)) * (H - M.t - M.b);
    return { Y, ticks: t, zero: null };
  }
  const yr = ticks(Math.min(0, ...values), Math.max(...values), 5);
  const Y = (v) => H - M.b - ((v - yr.lo) / (yr.hi - yr.lo || 1)) * (H - M.t - M.b);
  return { Y, ticks: yr.ticks, zero: yr.lo < 0 && yr.hi > 0 ? 0 : null };
}

function drawChart(plot, spec) {
  const W = Math.max(260, plot.clientWidth), H = spec.height || 230;
  const M = { l: 58, r: 14, t: 8, b: 38 };
  const all = spec.series.flatMap((s) => s.xs.map((x, i) => [x, s.ys[i]])).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y) && (!spec.logY || y > 0));
  if (!all.length) { plot.innerHTML = '<p class="hint">No data.</p>'; return; }
  const x0 = Math.min(...all.map((p) => p[0])), x1 = Math.max(...all.map((p) => p[0]));
  const xr = ticks(x0, x1, Math.max(3, Math.floor((W - M.l) / 90)));
  const ys = spec.zeroY === false ? all.map((p) => p[1]) : [...all.map((p) => p[1])];
  const { Y, ticks: yt, zero } = yScale(spec, ys, H, M);
  const X = (v) => M.l + ((v - x0) / (x1 - x0 || 1)) * (W - M.l - M.r);
  const xt = (spec.xTicks || xr.ticks).filter((v) => v >= x0 - 1e-9 && v <= x1 + 1e-9);
  const grid = yt.map((v) => `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${M.l - 8}" y="${Y(v)}" text-anchor="end" dominant-baseline="middle">${tickFmt(v)}</text>`).join('')
    + xt.map((v) => `<line x1="${X(v)}" x2="${X(v)}" y1="${H - M.b}" y2="${H - M.b + 4}" stroke="var(--muted)"/><text x="${X(v)}" y="${H - M.b + 16}" text-anchor="middle">${fmt(v, 6)}</text>`).join('');
  const axes = `<line x1="${M.l}" x2="${W - M.r}" y1="${H - M.b}" y2="${H - M.b}" stroke="var(--line)"/>`
    + (zero !== null ? `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--muted)" stroke-opacity=".45"/>` : '')
    + `<text x="${(M.l + W - M.r) / 2}" y="${H - 4}" text-anchor="middle">${esc(spec.x)}</text>`
    + `<text transform="translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">${esc(spec.y)}</text>`;
  const ok = (s, i) => Number.isFinite(s.ys[i]) && (!spec.logY || s.ys[i] > 0);
  const lines = spec.series.map((s) => {
    const pts = s.xs.map((x, i) => (ok(s, i) ? `${X(x).toFixed(1)},${Y(s.ys[i]).toFixed(1)}` : '')).filter(Boolean).join(' ');
    const dots = spec.dots || s.xs.length <= 30 ? s.xs.map((x, i) => (ok(s, i) ? `<circle cx="${X(x)}" cy="${Y(s.ys[i])}" r="2.6" fill="var(${s.color})"/>` : '')).join('') : '';
    return `<polyline points="${pts}" fill="none" stroke="var(${s.color})" stroke-width="${s.width || 1.8}" stroke-linejoin="round" ${s.dash ? 'stroke-dasharray="5 4"' : ''}/>${dots}`;
  }).join('');
  plot.innerHTML = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="${esc(spec.title)}">${grid}${axes}${lines}<line class="cross" y1="${M.t}" y2="${H - M.b}" stroke="var(--muted)" stroke-dasharray="3 3" visibility="hidden"/><g class="marks"></g></svg><div class="tip" hidden></div>`;

  const svg = $('svg', plot), tip = $('.tip', plot), cross = $('.cross', svg), marks = $('.marks', svg);
  const nearest = (xs, v) => {
    let lo = 0, hi = xs.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] < v) lo = m; else hi = m; }
    return Math.abs(xs[lo] - v) <= Math.abs(xs[hi] - v) ? lo : hi;
  };
  const move = (e) => {
    const rect = svg.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    if (px < M.l - 6 || px > W - M.r + 6) return hide();
    const v = x0 + ((px - M.l) / (W - M.l - M.r)) * (x1 - x0);
    const hits = spec.series.map((s) => { const i = nearest(s.xs, v); return { s, i, x: s.xs[i], y: s.ys[i] }; }).filter((h) => ok(h.s, h.i));
    if (!hits.length) return hide();
    const hx = X(hits[0].x);
    cross.setAttribute('x1', hx); cross.setAttribute('x2', hx); cross.setAttribute('visibility', 'visible');
    marks.innerHTML = hits.map((h) => `<circle cx="${X(h.x)}" cy="${Y(h.y)}" r="4" fill="var(--panel)" stroke="var(${h.s.color})" stroke-width="2"/>`).join('');
    tip.innerHTML = `<div class="tx">${esc(spec.x)} = ${fmt(hits[0].x, 6)}</div>` + hits.map((h) => `<div><i style="background:var(${h.s.color})"></i>${esc(h.s.name || spec.y)}: ${fmt(h.y, 6)}</div>`).join('');
    placeTip(tip, rect, (hx / W) * rect.width);
  };
  const hide = () => { tip.hidden = true; cross.setAttribute('visibility', 'hidden'); marks.innerHTML = ''; };
  svg.addEventListener('pointermove', move);
  svg.addEventListener('pointerdown', move);
  svg.addEventListener('pointerleave', hide);
}

function placeTip(tip, rect, left) {
  tip.hidden = false;
  const tw = tip.offsetWidth;
  tip.style.left = `${left + 12 + tw > rect.width ? Math.max(0, left - tw - 12) : left + 12}px`;
  tip.style.top = '8px';
}

// Grouped bars: spec.cats are the groups, each series has one value per group.
function drawBars(plot, spec) {
  const W = Math.max(260, plot.clientWidth), H = spec.height || 230;
  const M = { l: 58, r: 14, t: 8, b: spec.rotate ? 96 : 38 };
  const vals = spec.series.flatMap((s) => s.ys).filter(Number.isFinite);
  const { Y, ticks: yt } = yScale(spec, vals.length ? vals : [1], H, M);
  const n = spec.cats.length, bw = (W - M.l - M.r) / n, inner = Math.min(bw * 0.78, 34 * spec.series.length), each = inner / spec.series.length;
  const base = spec.logY ? H - M.b : Y(0);
  const grid = yt.map((v) => `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${M.l - 8}" y="${Y(v)}" text-anchor="end" dominant-baseline="middle">${tickFmt(v)}</text>`).join('');
  const bars = spec.cats.map((c, ci) => {
    const x = M.l + ci * bw + (bw - inner) / 2;
    const label = spec.rotate
      ? `<text transform="translate(${M.l + ci * bw + bw / 2} ${H - M.b + 12}) rotate(-35)" text-anchor="end">${esc(c)}</text>`
      : `<text x="${M.l + ci * bw + bw / 2}" y="${H - M.b + 16}" text-anchor="middle">${esc(c)}</text>`;
    return spec.series.map((s, si) => {
      const v = s.ys[ci];
      if (!Number.isFinite(v) || (spec.logY && v <= 0)) return '';
      const y = Y(v);
      return `<rect x="${x + si * each + 1}" y="${Math.min(y, base)}" width="${Math.max(1, each - 2)}" height="${Math.max(1, Math.abs(base - y))}" rx="2" fill="var(${s.color})"/>`;
    }).join('') + label;
  }).join('');
  const axes = `<line x1="${M.l}" x2="${W - M.r}" y1="${base}" y2="${base}" stroke="var(--line)"/>`
    + (spec.rotate ? '' : `<text x="${(M.l + W - M.r) / 2}" y="${H - 4}" text-anchor="middle">${esc(spec.x)}</text>`)
    + `<text transform="translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">${esc(spec.y)}</text>`;
  plot.innerHTML = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="${esc(spec.title)}">${grid}<rect class="band" y="${M.t}" height="${H - M.t - M.b}" width="${bw}" fill="var(--soft)" visibility="hidden"/>${bars}${axes}</svg><div class="tip" hidden></div>`;
  const svg = $('svg', plot), tip = $('.tip', plot), band = $('.band', svg);
  svg.insertBefore(band, svg.querySelector('rect:not(.band)'));
  const move = (e) => {
    const rect = svg.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const ci = Math.floor((px - M.l) / bw);
    if (ci < 0 || ci >= n) return hide();
    band.setAttribute('x', M.l + ci * bw); band.setAttribute('visibility', 'visible');
    tip.innerHTML = `<div class="tx">${esc(spec.cats[ci])}</div>` + spec.series.map((s) => `<div><i style="background:var(${s.color})"></i>${esc(s.name || spec.y)}: ${fmt(s.ys[ci], 6)}</div>`).join('');
    placeTip(tip, rect, ((M.l + ci * bw + bw / 2) / W) * rect.width);
  };
  const hide = () => { tip.hidden = true; band.setAttribute('visibility', 'hidden'); };
  svg.addEventListener('pointermove', move);
  svg.addEventListener('pointerdown', move);
  svg.addEventListener('pointerleave', hide);
}

// Shared pieces for the timing labs.
const speedupCharts = (rows, seqMs, what = 'workers') => {
  const ns = rows.map((r) => r.n);
  return `<div class="charts two">
    ${chart({ title: `Wall-clock time vs ${what}`, x: what, y: 'time, ms', xTicks: ns, series: [{ name: 'workers', xs: ns, ys: rows.map((r) => r.ms), color: '--c1' }, { name: 'sequential', xs: ns, ys: ns.map(() => seqMs), color: '--c2', dash: true }] })}
    ${chart({ title: 'Speedup over the sequential run', x: what, y: 'speedup ×', xTicks: ns, series: [{ name: 'measured', xs: ns, ys: rows.map((r) => seqMs / r.ms), color: '--c3' }, { name: 'ideal', xs: ns, ys: ns, color: '--muted', dash: true }] })}
  </div>`;
};
// `worst` is the slowest worker's own timer (omp_get_wtime inside the wasm), when there is one
const timingTable = (rows, seqMs, extra = []) => {
  const worst = rows.some((r) => Number.isFinite(r.worst));
  return `<div data-copyable><div class="copy-head" style="margin-top:16px"><h2>Timings <small>· sequential ${msFmt(seqMs)}</small></h2>${actions('table')}</div>
  <div class="table-wrap"><table><thead><tr><th class="num">Workers</th><th class="num">Wall clock</th>${worst ? '<th class="num">Slowest worker, own timer</th>' : ''}<th class="num">Speedup</th><th class="num">Efficiency</th>${extra.map((e) => `<th>${e.head}</th>`).join('')}</tr></thead><tbody>
  ${rows.map((r) => `<tr><td class="num">${r.n}</td><td class="num">${msFmt(r.ms)}</td>${worst ? `<td class="num">${msFmt(r.worst)}</td>` : ''}<td class="num">${fmt(seqMs / r.ms, 2)}×</td><td class="num">${fmt((seqMs / r.ms / r.n) * 100, 1)}%</td>${extra.map((e) => `<td>${e.cell(r)}</td>`).join('')}</tr>`).join('')}
  </tbody></table></div></div>`;
};
const consoleBlock = (title, lines, note = '') => `<div data-copyable data-title="${esc(title)}"><div class="copy-head"><h2>${esc(title)}${note ? ` <small>· ${note}</small>` : ''}</h2>${actions('text')}</div><pre class="console">${esc(lines.join('\n'))}</pre></div>`;
const check = (ok, yes = 'matches', no = 'differs') => `<span class="${ok ? 'ok' : 'bad'}">${ok ? `✓ ${yes}` : `✗ ${no}`}</span>`;
const sameArrays = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

// -------------------------------------------------------------- lab panels --
const workersField = { key: 'threads', label: 'Workers (threads)', value: '1, 2, 4, 8', type: 'text', list: true, wide: true };
const parseList = (s) => String(s).split(/[\s,;]+/).filter(Boolean).map(Number);

const LABS = [
  {
    id: 'lab1', tab: 'Lab 1', name: 'Hello threads', src: `${SRC}/1-lab/Parallel_OpenMP.cpp`,
    intro: 'Sets the thread count to <code>std::thread::hardware_concurrency()</code> and, inside <code>#pragma omp parallel</code>, every thread prints the greeting in a <code>critical</code> section. Here each thread is a worker running the lab\'s <code>main()</code>. All of them are released at the same instant, and the page prints the lines in the order they finish, so the order changes from run to run.',
    fields: [
      { key: 'n', label: 'Threads', value: Math.min(16, CORES), min: 1, max: 16 },
      { key: 'runs', label: 'Runs', value: 6, min: 1, max: 30 },
    ],
    hint: `The default is <code>navigator.hardwareConcurrency</code> (${CORES} here), the browser's version of <code>hardware_concurrency()</code>.`,
    run: runLab1, render: renderLab1,
  },
  {
    id: 'lab2', tab: 'Lab 2', name: 'Uneven split', src: `${SRC}/2-lab/Parallel_OpenMP.cpp`,
    intro: 'The <code>master</code> thread splits the array by halving from the end: the last thread gets half of it, the next a quarter, and thread 0 gets 1/2<sup>threads−1</sup>. After a <code>barrier</code>, each thread fills its chunk with <code>a[i] = i</code>, then computes the 3-point average <code>b[i]</code>. It times both phases per thread. Worker <i>k</i> runs the lab\'s <code>main()</code> as thread <i>k</i> and reports the block <code>main()</code> prints for it.',
    fields: [
      { key: 'threads', label: 'Threads', value: 8, min: 1, max: 16 },
      { key: 'len', label: 'Array length aLen', value: 2000000, min: 16, max: 8000000 },
    ],
    hint: 'The lab uses <code>aLen = 16000</code>, but browsers round timers in workers to about 0.1 ms on pages without cross-origin isolation, and 16 000 elements are filled in microseconds. The default here is 2 000 000. The build moves <code>a</code> and <code>b</code> to the heap so the length can grow.',
    run: runLab2, render: renderLab2,
  },
  {
    id: 'lab3', tab: 'Lab 3', name: 'Matrix multiply', src: `${SRC}/3-lab/Parallel_OpenMP.cpp`,
    intro: 'Multiplies two random <code>int</code> matrices (values 0–99) with a triple loop, sequentially and with <code>#pragma omp parallel for collapse(3)</code>, where every <code>C[i][u] += A[i][t] * B[t][u]</code> is an <code>atomic</code>, for 1, 2, 4 and 8 threads.',
    fields: [
      { key: 'size', label: 'SIZE', value: 500, min: 10, max: 900 },
      workersField,
    ],
    note: 'Ported: the worker share. The sequential run is the lab\'s <code>multiply()</code>. In the parallel runs, worker <i>k</i> runs the same loop body over the block of the SIZE³ (i, u, t) space that the static schedule gives thread <i>k</i>. Workers don\'t share memory, so instead of the <code>atomic</code>, each one fills its own C and the page adds them up after the join.',
    run: runLab3, render: renderLab3,
  },
  {
    id: 'lab4', tab: 'Lab 4', name: 'Koch & Hanoi', src: `${SRC}/4-lab/Parallel_OpenMP.cpp`,
    intro: 'Two recursive workloads parallelised with <code>#pragma omp task … if(…)</code>. <code>KochSnowflake</code> splits a line into 4 segments per level and writes the points to an SVG (1500 × 500, drawn at depth 7). <code>HanoiTower</code> moves <i>h</i> disks from rod A to rod B in 2<sup>h</sup> − 1 moves. Both use the lab\'s classes. The tasks are spawned by one thread (<code>single</code>) inside <code>#pragma omp parallel</code>, and every Koch leaf writes its two points into its own slots of the vector, so the order is the same as in the sequential run. For the parallel runs, the page splits the task tree a few levels down and hands the subtrees to the workers.',
    fields: [
      { key: 'draw', label: 'Koch depth to draw', value: 7, min: 0, max: 9 },
      { key: 'kochMax', label: 'Koch sweep: max depth', value: 10, min: 1, max: 10 },
      { key: 'example', label: 'Hanoi example height', value: 4, min: 1, max: 10 },
      { key: 'hanoiMax', label: 'Hanoi sweep: max height', value: 24, min: 6, max: 27 },
      workersField,
    ],
    hint: 'The lab sweeps Koch depths 0–13 and Hanoi heights 5–27. Depth 13 needs about 2 GB of points, which won\'t fit in WebAssembly memory, so the defaults are lower.',
    run: runLab4, render: renderLab4,
  },
  {
    id: 'lab5', tab: 'Lab 5', name: 'SIMD reduction', src: `${SRC}/5-lab/Parallel_OpenMP.cpp`,
    intro: 'Sums 1/(i+1) and 1/(RANGE−i) for i &lt; 10 000 in <code>float</code>, both in total and split by even/odd index, once in a plain loop and once under <code>#pragma omp simd reduction(+: …)</code>. <code>omp simd</code> asks for vector instructions within one thread, so there are no workers here. Instead, the lab is built twice: with <code>-fopenmp-simd</code> (clang honours the pragma, no runtime needed), and again with <code>-msimd128</code> added so WebAssembly SIMD can be used.',
    fields: [{ key: 'reps', label: 'Timing: calls averaged', value: 500, min: 1, max: 20000 }],
    hint: 'One call takes tens of microseconds, below the timer resolution in a worker, so each time is the average over many calls. The lab\'s own timings, printed by its <code>main()</code>, show what a single call measures here.',
    run: runLab5, render: renderLab5,
  },
  {
    id: 'lab6', tab: 'Lab 6', name: 'SIMD benchmark', src: `${SRC}/6-lab/Parallel_OpenMP.cpp`,
    intro: 'The lab 5 kernels in a benchmark loop: <code>main()</code> calls <code>calcSumsSeq</code> and then <code>calcSumsPar</code> RANGE × 10 times each and reports both totals. <code>6-lab/commands.txt</code> compares <code>icc</code> flags (-O0, -O1, -ipo, profile-guided, -parallel). Here the same source is built with emscripten flags instead: -O0 to -O3, -flto (the counterpart of -ipo), and -msimd128 with and without -ffast-math.',
    fields: [{ key: 'reps', label: 'Calls per kernel', value: 1000, min: 1, max: 100000 }],
    hint: 'The lab makes 100 000 calls per kernel, about 1 billion loop iterations. The default here is 1 000 per build, to keep all seven builds within a couple of seconds.',
    run: runLab6, render: renderLab6,
  },
  {
    id: 'lab7', tab: 'Lab 7', name: 'MPI ranks', src: `${SRC}/7-lab-mpi/Parallel_OpenMP.cpp`,
    intro: 'Two MPI ranks each generate a 10 000-element array. Rank 1 sends the second half of <code>a</code> to rank 0 (tag 1), and rank 0 sends the first half of <code>b</code> to rank 1 (tag 2). Each rank then prints its half of <code>a[i] + b[i]</code> and its <code>MPI_Wtime</code>. Two workers run the lab\'s unchanged <code>main()</code> as ranks 0 and 1. <code>MPI_Send</code> posts the buffer to the page, which forwards it to the other rank, and <code>MPI_Recv</code> blocks the rank (through Asyncify) until it arrives.',
    fields: [],
    run: runLab7, render: renderLab7,
  },
  {
    id: 'control', tab: 'Control', name: 'Layered pass', src: `${SRC}/control-task/Parallel_OpenMP.cpp`,
    intro: 'A feed-forward pass through layers of sizes 10 → 10 → 30 → … → 1. For each layer, the inputs are summed (<code>parallel for reduction(+)</code>), and every output is tanh(w · sum + 1) with a random weight w ∈ [0, 1) (<code>parallel for</code>). The sequential run is the lab\'s pass. In the parallel runs, each worker keeps its static-schedule chunk of every layer. Per layer, it sends its partial sum to the page and gets the reduced sum back, which is one fork and one join per layer.',
    fields: [
      { key: 'layers', label: 'Layer sizes', value: '10, 10, 30, 30, 40, 60, 60, 40, 30, 30, 1', type: 'text', layers: true, wide: true, xwide: true },
      workersField,
    ],
    examples: [['10, 10, 30, 30, 40, 60, 60, 40, 30, 30, 1', 'lab layers'], ['10, 1000000, 1000000, 1000000, 1', 'wide'], ['10, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 1', 'deep']],
    note: 'Ported: the weights. The lab draws each layer\'s weights from one <code>std::mt19937</code> before its <code>parallel for</code>. Workers don\'t share memory, so here each worker draws the weights for its own chunk from its own generator.',
    run: runControl, render: renderControl,
  },
];

// ------------------------------------------------------------------- lab 1 --
async function runLab1(p) {
  await warm(p.n, 'lab1');
  const runs = [];
  for (let r = 0; r < p.runs; r++) {
    const order = [];
    const at = startSoon();
    const t0 = performance.now();
    // each reply is one thread leaving the critical section; the page prints it right away
    await Promise.all(range(p.n, (tid) => call(tid, 'lab1', 'lab1', { tid, n: p.n, at }).then((d) => order.push({ tid, line: d.res.lines.join('\n'), t: performance.now() - t0 }))));
    runs.push(order);
  }
  return runs;
}
function renderLab1(runs, p) {
  const key = (o) => o.map((x) => x.tid).join(',');
  const distinct = new Set(runs.map(key)).size;
  const last = runs[runs.length - 1];
  const rows = runs.map((o, r) => `<tr><td class="num">${r + 1}</td>${o.map((x) => `<td class="num"><span class="sw" style="background:var(${color(x.tid)})"></span>${x.tid}</td>`).join('')}</tr>`).join('');
  return `<div class="stats">${stat(p.n, 'threads')}${stat(p.runs, 'runs')}${stat(distinct, 'different orders', distinct > 1 ? 'good' : '')}</div>
    ${consoleBlock('Program output (last run)', last.map((x) => x.line), 'one line per thread, in finishing order')}
    <div data-copyable style="margin-top:16px"><div class="copy-head"><h2>Finishing order per run <small>· which thread finished 1st, 2nd, …</small></h2>${actions('table')}</div>
    <div class="table-wrap"><table><thead><tr><th class="num">Run</th>${range(p.n, (i) => `<th class="num">#${i + 1}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table></div></div>
    ${distinct === 1 && p.runs > 1 ? '<p class="hint">Every run finished in the same order this time. Run it again, or with more threads.</p>' : ''}`;
}

// ------------------------------------------------------------------- lab 2 --
async function runLab2(p) {
  await warmUp(p.threads, [['lab2', 'lab2', (tid) => ({ tid, n: p.threads, len: 200000, at: 0 })]]);
  const at = startSoon();
  const t0 = performance.timeOrigin + performance.now();
  const { r } = await fork(p.threads, 'lab2', 'lab2', (tid) => ({ tid, n: p.threads, len: p.len, at }));
  const coef = SEC_COEF;
  const threads = r.map((d, tid) => {
    // the k-th block that main() printed is thread k's
    const text = d.res.lines.join('\n');
    const blocks = [...text.matchAll(/Total execution time for thread with (\S+) \((\S+)-(\S+)\) elements:\s+filling time: (\S+) ns\.\s+calc\. time: (\S+) ns\.\s+total: (\S+) ns\./g)];
    const b = blocks[tid];
    if (!b) throw new Error(`Couldn't read thread ${tid}'s block from the lab's output.`);
    const [, count, lo, hi, fill, calc, total] = b;
    return { tid, count: +count, lo: +lo, hi: +hi, printed: [fill, calc, total], fill: (+fill / coef) * 1e3, calc: (+calc / coef) * 1e3, block: b[0], start: d.res.t0 - t0, end: d.res.t1 - t0 };
  });
  return { threads, coef };
}
function renderLab2(d, p) {
  const T = d.threads;
  const totalFill = T.reduce((s, t) => s + t.fill, 0), totalCalc = T.reduce((s, t) => s + t.calc, 0);
  const slow = T.reduce((a, b) => (a.fill + a.calc >= b.fill + b.calc ? a : b));
  const cats = T.map((t) => `T${t.tid}`);
  const lastLines = [
    `Filling time (total): ${fmtC(totalFill / 1e3 * d.coef)} ns.`,
    `Calc. time (total): ${fmtC(totalCalc / 1e3 * d.coef)} ns.`,
    `Total (both): ${fmtC((totalFill + totalCalc) / 1e3 * d.coef)} ns.`,
  ];
  const rows = T.map((t) => `<tr><td><span class="sw" style="background:var(${color(t.tid)})"></span>${t.tid}</td><td class="num">${t.lo}–${t.hi}</td><td class="num">${t.count.toLocaleString('en')}</td><td class="num">${fmt((t.count / p.len) * 100, 3)}%</td><td class="num">${msFmt(t.fill)}</td><td class="num">${msFmt(t.calc)}</td><td class="num">${msFmt(t.fill + t.calc)}</td><td class="num">${t.printed.join(' / ')}</td></tr>`).join('');
  return `<div class="stats">${stat(p.len.toLocaleString('en'), 'elements')}${stat(p.threads, 'threads')}${stat(`T${slow.tid}`, 'slowest thread')}${stat(fmt((slow.count / p.len) * 100, 2) + '%', 'its share of the array')}${stat(msFmt(totalFill), 'fill time, all threads')}${stat(msFmt(totalCalc), 'calc time, all threads')}</div>
    <div class="charts two">
      ${chart({ kind: 'bar', title: 'Time per thread', x: 'thread', y: 'time, ms', cats, series: [{ name: 'fill a[i] = i', ys: T.map((t) => t.fill), color: '--c1' }, { name: 'calc b[i]', ys: T.map((t) => t.calc), color: '--c2' }] })}
      ${chart({ kind: 'bar', title: 'Elements per thread', x: 'thread', y: 'elements', cats, series: [{ name: 'elements', ys: T.map((t) => t.count), color: '--c3' }] })}
    </div>
    <div data-copyable style="margin-top:16px"><div class="copy-head"><h2>Per thread</h2>${actions('table')}</div>
    <div class="table-wrap"><table><thead><tr><th>Thread</th><th class="num">Range</th><th class="num">Elements</th><th class="num">Share</th><th class="num">Fill</th><th class="num">Calc</th><th class="num">Total</th><th class="num">Printed "ns" (fill / calc / total)</th></tr></thead><tbody>${rows}</tbody></table></div></div>
    <div style="margin-top:16px">${consoleBlock('Program output', [...T.map((t) => t.block).join('\n\n').split('\n'), '', ...lastLines], 'each thread\'s block from its worker; totals added up as main() does')}</div>`;
}
const SEC_COEF = 1e9; // the lab's secCoef, seconds to ns
const fmtC = (v) => String(+v.toPrecision(6)); // cout's default 6 significant digits

// ------------------------------------------------------------------- lab 3 --
async function runLab3(p) {
  const ns = p.threads, maxN = Math.max(...ns), size = p.size;
  const seed = (Date.now() / 1000) | 0; // the lab seeds with time(NULL)
  await warmUp(maxN, [['lab3', 'lab3setup', () => ({ size: 80, seed })], ['lab3', 'lab3seq', () => ({ size: 80 })], ['lab3', 'lab3chunk', (tid) => ({ tid, n: maxN, size: 80 })]]);
  await fork(maxN, 'lab3', 'lab3setup', () => ({ size, seed }));
  const seq = await fork(1, 'lab3', 'lab3seq', () => ({ size }));
  const ref = seq.r[0].res.C;
  const rows = [];
  for (const n of ns) {
    const { ms, r } = await fork(n, 'lab3', 'lab3chunk', (tid) => ({ tid, n, size }));
    const C = new Int32Array(size * size); // the join: add up each thread's contributions
    for (const d of r) { const part = d.res.C; for (let i = 0; i < C.length; i++) C[i] += part[i]; }
    rows.push({ n, ms, worst: Math.max(...r.map((d) => d.res.sec * 1e3)), ok: sameArrays(C, ref) });
    await pause();
  }
  return { rows, seqMs: seq.ms, seqLab: seq.r[0].res.sec * 1e3, size, sample: Array.from(ref.subarray(0, Math.min(6, size))) };
}
function renderLab3(d) {
  const best = d.rows.reduce((a, b) => (a.ms <= b.ms ? a : b));
  const allOk = d.rows.every((r) => r.ok);
  return `<div class="stats">${stat(`${d.size} × ${d.size}`, 'matrices')}${stat(msFmt(d.seqMs), 'sequential (wall clock)')}${stat(msFmt(d.seqLab), 'multiply() own timer')}${stat(`${fmt(d.seqMs / best.ms, 2)}× @ ${best.n}`, 'best speedup')}${stat(allOk ? 'C identical' : 'C differs', 'every run vs sequential', allOk ? 'good' : 'bad')}</div>
    ${speedupCharts(d.rows, d.seqMs)}
    ${timingTable(d.rows, d.seqMs, [{ head: 'C vs sequential', cell: (r) => check(r.ok) }])}
    <p class="hint">C[0][0…${d.sample.length - 1}] = ${d.sample.join(', ')}. The wall-clock times include sending each worker's C back to the page and adding them up.</p>`;
}

// ------------------------------------------------------------------- lab 4 --
const levelsFor = (n, branch, depth) => { let L = 0; while (branch ** L < n && L < depth) L++; return L; };

async function kochParallel(n, depth, keep) {
  const levels = levelsFor(n, 4, depth), lines = 4 ** levels;
  const { ms, r } = await fork(n, 'lab4', 'kochPart', (tid) => { const [lo, hi] = staticRange(lines, n, tid); return { depth, levels, first: lo, count: hi - lo }; });
  let pts = null;
  if (keep) { // the join: the threads' points, in task order
    pts = new Float64Array(r.reduce((s, d) => s + d.res.points.length, 0));
    let o = 0;
    for (const d of r) { pts.set(d.res.points, o); o += d.res.points.length; }
  }
  return { ms, worst: Math.max(...r.map((d) => d.res.sec * 1e3)), count: r.reduce((s, d) => s + d.res.count, 0), pts };
}
async function hanoiParallel(n, height) {
  const levels = levelsFor(n, 2, height - 1), subs = 2 ** levels;
  const { ms, r } = await fork(n, 'lab4', 'hanoiPart', (tid) => { const [lo, hi] = staticRange(subs, n, tid); return { height, levels, first: lo, count: hi - lo }; });
  return { ms, worst: Math.max(...r.map((d) => d.res.sec * 1e3)) };
}

async function runLab4(p) {
  const ns = p.threads, maxN = Math.max(...ns);
  await warmUp(maxN, [['lab4', 'kochSeq', () => ({ depth: 6 })], ['lab4', 'kochPart', () => ({ depth: 6, levels: 1, first: 0, count: 4 })], ['lab4', 'hanoiSeq', () => ({ height: 16 })], ['lab4', 'hanoiPart', () => ({ height: 16, levels: 1, first: 0, count: 2 })]]);
  const svg = (await call(0, 'lab4', 'kochSvg', { depth: p.draw })).res;
  const example = (await call(0, 'lab4', 'hanoiExample', { height: p.example })).res;
  // Koch: sequential vs maxN workers over the depths, then every worker count at the deepest
  const koch = [];
  for (let depth = 0; depth <= p.kochMax; depth++) {
    const t0 = performance.now();
    const s = (await call(0, 'lab4', 'kochSeq', { depth, keep: false })).res;
    const seqMs = performance.now() - t0;
    const par = await kochParallel(maxN, depth, false);
    koch.push({ depth, seqMs, lab: s.sec * 1e3, count: s.count, parMs: par.ms });
  }
  await pause();
  const deep = p.kochMax;
  const t0 = performance.now();
  const ref = (await call(0, 'lab4', 'kochSeq', { depth: deep, keep: true })).res;
  const kochSeqMs = performance.now() - t0;
  const kochRows = [];
  for (const n of ns) { const r = await kochParallel(n, deep, true); kochRows.push({ n, ms: r.ms, worst: r.worst, ok: sameArrays(r.pts, ref.points) }); }
  await pause();
  // Hanoi
  const hanoi = [];
  for (let h = 5; h <= p.hanoiMax; h++) {
    const s = await fork(1, 'lab4', 'hanoiSeq', () => ({ height: h }));
    const par = await hanoiParallel(maxN, h);
    hanoi.push({ h, seqMs: s.ms, lab: s.r[0].res.sec * 1e3, parMs: par.ms });
  }
  const hs = await fork(1, 'lab4', 'hanoiSeq', () => ({ height: p.hanoiMax }));
  const hanoiRows = [];
  for (const n of ns) hanoiRows.push({ n, ...(await hanoiParallel(n, p.hanoiMax)) });
  return { svg, example, koch, kochRows, kochSeqMs, hanoi, hanoiRows, hanoiSeqMs: hs.ms, maxN };
}

const svgSources = new Map();
function renderLab4(d, p) {
  const sid = `svg${++chartSeq}`;
  svgSources.set(sid, d.svg.svg);
  const shown = d.svg.svg.replace("<svg xmlns='http://www.w3.org/2000/svg' width='1500' height='500'>", "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1500 500' width='1500' height='500' role='img' aria-label='Koch curve drawn by the lab'>");
  const moves = d.example.lines;
  const kochOk = d.kochRows.every((r) => r.ok);
  const depthTicks = d.koch.map((k) => k.depth);
  return `<h2 class="sub">Koch snowflake</h2>
    <div class="stats">${stat(p.draw, 'depth drawn')}${stat(d.svg.points.toLocaleString('en'), 'points in the SVG')}${stat(msFmt(d.kochSeqMs), `sequential, depth ${p.kochMax}`)}${stat(kochOk ? 'identical' : 'differ', 'worker points vs sequential', kochOk ? 'good' : 'bad')}</div>
    <div class="drawing" data-copyable data-title="Koch snowflake, depth ${p.draw}" data-svg="${sid}"><div class="copy-head"><h2>drawSVG("koch.svg") <small>· the lab's file, 1500 × 500</small></h2>${actions('svg', 'png')}</div><div class="plot koch">${shown}</div></div>
    <p class="hint">The lab fractalizes one line, so this is a Koch curve. The closing <code>z</code> of the path draws the straight baseline under it.</p>
    <div class="charts two" style="margin-top:16px">
      ${chart({ title: `Koch time vs depth (log scale)`, x: 'depth', y: 'time, ms', logY: true, xTicks: depthTicks, series: [{ name: 'sequential', xs: depthTicks, ys: d.koch.map((k) => k.seqMs), color: '--c2' }, { name: `${d.maxN} workers`, xs: depthTicks, ys: d.koch.map((k) => k.parMs), color: '--c1' }, { name: 'calcFractalPoints() own timer', xs: depthTicks, ys: d.koch.map((k) => k.lab), color: '--c4', dash: true }] })}
      ${chart({ title: 'Points vs depth (log scale)', x: 'depth', y: 'points', logY: true, xTicks: depthTicks, series: [{ name: 'points', xs: depthTicks, ys: d.koch.map((k) => k.count), color: '--c3' }] })}
    </div>
    <h3 class="sub3">Workers at depth ${p.kochMax}</h3>
    ${speedupCharts(d.kochRows, d.kochSeqMs)}
    ${timingTable(d.kochRows, d.kochSeqMs, [{ head: 'Points vs sequential', cell: (r) => check(r.ok) }])}
    <p class="hint">Each worker sends its points back to the page (16 bytes per point), and that copy is part of the wall-clock time.</p>

    <h2 class="sub">Towers of Hanoi</h2>
    <div class="stats">${stat(p.example, 'example height')}${stat(moves.length, 'moves printed')}${stat(2 ** p.example - 1, '2^h − 1', moves.length === 2 ** p.example - 1 ? 'good' : 'bad')}${stat(((2 ** p.hanoiMax) - 1).toLocaleString('en'), `moves at height ${p.hanoiMax}`)}${stat(msFmt(d.hanoiSeqMs), `sequential, height ${p.hanoiMax}`)}</div>
    <div class="two-col">
      ${consoleBlock(`Moves for height ${p.example}`, moves, `rods A → B via C, as the lab logs them`)}
      ${chart({ title: 'Hanoi time vs height (log scale)', x: 'height', y: 'time, ms', logY: true, series: [{ name: 'sequential', xs: d.hanoi.map((h) => h.h), ys: d.hanoi.map((h) => h.seqMs), color: '--c2' }, { name: `${d.maxN} workers`, xs: d.hanoi.map((h) => h.h), ys: d.hanoi.map((h) => h.parMs), color: '--c1' }, { name: 'runCycle() own timer', xs: d.hanoi.map((h) => h.h), ys: d.hanoi.map((h) => h.lab), color: '--c4', dash: true }] })}
    </div>
    <h3 class="sub3">Workers at height ${p.hanoiMax}</h3>
    ${speedupCharts(d.hanoiRows, d.hanoiSeqMs)}
    ${timingTable(d.hanoiRows, d.hanoiSeqMs)}`;
}

// ------------------------------------------------------------- labs 5 and 6 --
const SUM_NAMES = ['sum1', 'sum2', 'sum3_1', 'sum3_2', 'sum4_1', 'sum4_2'];
function exactSums() { // the same series in double precision, for reference
  const R = 10000, s = [0, 0, 0, 0, 0, 0];
  for (let i = 0; i < R; i++) {
    const el = 1 / (i + 1), elSub = 1 / (R - i), even = (i + 1) % 2 === 0;
    s[0] += el; s[1] += elSub; s[even ? 2 : 3] += elSub; s[even ? 4 : 5] += el;
  }
  return s;
}

async function runLab5(p) {
  const out = (await call(0, 'lab5-scalar', 'main')).res.lines;
  const builds = {};
  for (const mod of ['lab5-scalar', 'lab5-simd128']) await warmUp(1, [[mod, 'simd', () => ({ reps: 50 })]]);
  for (const mod of ['lab5-scalar', 'lab5-simd128']) builds[mod] = (await call(0, mod, 'simd', { reps: p.reps })).res;
  return { out, builds };
}
function renderLab5(d) {
  const a = d.builds['lab5-scalar'], b = d.builds['lab5-simd128'];
  const ex = exactSums();
  const cols = [['calcSumsSeq', a.seqSums], ['calcSumsPar (omp simd)', a.parSums], ['calcSumsPar + simd128', b.parSums]];
  const withTotals = (s) => [...s, s[2] + s[3], s[4] + s[5]];
  const names = [...SUM_NAMES, 'sum3_1 + sum3_2', 'sum4_1 + sum4_2'];
  const exT = withTotals(ex);
  const rows = names.map((nm, i) => `<tr><td><code>${nm}</code></td>${cols.map(([, s]) => `<td class="num">${withTotals(s)[i].toPrecision(7)}</td>`).join('')}<td class="num muted">${exT[i].toPrecision(10)}</td></tr>`).join('');
  const cats = ['scalar build', '-msimd128 build'];
  const vec = b.par < a.par * 0.7;
  return `<div class="stats">${stat(msFmt(a.seq), 'calcSumsSeq')}${stat(msFmt(a.par), 'calcSumsPar (omp simd)')}${stat(msFmt(b.par), 'calcSumsPar, -msimd128')}${stat(`${fmt(a.seq / b.par, 2)}×`, 'SIMD128 vs sequential', a.seq / b.par > 1 ? 'good' : '')}${stat(vec ? 'yes' : 'no', 'omp simd loop sped up by simd128', vec ? 'good' : '')}</div>
    <div class="charts two">
      ${chart({ kind: 'bar', title: 'Time per call', x: 'build', y: 'time, ms', cats, series: [{ name: 'calcSumsSeq', ys: [a.seq, b.seq], color: '--c2' }, { name: 'calcSumsPar (omp simd)', ys: [a.par, b.par], color: '--c1' }] })}
      ${consoleBlock('Program output', d.out, 'the lab\'s main(), -fopenmp-simd build')}
    </div>
    <div data-copyable style="margin-top:16px"><div class="copy-head"><h2>Sums <small>· float, as printed with 7 digits; last column in double</small></h2>${actions('table')}</div>
    <div class="table-wrap"><table><thead><tr><th>Sum</th>${cols.map(([n]) => `<th class="num">${n}</th>`).join('')}<th class="num">double</th></tr></thead><tbody>${rows}</tbody></table></div></div>
    <p class="hint">The vectorized loop adds in a different order, so its last digits can differ from the sequential loop.</p>`;
}

const LAB6_BUILDS = [
  ['lab6-O0', '-O0'], ['lab6-O1', '-O1'], ['lab6-O2', '-O2'], ['lab6-O3', '-O3'], ['lab6-O3-lto', '-O3 -flto'],
  ['lab6-O3-simd128', '-O3 -msimd128'], ['lab6-O3-simd128-fast', '-O3 -msimd128 -ffast-math'],
];
async function runLab6(p) {
  const rows = [];
  for (const [mod] of LAB6_BUILDS) await call(0, mod, 'simdBench', { reps: 20 });
  await sleep(150);
  for (const [mod, flags] of LAB6_BUILDS) { rows.push({ mod, flags, ...(await call(0, mod, 'simdBench', { reps: p.reps })).res }); await pause(); }
  return rows;
}
function renderLab6(rows, p) {
  const cats = rows.map((r) => r.flags);
  const tr = rows.map((r) => `<tr><td><code>em++ ${r.flags} -fopenmp-simd</code></td><td class="num">${msFmt(r.seq)}</td><td class="num">${msFmt(r.par)}</td><td class="num">${fmt(r.seq / r.par, 2)}×</td></tr>`).join('');
  const o0 = rows[0], o3 = rows[3], simd = rows[5];
  return `<div class="stats">${stat(msFmt(o0.seq), 'Sequential, -O0')}${stat(msFmt(o3.seq), 'Sequential, -O3')}${stat(msFmt(o3.par), 'Parallel, -O3')}${stat(msFmt(simd.par), 'Parallel, -O3 -msimd128')}${stat(`${fmt(o0.seq / simd.par, 1)}×`, '-O0 sequential vs -O3 -msimd128 parallel', 'good')}</div>
    <div class="charts">
      ${chart({ kind: 'bar', rotate: true, height: 300, title: `${p.reps.toLocaleString('en')} calls per kernel`, x: 'build', y: 'time, ms', logY: true, cats, series: [{ name: 'Sequential', ys: rows.map((r) => r.seq), color: '--c2' }, { name: 'Parallel', ys: rows.map((r) => r.par), color: '--c1' }] })}
    </div>
    <div data-copyable style="margin-top:16px"><div class="copy-head"><h2>Builds <small>· the lab prints "Sequential" and "Parallel"</small></h2>${actions('table')}</div>
    <div class="table-wrap"><table><thead><tr><th>Build</th><th class="num">Sequential</th><th class="num">Parallel</th><th class="num">Sequential / Parallel</th></tr></thead><tbody>${tr}</tbody></table></div></div>
    <p class="hint">Every result goes into the lab's <code>volatile</code> checksum, so the optimizer can't drop the calls at any level. The <code>icc</code> options in <code>commands.txt</code> (<code>-prof-gen</code>/<code>-prof-use</code>, <code>-parallel</code>) have no emscripten counterpart, so they're left out.</p>`;
}

// ------------------------------------------------------------------- lab 7 --
async function runLab7() {
  await warm(2, 'lab7');
  const events = [];
  mpiHook = (e) => events.push(e);
  try {
    const t0 = performance.timeOrigin + performance.now();
    const r = await Promise.all([0, 1].map((rank) => call(rank, 'lab7', 'mpi', { rank, size: 2 })));
    const ranks = r.map((d, rank) => {
      const lines = d.res.lines;
      const m = lines.join('\n').match(/Time elapsed \(rank (\d)\): (\S+) ms\./);
      const pairs = [];
      for (const l of lines) { const x = l.match(/^a\[(\d+)\] \+ b\[\d+\] = (-?\d+) \+ (-?\d+) = (-?\d+)$/); if (x) pairs.push([+x[1], +x[2], +x[3], +x[4]]); }
      return { rank, lines, ms: m ? +m[2] : NaN, pairs, start: d.res.t0 - t0, end: d.res.t1 - t0 };
    });
    return { ranks, events: events.map((e) => ({ ...e, at: e.at - t0 })) };
  } finally { mpiHook = null; }
}
function renderLab7(d) {
  const sends = d.events.filter((e) => e.type === 'mpi');
  const both = [...d.ranks[1].pairs, ...d.ranks[0].pairs].sort((a, b) => a[0] - b[0]);
  const mid = both.find((x) => x[0] === 5000);
  return `<div class="stats">${stat(msFmt(d.ranks[0].ms), 'rank 0 MPI_Wtime')}${stat(msFmt(d.ranks[1].ms), 'rank 1 MPI_Wtime')}${stat(sends.length, 'messages')}${stat(`${(sends.reduce((s, e) => s + e.bytes, 0) / 1024).toFixed(1)} KB`, 'sent in total')}${stat(both.length.toLocaleString('en'), 'sums printed')}</div>
    <div class="panel inset" data-copyable data-title="MPI message timeline" id="mpi-box"><div class="copy-head"><h2>Message timeline <small>· ms since the page started both ranks</small></h2>${actions('png')}</div><div class="gantt" id="mpi-timeline"></div>
      <div class="legend"><span><span class="sw" style="background:var(--c1)"></span>rank running main()</span><span><span class="sw" style="background:var(--c2)"></span>blocked in MPI_Recv</span><span><span class="sw" style="background:var(--c3)"></span>MPI_Send → delivery</span></div></div>
    ${chart({ title: 'a[i] + b[i] by index', x: 'i', y: 'sum', series: [{ name: 'rank 1 (i < 5000)', xs: d.ranks[1].pairs.map((x) => x[0]), ys: d.ranks[1].pairs.map((x) => x[3]), color: '--c1' }, { name: 'rank 0 (i ≥ 5000)', xs: d.ranks[0].pairs.map((x) => x[0]), ys: d.ranks[0].pairs.map((x) => x[3]), color: '--c2' }] })}
    <p class="hint">Each rank seeds with <code>time(NULL) + worldRank</code>, so the two arrays differ. In the first half <code>a[i]</code> is halved and <code>b[i]</code> doubled, and the other way round from index 5000 on${mid ? ` (row 5000: ${mid[1]} + ${mid[2]} = ${mid[3]})` : ''}.</p>
    <div class="two-col" style="margin-top:16px">${[1, 0].map((r) => consoleBlock(`Rank ${r} output`, d.ranks[r].lines, `${d.ranks[r].lines.length.toLocaleString('en')} lines`)).join('')}</div>`;
}
function drawMpiTimeline(el, d) {
  const W = Math.max(el.clientWidth - 2, 300), M = { l: 62, r: 16, t: 26, b: 52 }, RH = 46, H = M.t + 2 * RH + M.b;
  const end = Math.max(...d.ranks.map((r) => r.end), ...d.events.map((e) => e.at));
  const start = Math.min(0, ...d.ranks.map((r) => r.start));
  const X = (v) => M.l + ((v - start) / (end - start || 1)) * (W - M.l - M.r);
  const Yc = (rank) => M.t + rank * RH + RH / 2;
  const tk = ticks(start, end, Math.max(2, Math.floor(W / 110)));
  const grid = tk.ticks.filter((v) => v >= start && v <= end).map((v) => `<line x1="${X(v)}" x2="${X(v)}" y1="${M.t}" y2="${H - M.b}" stroke="var(--grid)"/><text x="${X(v)}" y="${H - M.b + 16}" text-anchor="middle">${fmt(v, 2)}</text>`).join('');
  const lanes = d.ranks.map((r) => `<rect x="${X(r.start)}" y="${Yc(r.rank) - 9}" width="${Math.max(2, X(r.end) - X(r.start))}" height="18" rx="4" fill="var(--c1)" fill-opacity=".85"/><text x="${M.l - 10}" y="${Yc(r.rank)}" text-anchor="end" dominant-baseline="middle" style="fill:var(--ink);font-weight:600">rank ${r.rank}</text>`).join('');
  const waits = d.events.filter((e) => e.type === 'mpi-wait').map((w) => {
    const got = d.events.find((e) => e.type === 'mpi-recv' && e.rank === w.rank && e.from === w.from && e.tag === w.tag);
    return got ? `<rect x="${X(w.at)}" y="${Yc(w.rank) - 9}" width="${Math.max(2, X(got.at) - X(w.at))}" height="18" fill="var(--c2)"/>` : '';
  }).join('');
  const arrows = d.events.filter((e) => e.type === 'mpi').map((s) => {
    const got = d.events.find((e) => e.type === 'mpi-recv' && e.rank === s.dest && e.from === s.from && e.tag === s.tag);
    if (!got) return '';
    const x1 = X(s.at), y1 = Yc(s.from) + (s.from < s.dest ? 9 : -9), x2 = Math.max(X(got.at), x1 + 1), y2 = Yc(s.dest) + (s.from < s.dest ? -9 : 9);
    // label above the top lane for messages going up, below the bottom lane for messages going down
    const ly = s.from < s.dest ? Yc(1) + 24 : Yc(0) - 16;
    return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="var(--c3)" stroke-width="2.2" marker-end="url(#mpi-arrow)"/><text x="${x1}" y="${ly}" dominant-baseline="middle" style="fill:var(--ink)">rank ${s.from} → ${s.dest}: tag ${s.tag}, ${s.bytes / 4} ints, ${got.at - s.at < 0.1 ? '< 0.1' : fmt(got.at - s.at, 2)} ms</text>`;
  }).join('');
  el.innerHTML = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="MPI message timeline"><defs><marker id="mpi-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="var(--c3)"/></marker></defs>${grid}${lanes}${waits}${arrows}<text x="${(M.l + W) / 2}" y="${H - 4}" text-anchor="middle">time, ms</text></svg>`;
}

// ----------------------------------------------------------- control task --
async function ctrlParallel(n, layers) {
  const t0 = performance.now();
  const chunk = (len, tid) => { const [lo, hi] = staticRange(len, n, tid); return hi - lo; };
  let r = await Promise.all(range(n, (tid) => call(tid, 'control', 'ctrlInit', { length: chunk(layers[0], tid) })));
  const sums = [];
  for (let i = 1; i < layers.length; i++) {
    const sum = r.reduce((s, d) => s + d.res.partial, 0); // reduction(+: res)
    sums.push(sum);
    const last = i === layers.length - 1;
    r = await Promise.all(range(n, (tid) => call(tid, 'control', 'ctrlLayer', { sum, length: chunk(layers[i], tid), last })));
  }
  return { ms: performance.now() - t0, worst: NaN, sums, output: r.flatMap((d) => d.res.output) };
}
async function runControl(p) {
  const ns = p.threads, layers = p.layers;
  await warmUp(Math.max(...ns), [['control', 'ctrlSeq', () => ({ layers: [10, 20000, 1] })], ['control', 'ctrlInit', () => ({ length: 1000 })], ['control', 'ctrlLayer', () => ({ sum: 1, length: 20000, last: false })]]);
  const lab = (await call(0, 'control', 'main')).res.lines;
  const seq = await fork(1, 'control', 'ctrlSeq', () => ({ layers }));
  const rows = [];
  for (const n of ns) { rows.push({ n, ...(await ctrlParallel(n, layers)) }); await pause(); }
  return { lab, seqMs: seq.ms, seqLab: seq.r[0].res.sec * 1e3, sums: seq.r[0].res.sums, output: seq.r[0].res.output, rows, layers };
}
function renderControl(d) {
  const best = d.rows.reduce((a, b) => (a.ms <= b.ms ? a : b));
  const L = d.layers, idx = range(L.length);
  const outRows = d.output.slice(0, 200).map((v, i) => `<tr><td class="num">${i}</td><td class="num">${v.toPrecision(17)}</td></tr>`).join('');
  const neurons = L.slice(1).reduce((s, v) => s + v, 0);
  return `<div class="stats">${stat(L.length, 'layers')}${stat(neurons.toLocaleString('en'), 'neurons computed')}${stat(d.output.length === 1 ? d.output[0].toPrecision(10) : `${d.output.length} values`, 'output')}${stat(msFmt(d.seqMs), 'sequential (wall clock)')}${stat(msFmt(d.seqLab), 'pass, own timer')}${stat(`${fmt(d.seqMs / best.ms, 2)}× @ ${best.n}`, 'best speedup', d.seqMs / best.ms > 1 ? 'good' : 'bad')}</div>
    <div class="charts two">
      ${chart({ kind: 'bar', title: 'Layer sizes', x: 'layer', y: 'neurons', logY: Math.max(...L) / Math.min(...L) > 1000, cats: idx.map(String), series: [{ name: 'size', ys: L, color: '--c3' }] })}
      ${chart({ title: 'Sum of each layer\'s inputs (sequential run)', x: 'layer', y: 'sum', xTicks: range(L.length - 1, (i) => i + 1), series: [{ name: 'sum', xs: range(L.length - 1, (i) => i + 1), ys: d.sums, color: '--c4' }] })}
    </div>
    <div style="margin-top:16px">${speedupCharts(d.rows, d.seqMs)}</div>
    ${timingTable(d.rows, d.seqMs, [{ head: 'Output', cell: (r) => (r.output.length === 1 ? r.output[0].toPrecision(10) : `${r.output.length} values`) }])}
    <p class="hint">Every layer needs the sum of the whole previous layer, so each layer costs one round trip to the page. With the lab's 10–60-neuron layers, that overhead is far bigger than the work, and more workers only make it slower. Try the "wide" layers.</p>
    <div class="two-col" style="margin-top:16px">
      ${consoleBlock('Program output', d.lab, 'the lab\'s main() with its own layers')}
      <div data-copyable><div class="copy-head"><h2>Output layer <small>· ${d.output.length > 200 ? 'first 200 of ' : ''}${d.output.length}</small></h2>${actions('table')}</div><div class="table-wrap"><table><thead><tr><th class="num">#</th><th class="num">value</th></tr></thead><tbody>${outRows}</tbody></table></div></div>
    </div>`;
}

// ------------------------------------------------------------------ copying --
const ACTION_LABEL = { png: 'Copy image', csv: 'Copy CSV', table: 'Copy table', results: 'Copy results', text: 'Copy text', svg: 'Copy SVG' };
const ACTION_TIP = { png: 'Copy as a PNG image (downloads it if the clipboard is unavailable)', csv: 'Copy the plotted data as CSV', table: 'Copy the table as tab-separated text, pastes into spreadsheets', results: 'Copy the inputs and results as text', text: 'Copy the program output as text', svg: 'Copy the SVG source (downloads it if the clipboard is unavailable)' };
const actions = (...kinds) => `<span class="copy-actions">${kinds.map((k) => `<button type="button" class="mini" data-act="${k}" title="${ACTION_TIP[k]}">${ACTION_LABEL[k]}</button>`).join('')}</span>`;

// Render an on-screen SVG to PNG: computed colours and fonts are inlined so the image
// matches the current theme, with the caption drawn above it.
function svgPng(svg, title, legend = [], scale = 2) {
  const box = svg.viewBox.baseVal;
  const w = (box && box.width) || +svg.getAttribute('width') || svg.clientWidth;
  const h = (box && box.height) || +svg.getAttribute('height') || svg.clientHeight;
  const clone = svg.cloneNode(true);
  const src = [svg, ...svg.querySelectorAll('*')], dst = [clone, ...clone.querySelectorAll('*')];
  src.forEach((el, i) => {
    const cs = getComputedStyle(el), c = dst[i];
    for (const a of ['fill', 'stroke', 'stroke-opacity', 'fill-opacity', 'stroke-width', 'opacity', 'visibility']) c.setAttribute(a, cs.getPropertyValue(a));
    if (el.tagName === 'text') for (const a of ['font-family', 'font-size', 'font-weight']) c.setAttribute(a, cs.getPropertyValue(a));
    c.removeAttribute('style'); c.removeAttribute('class');
  });
  clone.querySelectorAll('title, .marks').forEach((t) => t.remove());
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  clone.setAttribute('width', w); clone.setAttribute('height', h);
  if (!clone.getAttribute('viewBox')) clone.setAttribute('viewBox', `0 0 ${w} ${h}`);
  const css = getComputedStyle(document.documentElement);
  const head = title ? 30 : 0, pad = 14;
  const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(clone)], { type: 'image/svg+xml' }));
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement('canvas');
      c.width = (w + pad * 2) * scale; c.height = (h + head + pad * 2) * scale;
      const ctx = c.getContext('2d');
      ctx.scale(scale, scale);
      ctx.fillStyle = css.getPropertyValue('--panel').trim();
      ctx.fillRect(0, 0, w + pad * 2, h + head + pad * 2);
      const family = getComputedStyle(document.body).fontFamily;
      if (title) {
        ctx.fillStyle = css.getPropertyValue('--ink').trim();
        ctx.font = `600 14px ${family}`;
        ctx.fillText(title, pad, pad + 14);
        ctx.font = `12px ${family}`;
        let x = w + pad;
        for (const { name, color: col } of [...legend].reverse()) {
          x -= ctx.measureText(name).width;
          ctx.fillStyle = css.getPropertyValue('--muted').trim();
          ctx.fillText(name, x, pad + 13);
          x -= 19;
          ctx.fillStyle = css.getPropertyValue(col).trim();
          ctx.fillRect(x, pad + 7, 14, 3);
          x -= 14;
        }
      }
      ctx.drawImage(img, pad, pad + head, w, h);
      URL.revokeObjectURL(url);
      c.toBlob((b) => (b ? resolve(b) : reject(new Error('PNG encoding failed'))), 'image/png');
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('SVG render failed')); };
    img.src = url;
  });
}

const csvCell = (v) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
function chartCsv(spec) {
  if (spec.kind === 'bar') return [[spec.x, ...spec.series.map((s) => s.name || spec.y)], ...spec.cats.map((c, i) => [c, ...spec.series.map((s) => s.ys[i])])].map((r) => r.map(csvCell).join(',')).join('\n');
  const s0 = spec.series[0];
  const shared = spec.series.every((s) => s.xs.length === s0.xs.length && s.xs.every((x, i) => x === s0.xs[i]));
  if (shared) return [[spec.x, ...spec.series.map((s) => s.name || spec.y)], ...s0.xs.map((x, i) => [x, ...spec.series.map((s) => s.ys[i])])].map((r) => r.map(csvCell).join(',')).join('\n');
  return [['series', spec.x, spec.y], ...spec.series.flatMap((s) => s.xs.map((x, i) => [s.name, x, s.ys[i]]))].map((r) => r.map(csvCell).join(',')).join('\n');
}
const tableTsv = (table) => [...table.rows].map((r) => [...r.cells].map((c) => c.innerText.trim().replace(/\s+/g, ' ')).join('\t')).join('\n');

function resultsText(sec) {
  const lab = LABS.find((l) => `panel-${l.id}` === sec.id);
  const p = params(lab);
  const lines = [`${lab.tab}: ${lab.name}`];
  if (lab.fields.length) lines.push(`Parameters: ${lab.fields.map((f) => `${f.label} = ${f.list || f.layers ? p[f.key].join(', ') : p[f.key]}`).join(', ')}`);
  const out = $('.out', sec);
  out.querySelectorAll('h2.sub, .note, .stats, .table-wrap table, pre.console').forEach((el) => {
    if (el.matches('h2.sub')) lines.push('', `## ${el.innerText.trim()}`);
    else if (el.matches('.note')) lines.push(el.innerText.trim());
    else if (el.matches('.stats')) el.querySelectorAll('.stat').forEach((x) => lines.push(`${$('span', x).innerText.trim()}: ${$('b', x).innerText.trim()}`));
    else if (el.matches('table')) { const h = el.closest('[data-copyable]')?.querySelector('h2')?.innerText.trim(); lines.push('', `${h || 'Table'}:`, tableTsv(el)); }
    else { const h = el.closest('[data-copyable]')?.querySelector('h2')?.innerText.trim(); lines.push('', `${h || 'Output'}:`, el.innerText); }
  });
  return lines.join('\n');
}

async function saveOrCopy(blobPromise, type, name, btn, done) {
  try {
    // pass the promise straight to ClipboardItem so Safari keeps the user gesture
    await navigator.clipboard.write([new ClipboardItem({ [type]: blobPromise })]);
    return done('Copied');
  } catch {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(await blobPromise);
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    return done('Downloaded');
  }
}

async function copyAction(btn) {
  const kind = btn.dataset.act, box = btn.closest('[data-copyable]');
  const done = (text) => { btn.style.minWidth = btn.style.minWidth || btn.getBoundingClientRect().width + 'px'; btn.textContent = text; setTimeout(() => (btn.textContent = ACTION_LABEL[kind]), 1400); };
  const name = (box?.dataset.title || 'chart').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  try {
    if (kind === 'png') {
      const svg = box.querySelector('.plot svg, .gantt svg, svg');
      const spec = charts.get(box.dataset.chart);
      const legend = spec && spec.series.length > 1 ? spec.series.map((x) => ({ name: x.name, color: x.color })) : [];
      return await saveOrCopy(svgPng(svg, box.dataset.title, legend), 'image/png', `${name}.png`, btn, done);
    }
    if (kind === 'svg') {
      const text = svgSources.get(box.dataset.svg);
      try { await navigator.clipboard.writeText(text); return done('Copied'); } catch {
        return await saveOrCopy(Promise.resolve(new Blob([text], { type: 'image/svg+xml' })), 'image/svg+xml', 'koch.svg', btn, done);
      }
    }
    const text = kind === 'csv' ? chartCsv(charts.get(box.dataset.chart))
      : kind === 'table' ? tableTsv(box.querySelector('table'))
      : kind === 'text' ? box.querySelector('pre').innerText
      : resultsText(btn.closest('section.panel'));
    await navigator.clipboard.writeText(text);
    done('Copied');
  } catch (e) { console.error(e); done('Failed'); }
}
document.addEventListener('click', (e) => { const b = e.target.closest('button[data-act]'); if (b) copyAction(b); });

// --------------------------------------------------------------- framework --
const nav = $('#nav'), main = $('#labs');
const state = new Map(); // lab id -> { ran, running, data }
let busy = false; // the workers are shared, so one lab runs at a time

function fieldHtml(lab, f) {
  const id = `${lab.id}-${f.key}`;
  const type = f.type || 'number';
  const attrs = type === 'number' ? ` min="${f.min}" max="${f.max}" step="${f.step || 1}" required` : ' spellcheck="false" autocomplete="off"';
  return `<label class="field${f.wide ? ' wide' : ''}${f.xwide ? ' xwide' : ''}" for="${id}">${esc(f.label)}<input id="${id}" name="${f.key}" type="${type}" value="${esc(f.value)}"${attrs}></label>`;
}

function build() {
  for (const lab of LABS) {
    const b = document.createElement('button');
    b.type = 'button'; b.role = 'tab'; b.dataset.lab = lab.id;
    b.setAttribute('aria-controls', `panel-${lab.id}`);
    b.innerHTML = `<span class="lang cpp">C++</span>${lab.tab}<span class="long muted" style="font-weight:400">· ${esc(lab.name)}</span>`;
    b.onclick = () => select(lab.id, true);
    nav.appendChild(b);

    const sec = document.createElement('section');
    sec.className = 'panel'; sec.id = `panel-${lab.id}`; sec.role = 'tabpanel'; sec.hidden = true;
    sec.innerHTML = `<div class="lab-head"><span class="lang cpp">C++ → Wasm</span><h3>${lab.tab}: ${esc(lab.name)}</h3><a href="${lab.src}" target="_blank" rel="noopener">View source</a></div>
      <p class="intro">${lab.intro}</p>
      ${lab.note ? `<p class="note">${lab.note}</p>` : ''}
      <form class="params" novalidate>${lab.fields.map((f) => fieldHtml(lab, f)).join('')}<button class="run" type="submit">Run</button></form>
      ${lab.examples ? `<div class="examples"><span>Try</span>${lab.examples.map(([v, t]) => `<button type="button" class="chip" data-v="${esc(v)}">${esc(t)}</button>`).join('')}</div>` : ''}
      ${lab.hint ? `<p class="hint">${lab.hint}</p>` : ''}
      <div class="out" style="margin-top:18px"></div>`;
    main.appendChild(sec);
    state.set(lab.id, { ran: false, running: false });

    const form = $('form', sec);
    form.addEventListener('submit', (e) => { e.preventDefault(); run(lab); });
    sec.querySelectorAll('.chip').forEach((c) => c.onclick = () => { form.elements[lab.fields[0].key].value = c.dataset.v; run(lab); });
  }
}

function params(lab) {
  const form = $(`#panel-${lab.id} form`);
  const fd = new FormData(form), p = {};
  for (const f of lab.fields) {
    const v = fd.get(f.key);
    p[f.key] = f.list || f.layers ? parseList(v) : f.type === 'text' ? v : Number(v);
  }
  return p;
}

function invalid(lab, p) {
  for (const f of lab.fields) {
    const v = p[f.key];
    if (f.list) {
      if (!v.length || v.some((n) => !Number.isInteger(n) || n < 1 || n > 16)) return `${f.label}: list whole numbers from 1 to 16, e.g. 1, 2, 4, 8.`;
      continue;
    }
    if (f.layers) {
      if (v.length < 2 || v.some((n) => !Number.isInteger(n) || n < 1 || n > 5000000)) return 'Layer sizes: at least two whole numbers from 1 to 5 000 000.';
      if (v.reduce((s, n) => s + n, 0) > 20000000) return 'Layer sizes: keep the total under 20 million neurons.';
      continue;
    }
    if (f.type === 'text') continue;
    if (!Number.isFinite(v) || v < f.min || v > f.max || !Number.isInteger(v)) return `${f.label} must be a whole number from ${f.min} to ${f.max}.`;
  }
  if (lab.id === 'lab4' && p.draw > p.kochMax) return 'The Koch depth to draw can\'t be more than the sweep\'s max depth.';
  return null;
}

async function run(lab) {
  const st = state.get(lab.id);
  if (st.running || busy) return;
  const sec = $(`#panel-${lab.id}`), btn = $('.run', sec), out = $('.out', sec);
  const p = params(lab);
  const bad = invalid(lab, p);
  if (bad) { out.innerHTML = `<div class="err">${esc(bad)}</div>`; return; }
  st.running = true; st.ran = true; busy = true;
  btn.disabled = true; btn.textContent = 'Running…';
  out.style.opacity = '.45';
  if (!out.children.length) out.innerHTML = loadingNote();
  try {
    await pause();
    const data = await lab.run(p);
    st.data = data;
    out.innerHTML = `<div class="out-actions">${actions('results')}</div>` + lab.render(data, p);
    drawCharts(out);
    redrawCustom();
  } catch (e) {
    console.error(e);
    out.innerHTML = `<div class="err">${esc(e.message || e)}</div>`;
  } finally {
    st.running = false; busy = false; btn.disabled = false; btn.textContent = 'Run'; out.style.opacity = '';
    const waiting = LABS.find((l) => !state.get(l.id).ran && !$(`#panel-${l.id}`).hidden);
    if (waiting) run(waiting);
  }
}
const loadingNote = () => `<div class="loading" role="status"><span class="spinner" aria-hidden="true"></span><div><b>Running in ${CORES >= 8 ? 'up to 8' : 'several'} Web Workers…</b><small>The first run also downloads and compiles this lab's WebAssembly module.</small></div><div class="bar" aria-hidden="true"><i></i></div></div>`;

// drawings that aren't regular charts
function redrawCustom() {
  const el = $('#mpi-timeline');
  if (el && el.offsetParent && state.get('lab7').data) drawMpiTimeline(el, state.get('lab7').data);
}

function select(id, push) {
  const lab = LABS.find((l) => l.id === id) || LABS[0];
  nav.querySelectorAll('button').forEach((b) => b.setAttribute('aria-selected', b.dataset.lab === lab.id));
  main.querySelectorAll('section.panel[role="tabpanel"]').forEach((s) => s.hidden = s.id !== `panel-${lab.id}`);
  if (push) try { history.replaceState(null, '', `#${lab.id}`); } catch {}
  document.title = `${lab.tab} · Parallel OpenMP Labs`;
  if (!state.get(lab.id).ran) {
    if (busy) $(`#panel-${lab.id} .out`).innerHTML = loadingNote();
    run(lab);
  } else { drawCharts($(`#panel-${lab.id}`)); redrawCustom(); }
}

build();
select(location.hash.slice(1), false);
window.addEventListener('hashchange', () => select(location.hash.slice(1), false));
