// The GPU ports: lab 3's matrix multiply and labs 5–6's six float sums, on WebGPU
// (matmul.wgsl, sums.wgsl) or WebGL2 (the fragment shaders below). Times are wall clock on
// the page: upload, kernels and the async read-back, after a warm-up that compiles the shaders.
import { gpu, gpuReady, pipeline, storage, uniform, bindGroup, runAndRead, submitJobs, nextFrame, free, glProgram, glTexture, glTarget, glDraw, glRead } from './gpu.js';

export { gpu, gpuReady, backendName } from './gpu.js';
const now = () => performance.now();

// ---------------------------------------------------------------- lab 3 --
const MATMUL_GL = `#version 300 es
precision highp float; precision highp int;
uniform highp sampler2D A, B;
uniform int N;
out vec4 o;
// one fragment per C[i][u]: x = u, y = i; A[i][t] is texel (t, i), B[t][u] is texel (u, t)
void main() {
  int u = int(gl_FragCoord.x), i = int(gl_FragCoord.y);
  float acc = 0.0; // the int products and sums stay below 2^24, so f32 is exact here
  for (int t = 0; t < N; t++) acc += texelFetch(A, ivec2(t, i), 0).r * texelFetch(B, ivec2(u, t), 0).r;
  o = vec4(acc, 0.0, 0.0, 1.0);
}`;

/** C = A × B for n × n int matrices (flat Int32Array). Returns { C, ms }. */
export async function matmul(A, B, n) {
  if (gpu.backend === 'webgpu') {
    const pipe = await pipeline('matmul.wgsl', 'multiply');
    const t0 = now();
    const a = storage(A), b = storage(B), c = storage(n * n * 4), p = uniform(new Uint32Array([n, 0, 0, 0]));
    const g = Math.ceil(n / 16);
    const C = await runAndRead([{ pipe, group: bindGroup(pipe, [a, b, c, p]), grid: [g, g, 1] }], c, n * n * 4, Int32Array);
    const ms = now() - t0;
    free(a, b, c, p);
    return { C, ms };
  }
  const gl = gpu.gl;
  const prog = await glProgram('matmul', MATMUL_GL);
  const t0 = now();
  const ta = glTexture(n, n, 'R32F', Float32Array.from(A)), tb = glTexture(n, n, 'R32F', Float32Array.from(B));
  const tc = glTexture(n, n, 'RGBA32F'), fb = glTarget([tc]);
  glDraw(prog, fb, n, n, { A: ta, B: tb }, { N: n });
  const [px] = await glRead(fb, n, n);
  const C = new Int32Array(n * n);
  for (let k = 0; k < C.length; k++) C[k] = px[k * 4];
  const ms = now() - t0;
  gl.deleteFramebuffer(fb); [ta, tb, tc].forEach((t) => gl.deleteTexture(t));
  return { C, ms };
}

// ------------------------------------------------------------ labs 5–6 --
const RANGE = 10000, WG = 256, BATCH = 500;
const SUMS_GL = `#version 300 es
precision highp float; precision highp int;
uniform int R;
layout(location = 0) out vec4 o0; // sum1, sum2, sum3_1, sum3_2
layout(location = 1) out vec4 o1; // sum4_1, sum4_2
// level 1: fragment g reduces elements g * 256 … g * 256 + 255
void main() {
  int g = int(gl_FragCoord.x);
  vec4 a = vec4(0.0); vec2 b = vec2(0.0);
  for (int k = 0; k < ${WG}; k++) {
    int i = g * ${WG} + k;
    if (i >= R) break;
    float el = 1.0 / float(i + 1), elSub = 1.0 / float(R - i);
    bool isEven = (i + 1) % 2 == 0;
    a += vec4(el, elSub, isEven ? elSub : 0.0, isEven ? 0.0 : elSub);
    b += vec2(isEven ? el : 0.0, isEven ? 0.0 : el);
  }
  o0 = a; o1 = vec4(b, 0.0, 1.0);
}`;
const FINISH_GL = `#version 300 es
precision highp float; precision highp int;
uniform highp sampler2D P0, P1;
uniform int G;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
// level 2: one fragment reduces the G partials
void main() {
  vec4 a = vec4(0.0), b = vec4(0.0);
  for (int g = 0; g < G; g++) { a += texelFetch(P0, ivec2(g, 0), 0); b += texelFetch(P1, ivec2(g, 0), 0); }
  o0 = a; o1 = vec4(b.xy, 0.0, 1.0);
}`;

/**
 * calcSumsPar() on the GPU, `reps` calls in one submission. Returns { sums: [6 floats],
 * perCall: ms per call (reps calls and one read-back, / reps), oneMs: one call with its read-back }.
 */
export async function sums(reps) {
  if (gpu.backend === 'webgpu') {
    const groups = Math.ceil(RANGE / (WG * 4)); // PER = 4 elements per invocation (sums.wgsl)
    const [p1, p2] = await Promise.all([pipeline('sums.wgsl', 'partial'), pipeline('sums.wgsl', 'finish')]);
    const part = storage(groups * 6 * 4), out = storage(32), par = uniform(new Uint32Array([RANGE, groups, 0, 0]));
    const g1 = bindGroup(p1, [part, null, par]), g2 = bindGroup(p2, [part, out, par]);
    const call = [{ pipe: p1, group: g1, grid: [groups, 1, 1] }, { pipe: p2, group: g2, grid: [1, 1, 1] }];
    // submitted in batches with a yield between them, so encoding many calls never blocks the page
    const go = async (k) => {
      for (; k > BATCH; k -= BATCH) { submitJobs(Array.from({ length: BATCH }, () => call).flat()); await nextFrame(); }
      return runAndRead(Array.from({ length: k }, () => call).flat(), out, 24, Float32Array);
    };
    let t0 = now();
    await go(1);
    const oneMs = now() - t0;
    t0 = now();
    const s = await go(reps);
    const perCall = (now() - t0) / reps;
    free(part, out, par);
    return { sums: Array.from(s), perCall, oneMs };
  }
  const gl = gpu.gl, groups = Math.ceil(RANGE / WG);
  const [prog1, prog2] = await Promise.all([glProgram('sums', SUMS_GL), glProgram('sums-finish', FINISH_GL)]);
  const pt = [glTexture(groups, 1, 'RGBA32F'), glTexture(groups, 1, 'RGBA32F')], ot = [glTexture(1, 1, 'RGBA32F'), glTexture(1, 1, 'RGBA32F')];
  const pf = glTarget(pt), of = glTarget(ot);
  const go = async (k) => {
    for (let r = 0; r < k; r++) {
      glDraw(prog1, pf, groups, 1, {}, { R: RANGE });
      glDraw(prog2, of, 1, 1, { P0: pt[0], P1: pt[1] }, { G: groups });
      if (r % BATCH === BATCH - 1) { gl.flush(); await nextFrame(); }
    }
    const [a, b] = await glRead(of, 1, 1, 2);
    return [a[0], a[1], a[2], a[3], b[0], b[1]];
  };
  let t0 = now();
  await go(1);
  const oneMs = now() - t0;
  t0 = now();
  const s = await go(reps);
  const perCall = (now() - t0) / reps;
  gl.deleteFramebuffer(pf); gl.deleteFramebuffer(of); [...pt, ...ot].forEach((t) => gl.deleteTexture(t));
  return { sums: s, perCall, oneMs };
}

/** Compile the kernels once (on a small input) so the timed runs don't include it. */
export async function warmGpu(which) {
  if (which === 'matmul') { const A = new Int32Array(32 * 32).fill(1); await matmul(A, A, 32); }
  else await sums(2);
}
