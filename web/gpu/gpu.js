// GPU backends for the showcase's optional GPU mode: WebGPU compute (WGSL, gpu/*.wgsl) first,
// WebGL2 fragment-shader GPGPU (float render targets via EXT_color_buffer_float) as the fallback.
// `?gpu=webgl` skips WebGPU, `?gpu=none` turns the GPU mode off, for testing.

export const gpu = { backend: null, device: null, gl: null, info: '', reason: '' };

const force = new URLSearchParams(location.search).get('gpu');

export const gpuReady = (async () => {
  const why = [];
  if (force !== 'webgl' && force !== 'none') {
    if (await initWebGpu()) return gpu.backend = 'webgpu';
    why.push(gpu.reason);
  }
  if (force !== 'none') {
    if (await initWebGl()) return gpu.backend = 'webgl';
    why.push(gpu.reason);
  }
  gpu.reason = force === 'none' ? 'GPU mode is turned off by ?gpu=none' : why.join('; ');
  return null;
})();

export const backendName = () => ({ webgpu: 'WebGPU', webgl: 'WebGL2' }[gpu.backend] || 'none');

async function initWebGpu() {
  if (!navigator.gpu) { gpu.reason = 'no WebGPU (navigator.gpu is missing)'; return false; }
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) { gpu.reason = 'no WebGPU adapter'; return false; }
    const device = await adapter.requestDevice();
    device.lost.then((e) => { if (gpu.device === device) { gpu.device = null; gpu.backend = null; gpu.reason = `the GPU device was lost (${e.message || e.reason})`; } });
    gpu.device = device;
    const info = adapter.info || {};
    gpu.info = [info.vendor, info.architecture, info.description].filter(Boolean).join(' · ') || 'unknown adapter';
    return true;
  } catch (e) {
    gpu.reason = `WebGPU failed to start (${e.message || e})`;
    return false;
  }
}

const probeWebGl = () => new Promise((resolve) => {
  try {
    const w = new Worker(new URL('./glprobe.js', import.meta.url));
    w.onmessage = (e) => { w.terminate(); resolve(e.data); };
    w.onerror = (e) => { e.preventDefault(); w.terminate(); resolve('the WebGL2 check failed'); };
  } catch (e) { resolve(`the WebGL2 check failed (${e.message || e})`); }
});

async function initWebGl() {
  const probe = await probeWebGl();
  if (probe) { gpu.reason = probe; return false; }
  try {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 1;
    const gl = canvas.getContext('webgl2', { antialias: false, depth: false, stencil: false, powerPreference: 'high-performance' });
    if (!gl) { gpu.reason = 'no WebGL2'; return false; }
    if (!gl.getExtension('EXT_color_buffer_float')) { gpu.reason = 'WebGL2 has no float render targets (EXT_color_buffer_float)'; return false; }
    gl.getExtension('KHR_parallel_shader_compile');
    canvas.addEventListener('webglcontextlost', (e) => { e.preventDefault(); gpu.gl = null; gpu.backend = null; gpu.reason = 'the WebGL2 context was lost'; });
    gpu.gl = gl;
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    gpu.info = (dbg && gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL)) || gl.getParameter(gl.RENDERER) || 'unknown renderer';
    return true;
  } catch (e) {
    gpu.reason = `WebGL2 failed to start (${e.message || e})`;
    return false;
  }
}

// ------------------------------------------------------------------- WebGPU --
const modules = new Map(), pipelines = new Map();
async function shaderModule(name) {
  if (!modules.has(name)) {
    modules.set(name, (async () => {
      const res = await fetch(new URL(name, import.meta.url));
      if (!res.ok) throw new Error(`gpu/${name}: HTTP ${res.status}`);
      const module = gpu.device.createShaderModule({ label: name, code: await res.text() });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === 'error');
      if (errors.length) throw new Error(`${name}: ${errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('; ')}`);
      return module;
    })());
  }
  return modules.get(name);
}

/** A compute pipeline for file/entry, cached; compiled off the main thread (createComputePipelineAsync). */
export function pipeline(file, entryPoint) {
  const key = `${file}|${entryPoint}`;
  if (!pipelines.has(key)) {
    const p = (async () => {
      const module = await shaderModule(file);
      gpu.device.pushErrorScope('validation');
      const pipe = await gpu.device.createComputePipelineAsync({ label: key, layout: 'auto', compute: { module, entryPoint } });
      const err = await gpu.device.popErrorScope();
      if (err) throw new Error(err.message);
      return pipe;
    })();
    pipelines.set(key, p);
    p.catch(() => pipelines.delete(key));
  }
  return pipelines.get(key);
}

const U = () => GPUBufferUsage;
export function storage(dataOrBytes) {
  const bytes = typeof dataOrBytes === 'number' ? dataOrBytes : dataOrBytes.byteLength;
  const buf = gpu.device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 4) * 4), usage: U().STORAGE | U().COPY_SRC | U().COPY_DST });
  if (typeof dataOrBytes !== 'number') gpu.device.queue.writeBuffer(buf, 0, dataOrBytes);
  return buf;
}
export function uniform(data) {
  const buf = gpu.device.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 16) * 16), usage: U().UNIFORM | U().COPY_DST });
  gpu.device.queue.writeBuffer(buf, 0, data);
  return buf;
}
/** buffers[k] goes to @binding(k); null skips a binding the entry point doesn't use (layout: 'auto'). */
export function bindGroup(pipe, buffers) {
  return gpu.device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => buffer && { binding, resource: { buffer } }).filter(Boolean),
  });
}
function encode(jobs) {
  const enc = gpu.device.createCommandEncoder();
  const pass = enc.beginComputePass();
  for (const { pipe, group, grid } of jobs) {
    pass.setPipeline(pipe);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(...grid);
  }
  pass.end();
  return enc;
}
/** Encode and submit jobs = [{ pipe, group, grid }] in one compute pass, without waiting. */
export function submitJobs(jobs) { gpu.device.queue.submit([encode(jobs).finish()]); }
/** Encode jobs in one compute pass, submit, and read `src` back (async map). */
export async function runAndRead(jobs, src, bytes, Type) {
  const rb = gpu.device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 4) * 4), usage: U().MAP_READ | U().COPY_DST });
  const enc = encode(jobs);
  enc.copyBufferToBuffer(src, 0, rb, 0, Math.ceil(bytes / 4) * 4);
  gpu.device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const out = new Type(rb.getMappedRange().slice(0, bytes));
  rb.unmap(); rb.destroy();
  return out;
}
export const free = (...bufs) => bufs.forEach((b) => b && b.destroy());

// ------------------------------------------------------------------- WebGL2 --
const VERT = '#version 300 es\nvoid main() { vec2 p = vec2(float((gl_VertexID & 1) << 2) - 1.0, float((gl_VertexID & 2) << 1) - 1.0); gl_Position = vec4(p, 0.0, 1.0); }';
const programs = new Map();
export const nextFrame = () => new Promise((r) => setTimeout(r, 0));

/** A linked program for a fragment shader, cached. Waits for KHR_parallel_shader_compile when there is one. */
export async function glProgram(name, frag) {
  if (programs.has(name)) return programs.get(name);
  const gl = gpu.gl;
  const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; };
  const vs = sh(gl.VERTEX_SHADER, VERT), fs = sh(gl.FRAGMENT_SHADER, frag);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog);
  const ext = gl.getExtension('KHR_parallel_shader_compile');
  if (ext) while (!gl.getProgramParameter(prog, ext.COMPLETION_STATUS_KHR)) await nextFrame();
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error(`${name}: ${gl.getShaderInfoLog(fs) || gl.getProgramInfoLog(prog)}`);
  }
  programs.set(name, prog);
  return prog;
}

/** A float texture (RGBA32F or R32F), NEAREST, with optional data. */
export function glTexture(w, h, internal, data = null) {
  const gl = gpu.gl;
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
  for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);
  const fmt = internal === 'R32F' ? gl.RED : gl.RGBA;
  gl.texImage2D(gl.TEXTURE_2D, 0, gl[internal], w, h, 0, fmt, gl.FLOAT, data);
  return tex;
}
/** A framebuffer drawing into the given textures (multiple render targets). */
export function glTarget(textures) {
  const gl = gpu.gl;
  const fb = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  textures.forEach((t, k) => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + k, gl.TEXTURE_2D, t, 0));
  gl.drawBuffers(textures.map((_, k) => gl.COLOR_ATTACHMENT0 + k));
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error('WebGL2: float framebuffer is incomplete');
  return fb;
}
/** Draw a full-screen triangle with `prog` into `fb` (w × h), binding textures to samplers by name. */
export function glDraw(prog, fb, w, h, samplers = {}, ints = {}) {
  const gl = gpu.gl;
  gl.useProgram(prog);
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.viewport(0, 0, w, h);
  Object.entries(samplers).forEach(([name, tex], unit) => {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(gl.getUniformLocation(prog, name), unit);
  });
  for (const [name, v] of Object.entries(ints)) gl.uniform1i(gl.getUniformLocation(prog, name), v);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}
/** Read RGBA32F pixels of attachments from `fb` without blocking: PIXEL_PACK_BUFFER + fence, polled. */
export async function glRead(fb, w, h, attachments = 1) {
  const gl = gpu.gl;
  const n = w * h * 4;
  const bufs = [];
  gl.bindFramebuffer(gl.READ_FRAMEBUFFER, fb);
  for (let k = 0; k < attachments; k++) {
    const pbo = gl.createBuffer();
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbo);
    gl.bufferData(gl.PIXEL_PACK_BUFFER, n * 4, gl.STREAM_READ);
    gl.readBuffer(gl.COLOR_ATTACHMENT0 + k);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, 0);
    bufs.push(pbo);
  }
  gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
  const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
  gl.flush();
  while (gl.clientWaitSync(sync, 0, 0) === gl.TIMEOUT_EXPIRED) await nextFrame();
  gl.deleteSync(sync);
  return bufs.map((pbo) => {
    const out = new Float32Array(n);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbo);
    gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, out);
    gl.deleteBuffer(pbo);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    return out;
  });
}
