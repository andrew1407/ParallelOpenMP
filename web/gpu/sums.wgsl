// WebGPU port of labs 5–6 calcSumsPar(): the six float sums over i < RANGE, as a two-level
// parallel reduction. `partial`: each invocation adds el = 1/(i+1) and elSub = 1/(RANGE−i) for
// PER values of i, then each 256-invocation workgroup tree-reduces the six sums in
// var<workgroup> memory between barriers. `finish`: one workgroup reduces the partials the
// same way. Sum order is sum1, sum2, sum3_1, sum3_2, sum4_1, sum4_2, as in the lab's struct.

const WG: u32 = 256u;
const SUMS: u32 = 6u;
const PER: u32 = 4u; // elements per invocation: 10 workgroups of 256 cover RANGE = 10 000

struct Params { range: u32, groups: u32 }

@group(0) @binding(0) var<storage, read_write> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> sums: array<f32>;
@group(0) @binding(2) var<uniform> P: Params;

var<workgroup> cache: array<f32, 1536>; // SUMS * WG

fn reduce(lid: u32) {
  for (var s = WG / 2u; s > 0u; s /= 2u) {
    workgroupBarrier();
    if (lid < s) {
      for (var k = 0u; k < SUMS; k++) {
        cache[k * WG + lid] += cache[k * WG + lid + s];
      }
    }
  }
  workgroupBarrier();
}

@compute @workgroup_size(WG)
fn partial(
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let lid = lid3.x;
  var s = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  // each invocation adds PER elements of its workgroup's block, strided by WG
  for (var k = 0u; k < PER; k++) {
    let i = (wid.x * PER + k) * WG + lid;
    if (i < P.range) {
      let el = 1.0 / f32(i + 1u);
      let elSub = 1.0 / f32(P.range - i);
      let isEven = (i + 1u) % 2u == 0u;
      s[0] += el;
      s[1] += elSub;
      s[2] += select(0.0, elSub, isEven);
      s[3] += select(elSub, 0.0, isEven);
      s[4] += select(0.0, el, isEven);
      s[5] += select(el, 0.0, isEven);
    }
  }
  for (var k = 0u; k < SUMS; k++) {
    cache[k * WG + lid] = s[k];
  }
  reduce(lid);
  if (lid < SUMS) {
    partials[wid.x * SUMS + lid] = cache[lid * WG];
  }
}

@compute @workgroup_size(WG)
fn finish(@builtin(local_invocation_id) lid3: vec3<u32>) {
  let lid = lid3.x;
  for (var k = 0u; k < SUMS; k++) {
    var acc = 0.0;
    for (var g = lid; g < P.groups; g += WG) {
      acc += partials[g * SUMS + k];
    }
    cache[k * WG + lid] = acc;
  }
  reduce(lid);
  if (lid < SUMS) {
    sums[lid] = cache[lid * WG];
  }
}
