// WebGPU port of lab 3's C[i][u] += A[i][t] * B[t][u] over the whole SIZE³ space. One invocation
// owns one C[i][u]; a 16 × 16 workgroup stages 16 × 16 tiles of A and B in var<workgroup> memory
// between barriers, so each value is read from global memory SIZE / 16 times instead of SIZE.
// int stays int (i32), so the result must match the CPU exactly.

const TILE: u32 = 16u;

struct Params { size: u32 }

@group(0) @binding(0) var<storage, read> A: array<i32>;
@group(0) @binding(1) var<storage, read> B: array<i32>;
@group(0) @binding(2) var<storage, read_write> C: array<i32>;
@group(0) @binding(3) var<uniform> P: Params;

var<workgroup> tileA: array<i32, 256>;
var<workgroup> tileB: array<i32, 256>;

@compute @workgroup_size(TILE, TILE, 1)
fn multiply(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let n = P.size;
  let i = gid.y;   // row of C
  let u = gid.x;   // column of C
  var acc = 0i;
  let tiles = (n + TILE - 1u) / TILE;
  for (var k = 0u; k < tiles; k++) {
    // every invocation loads one element of each tile (zero past the edge)
    let ta = k * TILE + lid.x;
    let tb = k * TILE + lid.y;
    tileA[lid.y * TILE + lid.x] = select(0i, A[i * n + ta], i < n && ta < n);
    tileB[lid.y * TILE + lid.x] = select(0i, B[tb * n + u], tb < n && u < n);
    workgroupBarrier();
    for (var t = 0u; t < TILE; t++) {
      acc += tileA[lid.y * TILE + t] * tileB[t * TILE + lid.x];
    }
    workgroupBarrier();
  }
  if (i < n && u < n) {
    C[i * n + u] = acc;
  }
}
