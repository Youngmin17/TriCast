// Browser-GPU execution of TriCast's CoFDA GEMM on FP8 E4M3 operands (WebGPU v1).
//
// createEngine() → {gemm(args), gemmF32(args), destroy()}, args = {a: Uint8Array [M·K], b: Uint8Array [N·K],
// M, N, K, scaleA, scaleB, mma, repeat?}. gemm returns TriCast's fp32 output bits (bit-exact with
// tricast.reference.mma, see cofda.wgsl.js); gemmF32 returns the bits of an f32 fma chain on the
// dequantized operands. ms is the GPU time per dispatch: wall time from submit to completion of `repeat`
// back-to-back dispatches (default 1, which includes the submission latency) divided by `repeat`.
// Operand upload and readback are not timed.
import {COFDA_WGSL, F32_WGSL} from "./cofda.wgsl.js";
import {checkArgs, checkMma, floatToBits} from "./reference.js";

const WORKGROUP = 64;
const usage = () => globalThis.GPUBufferUsage;

function packRows(codes, rows, K) {
  const KW = Math.ceil(K / 4);
  const words = new Uint32Array(rows * KW);
  const bytes = new Uint8Array(words.buffer);
  for (let r = 0; r < rows; r += 1) bytes.set(codes.subarray(r * K, (r + 1) * K), r * KW * 4);
  return words;
}

// k4-major: word (k/4)·N + n holds codes k..k+3 of row n, so neighbouring invocations read neighbouring words.
function packColumns(codes, N, K) {
  const KW = Math.ceil(K / 4);
  const words = new Uint32Array(KW * N);
  const bytes = new Uint8Array(words.buffer);
  for (let n = 0; n < N; n += 1) {
    const row = n * K;
    for (let k = 0; k < K; k += 1) bytes[((k >> 2) * N + n) * 4 + (k & 3)] = codes[row + k];
  }
  return words;
}

async function compiled(device, code, label) {
  const module = device.createShaderModule({code, label});
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  if (errors.length) {
    throw new Error(`${label} 셰이더 컴파일 실패: ${errors.map((e) => `${e.lineNum}:${e.linePos} ${e.message}`).join("; ")}`);
  }
  return module;
}

export async function createEngine() {
  if (typeof navigator === "undefined" || !navigator.gpu) throw new Error("이 브라우저는 WebGPU를 지원하지 않습니다");
  const adapter = await navigator.gpu.requestAdapter({powerPreference: "high-performance"});
  if (!adapter) throw new Error("WebGPU 어댑터를 얻지 못했습니다");
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  let lost = null;
  device.lost.then((info) => {
    lost = info;
  });
  const modules = {
    cofda: await compiled(device, COFDA_WGSL, "cofda"),
    f32: await compiled(device, F32_WGSL, "f32"),
  };
  const pipelines = new Map();

  async function pipeline(kind, constants) {
    const key = `${kind}:${JSON.stringify(constants)}`;
    if (!pipelines.has(key)) {
      pipelines.set(key, device.createComputePipelineAsync({
        layout: "auto", label: key, compute: {module: modules[kind], entryPoint: "main", constants},
      }));
    }
    return pipelines.get(key);
  }

  async function run(kind, constants, args, scaleA, scaleB) {
    if (lost) throw new Error(`GPU 장치를 잃었습니다: ${lost.message || lost.reason}`);
    const {M, N, K} = args;
    const repeat = args.repeat ?? 1;
    if (!Number.isInteger(repeat) || repeat < 1) throw new Error("repeat는 1 이상의 정수여야 합니다");
    const groups = Math.ceil(N / WORKGROUP);
    const maxGroups = device.limits.maxComputeWorkgroupsPerDimension;
    if (groups > maxGroups || M > maxGroups) throw new Error(`행렬이 너무 큽니다: 디스패치 한도 ${maxGroups} 초과 (M=${M}, N=${N})`);
    const a = packRows(args.a, M, K);
    const b = packColumns(args.b, N, K);
    const outBytes = M * N * 4;
    for (const [name, size] of [["A", a.byteLength], ["B", b.byteLength], ["출력", outBytes]]) {
      if (size > device.limits.maxStorageBufferBindingSize || size > device.limits.maxBufferSize) {
        throw new Error(`${name} 버퍼(${size} B)가 이 GPU의 저장 버퍼 한도(${device.limits.maxStorageBufferBindingSize} B)를 넘습니다`);
      }
    }
    const pipe = await pipeline(kind, constants);
    const U = usage();
    const buffer = (size, flags) => device.createBuffer({size, usage: flags});
    const bufA = buffer(a.byteLength, U.STORAGE | U.COPY_DST);
    const bufB = buffer(b.byteLength, U.STORAGE | U.COPY_DST);
    const bufOut = buffer(outBytes, U.STORAGE | U.COPY_SRC);
    const bufRead = buffer(outBytes, U.MAP_READ | U.COPY_DST);
    const bufParams = buffer(32, U.UNIFORM | U.COPY_DST);
    try {
      device.queue.writeBuffer(bufA, 0, a);
      device.queue.writeBuffer(bufB, 0, b);
      device.queue.writeBuffer(bufParams, 0, Uint32Array.of(M, N, K, Math.ceil(K / 4), floatToBits(scaleA),
        floatToBits(scaleB), 0, 0));
      const bind = device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [bufA, bufB, bufOut, bufParams].map((buf, binding) => ({binding, resource: {buffer: buf}})),
      });
      await device.queue.onSubmittedWorkDone();  // uploads are not timed
      device.pushErrorScope("validation");
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bind);
      for (let r = 0; r < repeat; r += 1) pass.dispatchWorkgroups(groups, M);
      pass.end();
      const started = performance.now();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      const ms = (performance.now() - started) / repeat;
      const copy = device.createCommandEncoder();
      copy.copyBufferToBuffer(bufOut, 0, bufRead, 0, outBytes);
      device.queue.submit([copy.finish()]);
      const error = await device.popErrorScope();
      if (error) throw new Error(`WebGPU 검증 오류: ${error.message}`);
      await bufRead.mapAsync(globalThis.GPUMapMode.READ);
      const bits = new Uint32Array(bufRead.getMappedRange().slice(0));
      bufRead.unmap();
      if (lost) throw new Error(`GPU 장치를 잃었습니다: ${lost.message || lost.reason}`);
      return {bits, ms};
    } finally {
      for (const buf of [bufA, bufB, bufOut, bufRead, bufParams]) buf.destroy();
    }
  }

  return {
    async gemm(args) {
      const cfg = checkMma(args.mma);
      const {scaleA, scaleB} = checkArgs(args);
      const wide = cfg.chunk * 225 * 2 ** (cfg.fBits - 6) >= 2 ** 31;  // i32 chunk sums would overflow
      const constants = {F: cfg.fBits, CS: cfg.chunk, F2: cfg.decoupled ? cfg.f2Bits : 23, DECOUPLED: cfg.decoupled ? 1 : 0,
        RNE: cfg.rne ? 1 : 0, WIDE: wide ? 1 : 0, WORDS: cfg.chunk % 4 === 0 ? 1 : 0};
      return run("cofda", constants, args, scaleA, scaleB);
    },
    async gemmF32(args) {
      const {scaleA, scaleB} = checkArgs(args);
      return run("f32", {}, args, scaleA, scaleB);
    },
    destroy() {
      pipelines.clear();
      device.destroy();
    },
  };
}
