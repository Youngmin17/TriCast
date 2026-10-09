// What the browser offers for the WebGPU path. Never throws; reasons are user-facing (Korean).
export async function probeWebGPU() {
  const result = {available: false, reason: null, adapter: null, limits: null, features: [], fallback: false};
  try {
    if (typeof navigator === "undefined" || !navigator.gpu) {
      result.reason = globalThis.isSecureContext === false
        ? "WebGPU는 HTTPS 또는 localhost 주소에서만 쓸 수 있습니다"
        : "이 브라우저는 WebGPU를 지원하지 않습니다";
      return result;
    }
    const adapter = await navigator.gpu.requestAdapter({powerPreference: "high-performance"});
    if (!adapter) {
      result.reason = "WebGPU 어댑터를 얻지 못했습니다 (GPU 가속이 꺼져 있거나 이 GPU·드라이버가 차단되었습니다)";
      return result;
    }
    const info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
    result.adapter = {vendor: info.vendor || "", architecture: info.architecture || "", device: info.device || "",
      description: info.description || ""};
    const {limits} = adapter;
    result.limits = {maxBufferSize: limits.maxBufferSize, maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup};
    result.features = [...adapter.features].sort();
    result.fallback = Boolean(info.isFallbackAdapter ?? adapter.isFallbackAdapter);
    const device = await adapter.requestDevice();
    device.destroy();
    result.available = true;
  } catch (error) {
    result.available = false;
    result.reason = `WebGPU 초기화에 실패했습니다: ${error?.message || error}`;
  }
  return result;
}
