// Runs the exact JS reference off the main thread: CoFDA bits for the GPU parity check, FP64 bits for ULPs.
import {cofdaGemm, fp64Gemm} from "./webgpu/reference.js";

self.onmessage = ({data}) => {
  const started = performance.now();
  try {
    const cofda = cofdaGemm(data);
    const exact = fp64Gemm(data);
    self.postMessage({cofda, exact, ms: performance.now() - started}, [cofda.buffer, exact.buffer]);
  } catch (error) {
    self.postMessage({error: error.message});
  }
};
