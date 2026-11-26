"""Bit-width-accurate ONNX export and ONNX Runtime verification for quantized PyTorch / HF models.

    python -m tricast.onnx_export build --model Qwen/Qwen3-0.6B --revision REV \
        --recipe recipe.yaml --out model.onnx
    python -m tricast.onnx_export from-pack --package pkg_dir --out model.onnx \
        [--kernel standard|matmul_nbits]
    python -m tricast.onnx_export run --onnx model.onnx [--tokenizer pkg_dir] [--prompt TEXT ...]

Exports a traceable model patched with ``EmuLinear`` or ``EmuConv2d`` (including full-attention Hugging Face
causal LMs), or an existing ``tricast.pack`` package, to ONNX (opset 21, IR 10). By default every selected
layer must be stored at its actual bit-width; dense fallbacks require explicit opt-in.

- ``kernel="standard"`` (default, 100% standard ONNX opset 21):
  - Native ONNX ``DequantizeLinear`` initializers for 4-bit (``INT4`` / ``UINT4``, 2 weights/byte) and 8-bit
    (``INT8`` / ``UINT8`` / ``FLOAT8E4M3FN`` / ``FLOAT8E5M2`` / ``FLOAT8E4M3FNUZ`` / ``FLOAT8E5M2FNUZ``)
    tensor, per-row/channel, and exact group-quantized layers.
  - Sub-byte bit-packed ``UINT8`` streams (``ceil(N * bits / 8)`` bytes — exact 1, 2, 3, 4,
    5, 6, 7, 8 bits/weight) unpacked with standard ONNX ``BitShift`` / ``BitwiseAnd`` + ``Gather`` codebooks
    for 2-bit, 3-bit, 6-bit, ``NF4``, ``FP4`` (``MXFP4`` / ``NVFP4``), ``FP6``, fixed-point ``BFP`` /
    ``MXINT``, ``RaZeR``, 2-D block scales, two-level scales, and ragged groups.
  - Before writing the file, every layer's ONNX decode subgraph is executed in ``onnxruntime`` (optimizations
    disabled) and checked bit for bit (``same_bits``, signed zero aside) against the FP32 operand TriCast
    multiplies with.
- ``kernel="matmul_nbits"``:
  - Emits ``com.microsoft::MatMulNBits`` nodes for eligible 2-bit, 4-bit, and 8-bit
    group-quantized ``Linear`` layers (runtime behavior depends on the execution provider), while
    encoding any other bit-widths (e.g. 3-bit, NF4, MXFP4) in the same mixed-precision graph with the standard
    bit-packed path and verifying them bit for bit.
  - MatMulNBits computes ``(code - zero) * scale`` in fp32 without the recipe's dequant rounding (BF16 for
    ``mma_input: dequant``), so its weight can differ from TriCast's by that rounding. Each such layer
    records ``bit_exact_weight`` and ``weight_max_abs_diff`` in the manifest; ``standard`` is the exact mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .bitpack import (
    encode_scale_tensor,
    float_codebook,
    int_codebook,
    narrow_float_tensor,
    pow2_codebook,
    same_bits,
    unpack_bits,
)
from .formats import FP4_E2M1, FloatFormat, IntFormat, Pow2Format, container_dtype, get_format
from .nn.conv import EmuConv2d
from .nn.linear import EmuLinear
from .nn.patch import decoder_layers, iter_emuconv2d, iter_emulinear
from .pack import (
    CONFIG,
    MANIFEST,
    WEIGHTS,
    Encoded,
    effective_weight,
    encode_dense,
    encode_nf4,
    encode_qtensor,
    encode_razer,
)
from .quant.qtensor import QTensor
from .transforms import LinearTransform

SCHEMA = "tricast.onnx/1"
KernelMode = Literal["standard", "matmul_nbits"]
_OUT_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
_OUT_FORMATS = (None, "bf16", "fp16", "fp32")
_EXTERNAL_THRESHOLD_DEFAULT = 1_800_000_000


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_layer_supported(name: str, spec: Any) -> None:
    if spec.sparsity.kind != "none" or spec.outliers is not None:
        raise ValueError(f"{name}: sparsity and outliers are not exported to ONNX")
    if spec.activation is not None:
        raise ValueError(f"{name}: activation quantization is not exported to ONNX (weights only)")
    out_format = getattr(getattr(spec, "mma", None), "out_format", None)
    out_name = getattr(out_format, "name", None)
    if out_name not in _OUT_FORMATS:
        raise ValueError(f"{name}: MMA output format {out_name} is not exported (bf16, fp16 or fp32 only)")


# ---------- tracing wrappers ----------
class _TraceLinear(nn.Module):
    """Export-time stand-in for ``EmuLinear`` / ``PackedLinear``: traces input transforms into standard ONNX
    ops and multiplies by ``self.weight_t`` (``[K, N]``) without a ``Transpose`` so ``do_constant_folding``
    preserves the exact initializer name ``<layer>.weight_t``."""

    def __init__(
        self,
        weight_f32: torch.Tensor,
        bias: torch.Tensor | None,
        transform: LinearTransform,
        out_dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        out_features, in_features = weight_f32.shape
        self.in_features, self.out_features = in_features, out_features
        self.transform_kind = transform.kind
        self.block = transform.block
        self.out_dtype = out_dtype
        self.weight_t = nn.Parameter(weight_f32.detach().float().T.contiguous(), requires_grad=False)
        self.bias = (
            None if bias is None else nn.Parameter(bias.detach().float().contiguous(), requires_grad=False)
        )
        if transform.diag is not None:
            self.register_buffer("diag", transform.diag.detach().float().cpu().contiguous())
        else:
            self.diag = None
        if transform.kind == "random_hadamard" and (
            self.diag is None or not torch.all((self.diag == -1) | (self.diag == 1))
        ):
            raise ValueError("random Hadamard requires a sign diagonal")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Preserve the order of the FP32 butterfly adds/subtracts and the final normalization.
        # Applying a pre-normalized dense matrix has different rounding, even for identical weights.
        xin = LinearTransform(self.transform_kind, self.block, self.diag).apply_activation(x).float()
        out = torch.matmul(xin, self.weight_t)
        if self.bias is not None:
            out = out + self.bias
        if self.out_dtype is not None:
            out = out.to(self.out_dtype)
        return out.to(x.dtype)


class _TraceConv2d(nn.Module):
    """Export-time stand-in for ``EmuConv2d``: keeps ``self.weight`` in ``[out_c, in_c/g, kh, kw]`` so
    ``do_constant_folding`` preserves ``<layer>.weight``."""

    def __init__(self, conv: EmuConv2d, weight_f32: torch.Tensor, out_dtype: torch.dtype | None) -> None:
        super().__init__()
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.padding_mode = conv.padding_mode
        self._padding = conv._padding
        self.out_dtype = out_dtype
        self.weight = nn.Parameter(weight_f32.detach().float().contiguous(), requires_grad=False)
        self.bias = (
            None
            if conv.bias is None
            else nn.Parameter(conv.bias.detach().float().contiguous(), requires_grad=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xin = x.float()
        if self.padding_mode != "zeros" and any(self._padding):
            xin = F.pad(xin, self._padding, mode=self.padding_mode)
            out = F.conv2d(xin, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)
        else:
            out = F.conv2d(xin, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.out_dtype is not None:
            out = out.to(self.out_dtype)
        return out.to(x.dtype)


class _CausalLMExportWrapper(nn.Module):
    """Traces a Hugging Face causal LM with dynamic batch/sequence axes and pure-tensor 4-D causal masking
    (avoiding functorch ``vmap`` inside ``transformers.masking_utils``)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        cfg = model.config
        if (
            getattr(cfg, "sliding_window", None)
            or getattr(cfg, "attention_chunk_size", None)
            or any(layer_type != "full_attention" for layer_type in (getattr(cfg, "layer_types", None) or []))
        ):
            raise ValueError("the ONNX causal LM wrapper supports full causal attention only")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        idx = torch.arange(seq_len, device=input_ids.device)
        causal = idx.unsqueeze(0) <= idx.unsqueeze(1)
        valid = causal.unsqueeze(0).unsqueeze(0) & attention_mask.to(torch.bool).unsqueeze(1).unsqueeze(2)
        neg = torch.full((), torch.finfo(torch.float32).min, dtype=torch.float32, device=input_ids.device)
        zero = torch.zeros((), dtype=torch.float32, device=input_ids.device)
        mask4d = torch.where(valid, zero, neg)
        pos = idx.unsqueeze(0)
        return self.model(
            input_ids=input_ids, attention_mask=mask4d, position_ids=pos, use_cache=False
        ).logits


def _is_hf_causal_lm(model: nn.Module) -> bool:
    config = getattr(model, "config", None)
    archs = getattr(config, "architectures", None) or []
    cls_name = type(model).__name__
    return cls_name.endswith("ForCausalLM") or any(str(a).endswith("ForCausalLM") for a in archs)


# ---------- layer capture ----------
@dataclass
class _CapturedLayer:
    name: str
    op_kind: Literal["linear", "conv2d"]
    spec: Any
    encoded: list[Encoded]  # length 1 for Linear or Conv2d(groups=1); length = groups for grouped Conv2d
    target_weight: torch.Tensor  # [out, in] for Linear, [out_c, in_c/g, kh, kw] for Conv2d (CPU fp32)
    fallback: str | None
    transform: LinearTransform
    bias: torch.Tensor | None
    conv_ref: EmuConv2d | None = None


def _encode_single_emulinear(layer: EmuLinear) -> tuple[Encoded, torch.Tensor, str | None]:
    """Recover the stored operand codes without modifying the calibrated layer."""
    if not layer._calibrated or layer._weight_operand is None:
        raise RuntimeError(f"{layer.name}: calibration is required; call tricast.calibrate(...)")
    _check_layer_supported(layer.name, layer.spec)
    target = effective_weight(layer).detach().cpu()

    encoded: Encoded | None = None
    fallback: str | None = None
    spec = layer.spec

    if spec.weight_algo.kind == "nf4":
        w_in = (
            layer.weight.detach()
            if spec.transform.kind == "none"
            else layer.transform.apply_weight(layer.weight.detach())
        )
        encoded = encode_nf4(w_in)
        from .pack import decode as pack_decode

        entry = {"kind": encoded.kind, "shape": list(encoded.shape), **encoded.meta}
        if not same_bits(pack_decode(entry, encoded.tensors, spec), target):
            encoded = None
            fallback = "NF4 codes did not decode to the replaced weight"
    elif spec.weight_algo.kind == "razer":
        w_in = (
            layer.weight.detach()
            if spec.transform.kind == "none"
            else layer.transform.apply_weight(layer.weight.detach())
        )
        try:
            encoded = encode_razer(w_in)
            from .pack import decode as pack_decode

            entry = {"kind": encoded.kind, "shape": list(encoded.shape), **encoded.meta}
            if not same_bits(pack_decode(entry, encoded.tensors, spec), target):
                encoded = None
                fallback = "RaZeR codes did not decode to the replaced weight"
        except ValueError:
            encoded = None
            fallback = "RaZeR weight numel not divisible by block size"
    elif spec.weight is None:
        fallback = f"{spec.weight_algo.kind} keeps no integer codes TriCast can read back"
    else:
        encoded, fallback = _recover_calibrated_qtensor(layer, target)

    if encoded is not None:
        from .pack import decode as pack_decode

        entry = {"kind": encoded.kind, "shape": list(encoded.shape), **encoded.meta}
        if not same_bits(pack_decode(entry, encoded.tensors, spec), target):
            encoded = None
            fallback = "encoded codes did not decode bit-for-bit to TriCast's operand"
    if encoded is None:
        encoded = encode_dense(target)
    return encoded, target, fallback


def _recover_calibrated_qtensor(layer: EmuLinear, target: torch.Tensor) -> tuple[Encoded | None, str | None]:
    """Recover the exact ``Encoded`` representation of an already-calibrated ``EmuLinear``."""
    spec = layer.spec
    qspec = spec.weight
    assert qspec is not None
    op = layer._weight_operand
    assert op is not None
    # Case 1: mma_input == "scaled" -> the operand stores grid values, scale and global_scale.
    if qspec.mma_input == "scaled":
        scale = op.scale
        if scale is not None and qspec.granularity == "block":
            # The GEMM operand repeats each block scale across its rows.
            scale = scale[:: qspec.block[0]]
        qt = QTensor(
            values=op.values.detach().to(container_dtype(qspec.format)).cpu(),
            scale=None if scale is None else scale.detach().cpu(),
            zero_point=None,
            global_scale=None if op.alpha is None else op.alpha.detach().cpu(),
            spec=qspec,
            shape=(layer.out_features, layer.in_features),
        )
        enc = encode_qtensor(qt)
        return (enc, None if enc is not None else "unsupported scaled QTensor layout")

    # Case 2: RTN / awq_clip with a fitted transform or clip (no Hessian) -> quantize directly.
    if spec.weight_algo.kind != "gptq":
        from .quant.api import quantize
        from .weight_quant.clip import apply_clip

        w = (
            layer.weight.detach()
            if spec.transform.kind == "none"
            else layer.transform.apply_weight(layer.weight.detach())
        )
        if layer._clip is not None:
            w = apply_clip(w, layer._clip)
        qt = quantize(w, qspec, backend=layer.backend, noise=layer._weight_noise)
        enc = encode_qtensor(qt)
        return (enc, None if enc is not None else "unsupported QTensor layout")

    # Case 3: Calibrated GPTQ (or SR) where mma_input == "dequant":
    # In GPTQ (src/tricast/weight_quant/gptq.py), static group/row/tensor/block scales and zero points are
    # computed by ``compute_scale(w_transformed, qspec)`` on the transformed weight. Only the integer/float
    # grid codes were chosen column-by-column by Hessian compensation, and ``target`` holds the exact BF16
    # (or FP16/FP32) dequantized weight round_to_format((values - zp) * scale * gscale, dequant_fmt).
    # For every <=8-bit format we can invert ``target`` onto the 2**bits candidate dequantized values per
    # domain and verify ``same_bits(decoded, target)``.
    from .reference.quantize import compute_scale, expand_scale, view_2d
    from .weight_quant.clip import apply_clip

    w = (
        layer.weight.detach()
        if spec.transform.kind == "none"
        else layer.transform.apply_weight(layer.weight.detach())
    )
    if layer._clip is not None:
        w = apply_clip(w, layer._clip)
    x2d = view_2d(w)
    rows, k = x2d.shape
    scale, zero_point, global_scale = compute_scale(x2d, qspec)

    if isinstance(qspec.format, IntFormat) and 1 <= qspec.format.bits <= 8:
        grid = int_codebook(qspec.format).to(x2d.device)  # [2**bits]
    elif isinstance(qspec.format, FloatFormat) and 1 <= qspec.format.bits <= 8:
        grid = float_codebook(qspec.format).to(x2d.device)
    else:
        return None, "element format wider than 8 bits"

    # Candidate dequantized value for every (row, col, code_idx): [rows, k, 2**bits] in chunks of rows so
    # memory stays tiny even on large layers.
    scale_pe = None if scale is None else expand_scale(scale, qspec, rows, k).float()
    zp_pe = None if zero_point is None else expand_scale(zero_point, qspec, rows, k).float()
    g_val = None if global_scale is None else global_scale.float()
    from .reference.cast import round_to_format
    from .reference.quantize import _add, _mul, _sub
    from .rounding import Rounding

    recovered = torch.empty((rows, k), dtype=torch.float32, device="cpu")
    target_dev = target.to(x2d.device)
    chunk_rows = max(1, (1 << 20) // max(1, k * int(grid.numel())))
    for r0 in range(0, rows, chunk_rows):
        r1 = min(rows, r0 + chunk_rows)
        cand = grid.view(1, 1, -1).expand(r1 - r0, k, -1)
        if zp_pe is not None and qspec.zero_point == "int":
            cand = _sub(cand, zp_pe[r0:r1, :, None])
        if scale_pe is not None:
            cand = _mul(cand, scale_pe[r0:r1, :, None])
        if zp_pe is not None and qspec.zero_point == "float":
            cand = _add(cand, zp_pe[r0:r1, :, None])
        if g_val is not None:
            cand = _mul(cand, g_val)
        deq = round_to_format(
            cand.reshape(-1, grid.numel()), qspec.dequant_format, Rounding.RNE, saturate=False
        ).reshape(r1 - r0, k, grid.numel())
        # Match bit pattern (signed zero normalized to +0.0)
        deq_bits = (deq + 0.0).view(torch.int32)
        tgt_bits = (target_dev[r0:r1, :, None] + 0.0).view(torch.int32)
        eq = deq_bits == tgt_bits
        if not bool(eq.any(dim=-1).all()):
            # act_order=False with group < block refreshes group scales from compensated weights during GPTQ;
            # those layers require live Capture during calibrate() (via build_onnx / tricast.pack).
            return None, "GPTQ dynamic group scale requires live capture during calibration (use build_onnx)"
        idx = eq.to(torch.int32).argmax(dim=-1)
        recovered[r0:r1] = grid[idx].cpu()

    qt = QTensor(
        values=recovered.to(container_dtype(qspec.format)),
        scale=None if scale is None else scale.cpu(),
        zero_point=None if zero_point is None else zero_point.cpu(),
        global_scale=None if global_scale is None else global_scale.cpu(),
        spec=qspec,
        shape=(rows, k),
    )
    enc = encode_qtensor(qt)
    return (enc, None if enc is not None else "unsupported QTensor layout")


def _capture_patched_model(
    model: nn.Module, live_encoded: dict[str, Encoded] | None = None
) -> list[_CapturedLayer]:
    """Collect every ``EmuLinear`` and ``EmuConv2d`` in ``model`` and verify unique module ownership."""
    linears = dict(iter_emulinear(model))
    convs = dict(iter_emuconv2d(model))
    names = {id(m): n for n, m in linears.items()} | {id(m): n for n, m in convs.items()}
    shared = [
        name
        for name, mod in model.named_modules(remove_duplicate=False)
        if isinstance(mod, (EmuLinear, EmuConv2d)) and names.get(id(mod)) != name
    ]
    if shared:
        raise ValueError(f"a quantized module shared under two names is not exported: {shared[0]}")

    captured: list[_CapturedLayer] = []
    for name, layer in linears.items():
        if not layer._calibrated or not layer._weight_current():
            raise ValueError(f"{name}: weight cache is stale; refresh or calibrate before exporting")
        _check_layer_supported(name, layer.spec)
        target = effective_weight(layer).detach().cpu()
        if live_encoded is not None and name in live_encoded:
            enc = live_encoded[name]
            fallback = None if enc.kind != "dense" else "stored as dense weight"
        else:
            enc, target, fallback = _encode_single_emulinear(layer)
        captured.append(
            _CapturedLayer(
                name=name,
                op_kind="linear",
                spec=layer.spec,
                encoded=[enc],
                target_weight=target,
                fallback=fallback,
                transform=layer.transform,
                bias=None if layer.bias is None else layer.bias.detach().float().cpu(),
            )
        )
    for name, conv in convs.items():
        _check_layer_supported(name, conv.spec)
        if not conv._engines:
            conv.refresh()
        group_encs: list[Encoded] = []
        group_targets: list[torch.Tensor] = []
        fallbacks: list[str] = []
        for eng in conv._engines:
            enc, tgt, fb = _encode_single_emulinear(eng)
            group_encs.append(enc)
            group_targets.append(tgt)
            if fb:
                fallbacks.append(fb)
        full_target = torch.cat(group_targets, dim=0).reshape(conv.weight.shape).contiguous()
        captured.append(
            _CapturedLayer(
                name=name,
                op_kind="conv2d",
                spec=conv.spec,
                encoded=group_encs,
                target_weight=full_target,
                fallback=fallbacks[0] if fallbacks else None,
                transform=LinearTransform("none"),
                bias=None if conv.bias is None else conv.bias.detach().float().cpu(),
                conv_ref=conv,
            )
        )
    if not captured:
        raise ValueError("model has no patched EmuLinear or EmuConv2d layers to export")
    return captured


# ---------- ONNX subgraph builders ----------
class _GraphBuilder:
    """Appends initializers and decode nodes to an ``onnx.ModelProto`` with collision-free names."""

    def __init__(self, model_proto: Any) -> None:
        import onnx

        self.onnx = onnx
        self.helper = onnx.helper
        self.numpy_helper = onnx.numpy_helper
        self.TensorProto = onnx.TensorProto
        self.model = model_proto
        self.initializers = {init.name: init for init in model_proto.graph.initializer}
        self.used_names = (
            set(self.initializers)
            | {vi.name for vi in model_proto.graph.input}
            | {vi.name for vi in model_proto.graph.output}
            | {out for node in model_proto.graph.node for out in node.output}
            | {node.name for node in model_proto.graph.node if node.name}
        )
        self.prepended_nodes: list[Any] = []
        self.payloads: dict[str, str] = {}
        self.used_ms_domain = False
        self._counter = 0

    def fresh(self, base: str) -> str:
        clean = base.replace(".", "_").replace("/", "_")
        cand = clean
        while cand in self.used_names:
            self._counter += 1
            cand = f"{clean}_{self._counter}"
        self.used_names.add(cand)
        return cand

    def add_init(self, base: str, array: np.ndarray) -> str:
        name = self.fresh(base)
        tensor = self.numpy_helper.from_array(np.ascontiguousarray(array), name)
        self.model.graph.initializer.append(tensor)
        self.initializers[name] = tensor
        return name

    def add_node(self, op_type: str, inputs: list[str], outputs: list[str], **attrs: Any) -> Any:
        node = self.helper.make_node(op_type, inputs, outputs, name=self.fresh(f"n_{op_type}"), **attrs)
        self.prepended_nodes.append(node)
        return node

    def emit_unpack(self, prefix: str, packed_u8: np.ndarray, bits: int, count: int) -> str:
        """Emit opset-21 nodes unpacking ``packed_u8`` (``pack_bits`` layout) into INT32 ``[count]``."""
        TP = self.TensorProto
        # The package pads to eight codes; ONNX stores only the last occupied byte.
        nbytes = (bits * count + 7) // 8
        if packed_u8.size < nbytes:
            raise ValueError(f"{prefix}: truncated packed codes")
        p_name = self.add_init(f"{prefix}__codes", packed_u8.reshape(-1)[:nbytes].astype(np.uint8))
        self.payloads[prefix] = p_name
        if bits == 8:
            sliced = p_name
            if nbytes != count:
                s0 = self.add_init(f"{prefix}__s0", np.array([0], dtype=np.int64))
                s1 = self.add_init(f"{prefix}__s1", np.array([count], dtype=np.int64))
                sliced = self.fresh(f"{prefix}__sl")
                self.add_node("Slice", [p_name, s0, s1], [sliced])
            out = self.fresh(f"{prefix}__i32")
            self.add_node("Cast", [sliced], [out], to=TP.INT32)
            return out

        if bits in (1, 2, 4):
            per_byte = 8 // bits
            sh_2d = self.add_init(f"{prefix}__sh2d", np.array([-1, 1], dtype=np.int64))
            shifts = self.add_init(
                f"{prefix}__rsh", np.array([i * bits for i in range(per_byte)], dtype=np.uint8)
            )
            mask = self.add_init(f"{prefix}__msk", np.array([(1 << bits) - 1], dtype=np.uint8))
            u2d = self.fresh(f"{prefix}__u2d")
            shf = self.fresh(f"{prefix}__shf")
            msk = self.fresh(f"{prefix}__mskd")
            self.add_node("Reshape", [p_name, sh_2d], [u2d])
            self.add_node("BitShift", [u2d, shifts], [shf], direction="RIGHT")
            self.add_node("BitwiseAnd", [shf, mask], [msk])
            raw_mat = msk
        else:
            # bits in (3, 5, 6, 7): 8 codes packed LSB-first across ``bits`` bytes -> UINT32 (3b) or UINT64.
            word_tp = TP.UINT32 if bits <= 3 else TP.UINT64
            np_word = np.uint32 if bits <= 3 else np.uint64
            sh_grp = self.add_init(f"{prefix}__shg", np.array([-1, bits], dtype=np.int64))
            g_u8 = self.fresh(f"{prefix}__gu8")
            g_w = self.fresh(f"{prefix}__gw")
            cols = [self.fresh(f"{prefix}__c{j}") for j in range(bits)]
            if nbytes % bits:
                pads = self.add_init(f"{prefix}__bytepad", np.array([0, -nbytes % bits], dtype=np.int64))
                padded = self.fresh(f"{prefix}__bytes")
                self.add_node("Pad", [p_name, pads], [padded])
                p_name = padded
            self.add_node("Reshape", [p_name, sh_grp], [g_u8])
            self.add_node("Cast", [g_u8], [g_w], to=word_tp)
            self.add_node("Split", [g_w], cols, axis=1, num_outputs=bits)
            acc = cols[0]
            for j in range(1, bits):
                lsh = self.add_init(f"{prefix}__lsh{j}", np.array([8 * j], dtype=np_word))
                shifted = self.fresh(f"{prefix}__cs{j}")
                combined = self.fresh(f"{prefix}__or{j}")
                self.add_node("BitShift", [cols[j], lsh], [shifted], direction="LEFT")
                self.add_node("BitwiseOr", [acc, shifted], [combined])
                acc = combined
            rshifts = self.add_init(f"{prefix}__rsh", np.array([i * bits for i in range(8)], dtype=np_word))
            mask = self.add_init(f"{prefix}__msk", np.array([(1 << bits) - 1], dtype=np_word))
            shf = self.fresh(f"{prefix}__shf")
            msk = self.fresh(f"{prefix}__mskd")
            self.add_node("BitShift", [acc, rshifts], [shf], direction="RIGHT")
            self.add_node("BitwiseAnd", [shf, mask], [msk])
            raw_mat = msk

        sh_flat = self.add_init(f"{prefix}__shf1", np.array([-1], dtype=np.int64))
        flat = self.fresh(f"{prefix}__flat")
        self.add_node("Reshape", [raw_mat, sh_flat], [flat])
        total_slots = nbytes * per_byte if bits in (1, 2, 4) else ((nbytes + bits - 1) // bits) * 8
        if total_slots != count:
            s0 = self.add_init(f"{prefix}__s0", np.array([0], dtype=np.int64))
            s1 = self.add_init(f"{prefix}__s1", np.array([count], dtype=np.int64))
            trimmed = self.fresh(f"{prefix}__trim")
            self.add_node("Slice", [flat, s0, s1], [trimmed])
            flat = trimmed
        out = self.fresh(f"{prefix}__i32")
        self.add_node("Cast", [flat], [out], to=TP.INT32)
        return out

    def emit_scale_f32(self, prefix: str, scale_tensor: torch.Tensor, meta: dict[str, Any]) -> str:
        """Store ``scale_tensor`` at its true bit-width / narrow dtype; return its FP32 tensor name."""
        TP = self.TensorProto
        kind = meta["scale_dtype"]
        shape = [int(d) for d in meta["scale_shape"]]
        if kind in ("packed_pow2", "packed_float"):
            s_bits = int(meta["scale_bits"])
            s_count = int(meta["scale_count"])
            codes_i32 = self.emit_unpack(f"{prefix}__sc", scale_tensor.numpy(), s_bits, s_count)
            fmt = get_format(meta["scale_format"])
            lut = (
                pow2_codebook(fmt).numpy() if isinstance(fmt, Pow2Format) else float_codebook(fmt).numpy()  # type: ignore[arg-type]
            )
            lut_init = self.add_init(f"{prefix}__sclut", lut.astype(np.float32))
            gathered = self.fresh(f"{prefix}__scgat")
            self.add_node("Gather", [lut_init, codes_i32], [gathered], axis=0)
            if shape:
                sh = self.add_init(f"{prefix}__scsh", np.array(shape, dtype=np.int64))
                out = self.fresh(f"{prefix}__scf32")
                self.add_node("Reshape", [gathered, sh], [out])
                return out
            sh = self.add_init(f"{prefix}__scsh0", np.array([], dtype=np.int64))
            out = self.fresh(f"{prefix}__scf32")
            self.add_node("Reshape", [gathered, sh], [out])
            return out
        if kind == "float16":
            raw = self.add_init(f"{prefix}__scfp16", scale_tensor.numpy().astype(np.float16))
            out = self.fresh(f"{prefix}__scf32")
            self.add_node("Cast", [raw], [out], to=TP.FLOAT)
            return out
        if kind == "bfloat16":
            import ml_dtypes

            arr = scale_tensor.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
            raw = self.add_init(f"{prefix}__scbf16", arr)
            out = self.fresh(f"{prefix}__scf32")
            self.add_node("Cast", [raw], [out], to=TP.FLOAT)
            return out
        return self.add_init(f"{prefix}__scf32", scale_tensor.float().numpy().astype(np.float32))

    def emit_expand_2d(
        self,
        prefix: str,
        domain_tensor: str,
        granularity: str,
        rows: int,
        k: int,
        group_size: int = 0,
        block: list[int] | None = None,
    ) -> str:
        """Expand a scale or zero-point tensor to ``[rows, k]`` exactly like ``expand_scale``."""
        if granularity in ("tensor", "row"):
            sh = self.add_init(f"{prefix}__exsh", np.array([rows, k], dtype=np.int64))
            out = self.fresh(f"{prefix}__exp")
            self.add_node("Expand", [domain_tensor, sh], [out])
            return out
        if granularity == "group":
            ng = (k + group_size - 1) // group_size
            sh3 = self.add_init(f"{prefix}__sh3", np.array([rows, ng, 1], dtype=np.int64))
            ex3 = self.add_init(f"{prefix}__ex3", np.array([rows, ng, group_size], dtype=np.int64))
            sh2 = self.add_init(f"{prefix}__sh2", np.array([rows, ng * group_size], dtype=np.int64))
            u3 = self.fresh(f"{prefix}__u3")
            e3 = self.fresh(f"{prefix}__e3")
            r2 = self.fresh(f"{prefix}__r2")
            self.add_node("Reshape", [domain_tensor, sh3], [u3])
            self.add_node("Expand", [u3, ex3], [e3])
            self.add_node("Reshape", [e3, sh2], [r2])
            if ng * group_size != k:
                s0 = self.add_init(f"{prefix}__gs0", np.array([0], dtype=np.int64))
                s1 = self.add_init(f"{prefix}__gs1", np.array([k], dtype=np.int64))
                ax = self.add_init(f"{prefix}__gax", np.array([1], dtype=np.int64))
                sliced = self.fresh(f"{prefix}__gsl")
                self.add_node("Slice", [r2, s0, s1, ax], [sliced])
                return sliced
            return r2
        assert block is not None and len(block) == 2
        br, bc = int(block[0]), int(block[1])
        nr, nc = (rows + br - 1) // br, (k + bc - 1) // bc
        sh4 = self.add_init(f"{prefix}__bsh4", np.array([nr, 1, nc, 1], dtype=np.int64))
        ex4 = self.add_init(f"{prefix}__bex4", np.array([nr, br, nc, bc], dtype=np.int64))
        sh2 = self.add_init(f"{prefix}__bsh2", np.array([nr * br, nc * bc], dtype=np.int64))
        u4 = self.fresh(f"{prefix}__bu4")
        e4 = self.fresh(f"{prefix}__be4")
        r2 = self.fresh(f"{prefix}__br2")
        self.add_node("Reshape", [domain_tensor, sh4], [u4])
        self.add_node("Expand", [u4, ex4], [e4])
        self.add_node("Reshape", [e4, sh2], [r2])
        if nr * br != rows or nc * bc != k:
            s0 = self.add_init(f"{prefix}__bs0", np.array([0, 0], dtype=np.int64))
            s1 = self.add_init(f"{prefix}__bs1", np.array([rows, k], dtype=np.int64))
            ax = self.add_init(f"{prefix}__bax", np.array([0, 1], dtype=np.int64))
            sliced = self.fresh(f"{prefix}__bsl")
            self.add_node("Slice", [r2, s0, s1, ax], [sliced])
            return sliced
        return r2

    def emit_dequant_round(self, prefix: str, tensor_f32: str, qspec: Any) -> str:
        """Apply ``round_to_format(..., qspec.dequant_format)`` when ``qspec.mma_input == 'dequant'``."""
        if qspec is None or qspec.mma_input != "dequant":
            return tensor_f32
        TP = self.TensorProto
        fmt_name = qspec.dequant_format.name
        if fmt_name == "fp32":
            return tensor_f32
        cast_to = TP.BFLOAT16 if fmt_name == "bf16" else TP.FLOAT16 if fmt_name == "fp16" else None
        if cast_to is None:
            return tensor_f32
        mid = self.fresh(f"{prefix}__dq_narrow")
        out = self.fresh(f"{prefix}__dq_f32")
        self.add_node("Cast", [tensor_f32], [mid], to=cast_to)
        self.add_node("Cast", [mid], [out], to=TP.FLOAT)
        return out

    def emit_encoded_2d(self, prefix: str, enc: Encoded, spec: Any) -> tuple[str, dict[str, Any]]:
        """Emit ONNX ops decoding one 2-D ``Encoded`` matrix ``[rows, k]`` to FP32, preferring native
        opset-21 ``DequantizeLinear`` for 4/8-bit types and bit-packed ``pack_bits`` streams otherwise."""
        rows, k = enc.shape
        meta = enc.meta
        tensors = enc.tensors
        qspec = getattr(spec, "weight", None)

        if enc.kind == "dense":
            w = tensors["weight"]
            if w.dtype == torch.bfloat16:
                import ml_dtypes

                arr = w.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
                raw = self.add_init(f"{prefix}__dense_bf16", arr)
                self.payloads[prefix] = raw
                out = self.fresh(f"{prefix}__dense_f32")
                self.add_node("Cast", [raw], [out], to=self.TensorProto.FLOAT)
                return out, {"encoding": "dense_bf16", "bits": 16, "params": rows * k}
            out = self.add_init(f"{prefix}__dense_f32", w.float().numpy().astype(np.float32))
            self.payloads[prefix] = out
            return out, {"encoding": "dense_fp32", "bits": 32, "params": rows * k}

        # Try native opset-21 DequantizeLinear (INT4, UINT4, INT8, UINT8, FLOAT8*) when bit-exact
        native = self._try_emit_native_dequantize_linear(prefix, enc, qspec)
        if native is not None:
            return native

        if enc.kind == "nf4":
            from .weight_quant.nf4 import NF4_CODE

            count = int(meta["count"])
            bs = int(meta["blocksize"])
            codes_i32 = self.emit_unpack(prefix, tensors["codes"].numpy(), 4, count)
            lut_init = self.add_init(f"{prefix}__nf4lut", np.array(NF4_CODE, dtype=np.float32))
            vals = self.fresh(f"{prefix}__nf4v")
            self.add_node("Gather", [lut_init, codes_i32], [vals], axis=0)
            pad = (-count) % bs
            if pad:
                pads = self.add_init(f"{prefix}__pad", np.array([0, pad], dtype=np.int64))
                zero = self.add_init(f"{prefix}__p0", np.array(0.0, dtype=np.float32))
                padded = self.fresh(f"{prefix}__vp")
                self.add_node("Pad", [vals, pads, zero], [padded])
                vals = padded
            sh_blk = self.add_init(f"{prefix}__bsh", np.array([-1, bs], dtype=np.int64))
            v_blk = self.fresh(f"{prefix}__vblk")
            self.add_node("Reshape", [vals, sh_blk], [v_blk])
            absmax = self.add_init(
                f"{prefix}__absmax", tensors["absmax"].float().numpy().reshape(-1, 1).astype(np.float32)
            )
            scaled = self.fresh(f"{prefix}__nf4sc")
            self.add_node("Mul", [v_blk, absmax], [scaled])
            if pad:
                sh_1d = self.add_init(f"{prefix}__sh1d", np.array([-1], dtype=np.int64))
                s0 = self.add_init(f"{prefix}__ns0", np.array([0], dtype=np.int64))
                s1 = self.add_init(f"{prefix}__ns1", np.array([count], dtype=np.int64))
                f1d = self.fresh(f"{prefix}__nf1d")
                slc = self.fresh(f"{prefix}__nfsl")
                self.add_node("Reshape", [scaled, sh_1d], [f1d])
                self.add_node("Slice", [f1d, s0, s1], [slc])
                scaled = slc
            sh_2d = self.add_init(f"{prefix}__sh2d", np.array([rows, k], dtype=np.int64))
            mat = self.fresh(f"{prefix}__nfmat")
            self.add_node("Reshape", [scaled, sh_2d], [mat])
            # NF4 always rounds to BF16 (as bitsandbytes / tricast.weight_quant.nf4 does)
            bf = self.fresh(f"{prefix}__nfbf")
            out = self.fresh(f"{prefix}__nff32")
            self.add_node("Cast", [mat], [bf], to=self.TensorProto.BFLOAT16)
            self.add_node("Cast", [bf], [out], to=self.TensorProto.FLOAT)
            return out, {"encoding": "bitpack_nf4", "bits": 4, "params": rows * k}

        if enc.kind == "razer":
            from .weight_quant.razer import INLIER

            count = int(meta["count"])
            gs = int(meta["group_size"])
            codes_i32 = self.emit_unpack(prefix, tensors["codes"].numpy(), 4, count)
            sh_blk = self.add_init(f"{prefix}__rzsh", np.array([-1, gs], dtype=np.int64))
            c_blk = self.fresh(f"{prefix}__rzcb")
            self.add_node("Reshape", [codes_i32, sh_blk], [c_blk])
            fp4_lut = self.add_init(f"{prefix}__fp4lut", float_codebook(FP4_E2M1).numpy().astype(np.float32))
            elem = self.fresh(f"{prefix}__rzel")
            self.add_node("Gather", [fp4_lut, c_blk], [elem], axis=0)

            sc_u8 = self.add_init(f"{prefix}__rzsc", tensors["scale"].numpy().astype(np.uint8))
            m80 = self.add_init(f"{prefix}__m80", np.array([0x80], dtype=np.uint8))
            m7f = self.add_init(f"{prefix}__m7f", np.array([0x7F], dtype=np.uint8))
            z_u8 = self.add_init(f"{prefix}__zu8", np.array([0], dtype=np.uint8))
            f_hi = self.fresh(f"{prefix}__rzhi")
            f_lo = self.fresh(f"{prefix}__rzlo")
            use_pos = self.fresh(f"{prefix}__rzpos")
            lo_i32 = self.fresh(f"{prefix}__rzloi")
            self.add_node("BitwiseAnd", [sc_u8, m80], [f_hi])
            self.add_node("Greater", [f_hi, z_u8], [use_pos])
            self.add_node("BitwiseAnd", [sc_u8, m7f], [f_lo])
            self.add_node("Cast", [f_lo], [lo_i32], to=self.TensorProto.INT32)
            # Positive E4M3 table (codes 0..127)
            e4m3_pos = float_codebook(get_format("fp8_e4m3")).numpy()[:128].astype(np.float32)  # type: ignore[arg-type]
            e4m3_lut = self.add_init(f"{prefix}__e4m3lut", e4m3_pos)
            bs_1d = self.fresh(f"{prefix}__rzbs1")
            self.add_node("Gather", [e4m3_lut, lo_i32], [bs_1d], axis=0)
            sh_c1 = self.add_init(f"{prefix}__shc1", np.array([-1, 1], dtype=np.int64))
            bs_2d = self.fresh(f"{prefix}__rzbs2")
            pos_2d = self.fresh(f"{prefix}__rzp2")
            self.add_node("Reshape", [bs_1d, sh_c1], [bs_2d])
            self.add_node("Reshape", [use_pos, sh_c1], [pos_2d])
            p5 = self.add_init(f"{prefix}__p5", np.array(INLIER, dtype=np.float32))
            n5 = self.add_init(f"{prefix}__n5", np.array(-INLIER, dtype=np.float32))
            spec_val = self.fresh(f"{prefix}__rzsp")
            self.add_node("Where", [pos_2d, p5, n5], [spec_val])
            eight = self.add_init(f"{prefix}__c8", np.array([8], dtype=np.int32))
            is_sp = self.fresh(f"{prefix}__rzis8")
            chosen = self.fresh(f"{prefix}__rzch")
            self.add_node("Equal", [c_blk, eight], [is_sp])
            self.add_node("Where", [is_sp, spec_val, elem], [chosen])
            prod1 = self.fresh(f"{prefix}__rzp1")
            self.add_node("Mul", [chosen, bs_2d], [prod1])
            ts = self.add_init(f"{prefix}__rzts", tensors["global_scale"].float().numpy().astype(np.float32))
            prod2 = self.fresh(f"{prefix}__rzp2f")
            self.add_node("Mul", [prod1, ts], [prod2])
            sh_out = self.add_init(f"{prefix}__rzsho", np.array([rows, k], dtype=np.int64))
            mat = self.fresh(f"{prefix}__rzmat")
            bf = self.fresh(f"{prefix}__rzbf")
            out = self.fresh(f"{prefix}__rzf32")
            self.add_node("Reshape", [prod2, sh_out], [mat])
            self.add_node("Cast", [mat], [bf], to=self.TensorProto.BFLOAT16)
            self.add_node("Cast", [bf], [out], to=self.TensorProto.FLOAT)
            return out, {"encoding": "bitpack_razer", "bits": 4, "params": rows * k}

        # General bit-packed QTensor ("affine" or "float", 1..8 bits)
        assert qspec is not None
        bits = int(meta["bits"])
        count = int(meta["count"])
        codes_i32 = self.emit_unpack(prefix, tensors["codes"].numpy(), bits, count)
        if enc.kind == "affine":
            assert isinstance(qspec.format, IntFormat)
            lut_t = int_codebook(qspec.format).to(container_dtype(qspec.format)).float()
            enc_tag = f"bitpack_int{bits}"
        else:
            assert isinstance(qspec.format, FloatFormat)
            lut_t = float_codebook(qspec.format).to(container_dtype(qspec.format)).float()
            enc_tag = f"bitpack_{qspec.format.name}"
        lut_init = self.add_init(f"{prefix}__lut", lut_t.numpy().astype(np.float32))
        flat_vals = self.fresh(f"{prefix}__vals1d")
        sh_2d = self.add_init(f"{prefix}__sh2d", np.array([rows, k], dtype=np.int64))
        cur = self.fresh(f"{prefix}__vals2d")
        self.add_node("Gather", [lut_init, codes_i32], [flat_vals], axis=0)
        self.add_node("Reshape", [flat_vals, sh_2d], [cur])

        gran = str(meta["granularity"])
        gs = int(meta.get("group_size", 0))
        blk = meta.get("block")

        if "zero" in tensors and meta.get("zero_kind") != "float":
            z_f32 = self._emit_int_zero_f32(prefix, tensors["zero"], meta)
            z_exp = self.emit_expand_2d(f"{prefix}__zp", z_f32, gran, rows, k, gs, blk)
            sub = self.fresh(f"{prefix}__subz")
            self.add_node("Sub", [cur, z_exp], [sub])
            cur = sub

        if "scale" in tensors:
            s_f32 = self.emit_scale_f32(prefix, tensors["scale"], meta)
            s_exp = self.emit_expand_2d(f"{prefix}__sc", s_f32, gran, rows, k, gs, blk)
            mul = self.fresh(f"{prefix}__muls")
            self.add_node("Mul", [cur, s_exp], [mul])
            cur = mul

        if "zero" in tensors and meta.get("zero_kind") == "float":
            zf_meta = {"scale_shape": meta["zero_shape"], "scale_dtype": meta["zero_dtype"]}
            zf_f32 = self.emit_scale_f32(f"{prefix}__zpf", tensors["zero"], zf_meta)
            zf_exp = self.emit_expand_2d(f"{prefix}__zpf", zf_f32, gran, rows, k, gs, blk)
            added = self.fresh(f"{prefix}__addz")
            self.add_node("Add", [cur, zf_exp], [added])
            cur = added

        if "global_scale" in tensors:
            gs_init = self.add_init(
                f"{prefix}__gscale", tensors["global_scale"].float().numpy().astype(np.float32)
            )
            gmul = self.fresh(f"{prefix}__mulg")
            self.add_node("Mul", [cur, gs_init], [gmul])
            cur = gmul

        cur = self.emit_dequant_round(prefix, cur, qspec)
        return cur, {"encoding": enc_tag, "bits": bits, "params": rows * k}

    def _emit_int_zero_f32(self, prefix: str, zero_tensor: torch.Tensor, meta: dict[str, Any]) -> str:
        z_bits = int(meta["zero_bits"])
        z_shape = [int(d) for d in meta["zero_shape"]]
        if z_bits in (16, 32):
            arr = zero_tensor.to(torch.float32).numpy().reshape(z_shape).astype(np.float32)
            return self.add_init(f"{prefix}__zwide", arr)
        bits = int(meta["bits"])
        qmin = int(meta["qmin"])
        z_count = int(meta["zero_count"])
        z_codes = self.emit_unpack(f"{prefix}__zp", zero_tensor.numpy(), bits, z_count)
        lut = (np.arange(1 << bits, dtype=np.float32) + np.float32(qmin)).astype(np.float32)
        lut_init = self.add_init(f"{prefix}__zplut", lut)
        gat = self.fresh(f"{prefix}__zpgat")
        self.add_node("Gather", [lut_init, z_codes], [gat], axis=0)
        sh = self.add_init(f"{prefix}__zpsh", np.array(z_shape, dtype=np.int64))
        out = self.fresh(f"{prefix}__zpf32")
        self.add_node("Reshape", [gat, sh], [out])
        return out

    def _try_emit_native_dequantize_linear(
        self, prefix: str, enc: Encoded, qspec: Any
    ) -> tuple[str, dict[str, Any]] | None:
        """Emit a native ONNX opset-21 ``DequantizeLinear`` node when ``enc`` is an exact 4-bit or 8-bit
        type supported by ``onnxruntime``."""
        import ml_dtypes

        if qspec is None or "global_scale" in enc.tensors or "scale" not in enc.tensors:
            return None
        meta = enc.meta
        rows, k = enc.shape
        gran = str(meta["granularity"])
        gs = int(meta.get("group_size", 0))
        if gran not in ("tensor", "row", "group"):
            return None
        if gran == "group" and (gs <= 0 or k % gs != 0):
            return None

        if enc.kind == "affine":
            bits = int(meta["bits"])
            if bits not in (4, 8) or int(meta.get("frac_bits", 0)) != 0:
                return None
            if "zero" in enc.tensors and (
                meta.get("zero_kind") == "float" or int(meta.get("zero_bits", 0)) != bits
            ):
                return None
            assert isinstance(qspec.format, IntFormat)
            signed = qspec.format.signed
            np_dt: Any = (
                (ml_dtypes.int4 if signed else ml_dtypes.uint4)
                if bits == 4
                else (np.int8 if signed else np.uint8)
            )
            dtype_label = f"{'int' if signed else 'uint'}{bits}"
            codes = unpack_bits(enc.tensors["codes"], bits, rows * k).to(torch.int64) + int(meta["qmin"])
            q_arr = codes.reshape(rows, k).numpy().astype(np_dt)
            if "zero" in enc.tensors:
                zc = unpack_bits(enc.tensors["zero"], bits, int(meta["zero_count"])).to(torch.int64) + int(
                    meta["qmin"]
                )
                zp_arr = zc.numpy().astype(np_dt)
            else:
                zp_arr = None
        elif enc.kind == "float":
            fmt_name = qspec.format.name
            f8_map = {
                "fp8_e4m3": (ml_dtypes.float8_e4m3fn, "float8e4m3fn"),
                "fp8_e5m2": (ml_dtypes.float8_e5m2, "float8e5m2"),
                "fp8_e4m3fnuz": (ml_dtypes.float8_e4m3fnuz, "float8e4m3fnuz"),
                "fp8_e5m2fnuz": (ml_dtypes.float8_e5m2fnuz, "float8e5m2fnuz"),
            }
            if fmt_name not in f8_map or "zero" in enc.tensors:
                return None
            np_dt, dtype_label = f8_map[fmt_name]
            bits = 8
            lut = float_codebook(qspec.format)
            codes = unpack_bits(enc.tensors["codes"], 8, rows * k)
            vals = (lut[codes.long()].reshape(rows, k) + 0.0).numpy()
            q_arr = vals.astype(np_dt)
            zp_arr = None
        else:
            return None

        q_name = self.add_init(f"{prefix}__q", q_arr)
        self.payloads[prefix] = q_name
        s_f32 = self.emit_scale_f32(prefix, enc.tensors["scale"], meta)
        dq_kwargs: dict[str, Any] = {}
        if gran == "tensor":
            sh_s = self.add_init(f"{prefix}__sc0sh", np.array([], dtype=np.int64))
            s_in = self.fresh(f"{prefix}__sc0")
            self.add_node("Reshape", [s_f32, sh_s], [s_in])
            zp_shape: tuple[int, ...] = ()
        elif gran == "row":
            sh_s = self.add_init(f"{prefix}__sc1sh", np.array([rows], dtype=np.int64))
            s_in = self.fresh(f"{prefix}__sc1")
            self.add_node("Reshape", [s_f32, sh_s], [s_in])
            dq_kwargs["axis"] = 0
            zp_shape = (rows,)
        else:
            sh_s = self.add_init(f"{prefix}__sc2sh", np.array([rows, k // gs], dtype=np.int64))
            s_in = self.fresh(f"{prefix}__sc2")
            self.add_node("Reshape", [s_f32, sh_s], [s_in])
            dq_kwargs["axis"] = 1
            dq_kwargs["block_size"] = gs
            zp_shape = (rows, k // gs)

        inputs = [q_name, s_in]
        if zp_arr is not None:
            inputs.append(self.add_init(f"{prefix}__zp", zp_arr.reshape(zp_shape)))
        deq = self.fresh(f"{prefix}__deq")
        self.add_node("DequantizeLinear", inputs, [deq], **dq_kwargs)
        out = self.emit_dequant_round(prefix, deq, qspec)
        return out, {"encoding": f"dequantize_linear_{dtype_label}", "bits": bits, "params": rows * k}

    def try_emit_matmul_nbits(self, init_name: str, cap: _CapturedLayer) -> dict[str, Any] | None:
        """Replace ``MatMul(x, <layer>.weight_t)`` with ``com.microsoft::MatMulNBits`` for eligible 2/4/8-bit
        group-quantized Linear layers."""
        if cap.op_kind != "linear" or len(cap.encoded) != 1:
            return None
        enc = cap.encoded[0]
        if enc.kind != "affine" or "global_scale" in enc.tensors or "scale" not in enc.tensors:
            return None
        meta = enc.meta
        bits = int(meta["bits"])
        gs = int(meta.get("group_size", 0))
        n_out, k_in = enc.shape
        if bits not in (2, 4, 8) or int(meta.get("frac_bits", 0)) != 0:
            return None
        if meta.get("granularity") != "group" or gs < 16 or (gs & (gs - 1)) != 0 or k_in % gs != 0:
            return None
        if "zero" in enc.tensors and (
            meta.get("zero_kind") == "float" or int(meta.get("zero_bits", 0)) != bits
        ):
            return None
        qmin = int(meta["qmin"])
        if "zero" not in enc.tensors and not (0 <= -qmin < (1 << bits)):
            return None

        consumers = [(idx, node) for idx, node in enumerate(self.model.graph.node) if init_name in node.input]
        if len(consumers) != 1:
            return None
        node_idx, matmul_node = consumers[0]
        if matmul_node.op_type != "MatMul" or list(matmul_node.input)[1] != init_name:
            return None

        n_blocks = k_in // gs
        per_byte = 8 // bits
        codes = unpack_bits(enc.tensors["codes"], bits, n_out * k_in).to(torch.int64)
        b_reshaped = codes.reshape(n_out, n_blocks, gs // per_byte, per_byte)
        shifts = torch.arange(per_byte, dtype=torch.int64) * bits
        b_packed = (b_reshaped << shifts).sum(dim=-1).to(torch.uint8).numpy()

        from .bitpack import decode_scale_tensor

        scales_f32 = decode_scale_tensor(enc.tensors["scale"], meta).reshape(n_out * n_blocks).numpy()
        if "zero" in enc.tensors:
            zp_codes = (
                unpack_bits(enc.tensors["zero"], bits, n_out * n_blocks)
                .to(torch.int64)
                .reshape(n_out, n_blocks)
            )
        else:
            zp_codes = torch.full((n_out, n_blocks), -qmin, dtype=torch.int64)
        zp_pad = (-n_blocks) % per_byte
        zp_padded = F.pad(zp_codes, (0, zp_pad)).reshape(n_out, -1, per_byte)
        zp_packed = (zp_padded << shifts).sum(dim=-1).to(torch.uint8).reshape(-1).numpy()

        prefix = cap.name
        b_init = self.add_init(f"{prefix}__nbits_B", b_packed)
        self.payloads[prefix] = b_init
        s_init = self.add_init(f"{prefix}__nbits_scales", scales_f32.astype(np.float32))
        z_init = self.add_init(f"{prefix}__nbits_zp", zp_packed)

        old_init = self.initializers.pop(init_name)
        self.model.graph.initializer.remove(old_init)
        nbits_node = self.helper.make_node(
            "MatMulNBits",
            [matmul_node.input[0], b_init, s_init, z_init],
            list(matmul_node.output),
            domain="com.microsoft",
            name=self.fresh(f"{prefix}__MatMulNBits"),
            K=k_in,
            N=n_out,
            bits=bits,
            block_size=gs,
        )
        self.model.graph.node.remove(matmul_node)
        self.model.graph.node.insert(node_idx, nbits_node)
        self.used_ms_domain = True
        # MatMulNBits computes (code - zp) * scale in fp32 and does not apply the recipe's dequant rounding
        # (e.g. BF16 for mma_input: dequant), so record whether the weight it uses is TriCast's bit for bit.
        nb_weight = (
            (codes.reshape(n_out, n_blocks, gs).float() - zp_codes.reshape(n_out, n_blocks, 1).float())
            * torch.from_numpy(scales_f32).reshape(n_out, n_blocks, 1)
        ).reshape(n_out, k_in)
        exact = same_bits(nb_weight, cap.target_weight)
        return {
            "encoding": f"ort_matmul_nbits_uint{bits}",
            "bits": bits,
            "block_size": gs,
            "params": n_out * k_in,
            "bit_exact_weight": exact,
            "weight_max_abs_diff": 0.0 if exact else float((nb_weight - cap.target_weight).abs().max()),
        }


def _rewrite_weights(
    model_proto: Any,
    captured: list[_CapturedLayer],
    init_prefix: str,
    kernel: KernelMode,
) -> tuple[dict[str, dict[str, Any]], dict[str, tuple[str, torch.Tensor]]]:
    """Rewrite float weight initializers into true-bitwidth ONNX subgraphs.

    Returns ``(stored_meta, decode_targets)`` where ``decode_targets`` maps each weight tensor name in the
    rewritten graph to ``(layer_name, expected_fp32_tensor)`` for bit-for-bit ORT verification.
    """
    gb = _GraphBuilder(model_proto)
    stored: dict[str, dict[str, Any]] = {}
    decode_targets: dict[str, tuple[str, torch.Tensor]] = {}

    for cap in captured:
        init_name = (
            f"{init_prefix}{cap.name}.weight_t"
            if cap.op_kind == "linear"
            else f"{init_prefix}{cap.name}.weight"
        )
        if init_name not in gb.initializers:
            raise RuntimeError(f"{cap.name}: expected ONNX initializer {init_name!r} was not found")

        if kernel == "matmul_nbits":
            nbits_info = gb.try_emit_matmul_nbits(init_name, cap)
            if nbits_info is not None:
                stored[cap.name] = {"op": cap.op_kind, **nbits_info}
                if cap.fallback:
                    stored[cap.name]["fallback"] = cap.fallback
                continue

        old_init = gb.initializers.pop(init_name)
        gb.model.graph.initializer.remove(old_init)

        if cap.op_kind == "linear":
            mat_2d, info = gb.emit_encoded_2d(cap.name, cap.encoded[0], cap.spec)
            gb.add_node("Transpose", [mat_2d], [init_name], perm=[1, 0])
            decode_targets[init_name] = (cap.name, cap.target_weight.T.contiguous())
        else:
            conv_shape = list(cap.target_weight.shape)
            if len(cap.encoded) == 1:
                mat_2d, info = gb.emit_encoded_2d(cap.name, cap.encoded[0], cap.spec)
                sh_4d = gb.add_init(f"{cap.name}__csh4d", np.array(conv_shape, dtype=np.int64))
                gb.add_node("Reshape", [mat_2d, sh_4d], [init_name])
            else:
                parts: list[str] = []
                info = {}
                groups = []
                for g_idx, g_enc in enumerate(cap.encoded):
                    part, info = gb.emit_encoded_2d(f"{cap.name}__g{g_idx}", g_enc, cap.spec)
                    groups.append(info)
                    parts.append(part)
                joined = gb.fresh(f"{cap.name}__gcat")
                gb.add_node("Concat", parts, [joined], axis=0)
                sh_4d = gb.add_init(f"{cap.name}__csh4d", np.array(conv_shape, dtype=np.int64))
                gb.add_node("Reshape", [joined, sh_4d], [init_name])
                if len({(g["encoding"], g["bits"]) for g in groups}) > 1:
                    info = {"encoding": "mixed_groups", "group_encodings": groups}
                info = {**info, "params": int(cap.target_weight.numel()), "groups": len(cap.encoded)}
            decode_targets[init_name] = (cap.name, cap.target_weight)

        stored[cap.name] = {"op": cap.op_kind, **info}
        if cap.fallback:
            stored[cap.name]["fallback"] = cap.fallback

    for cap in captured:
        names = []
        weight_bytes = 0
        for i, enc in enumerate(cap.encoded):
            prefix = cap.name if len(cap.encoded) == 1 else f"{cap.name}__g{i}"
            payload = gb.payloads[prefix]
            actual_bytes = len(gb.initializers[payload].raw_data)
            expected = (int(np.prod(enc.shape)) * int(enc.meta["bits"]) + 7) // 8
            if actual_bytes != expected:
                raise RuntimeError(f"{cap.name}: {payload} stores {actual_bytes} bytes, expected {expected}")
            names.append(payload)
            weight_bytes += actual_bytes
        stored[cap.name].update(weight_initializers=names, weight_bytes=weight_bytes)

    for node in reversed(gb.prepended_nodes):
        gb.model.graph.node.insert(0, node)

    has_ai_onnx = False
    for opset in gb.model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            opset.version = max(opset.version, 21)
            has_ai_onnx = True
    if not has_ai_onnx:
        gb.model.opset_import.append(gb.helper.make_opsetid("", 21))
    if gb.used_ms_domain and not any(op.domain == "com.microsoft" for op in gb.model.opset_import):
        gb.model.opset_import.append(gb.helper.make_opsetid("com.microsoft", 1))
    gb.model.ir_version = max(gb.model.ir_version, 10)
    return stored, decode_targets


def _verify_decoded_weights(
    model_proto: Any, decode_targets: dict[str, tuple[str, torch.Tensor]]
) -> list[str]:
    """Execute ONLY the weight-decoding subgraph in ``onnxruntime`` (with constant folding disabled so ORT
    runs the exact ONNX decode kernels) and verify bit-for-bit equality against TriCast's operand."""
    if not decode_targets:
        return []
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper

    # Backward slice from decode_targets outputs to collect every node & initializer involved in decoding.
    # Neither the extracted protobuf nor the ORT result needs to contain the entire model.
    # A 0.6B model otherwise materializes several GiB of decoded outputs at once.
    batches: list[dict[str, tuple[str, torch.Tensor]]] = [{}]
    output_bytes = 0
    for name, target in decode_targets.items():
        size = target[1].numel() * target[1].element_size()
        if batches[-1] and output_bytes + size > 64 << 20:
            batches.append({})
            output_bytes = 0
        batches[-1][name] = target
        output_bytes += size
    if len(batches) > 1:
        return [name for batch in batches for name in _verify_decoded_weights(model_proto, batch)]

    needed_vals = set(decode_targets)
    decode_nodes = []
    for node in reversed(model_proto.graph.node):
        if any(out in needed_vals for out in node.output):
            decode_nodes.append(node)
            needed_vals.update(inp for inp in node.input if inp)
    decode_nodes.reverse()
    inits = [init for init in model_proto.graph.initializer if init.name in needed_vals]
    outputs = [
        helper.make_tensor_value_info(out_name, TensorProto.FLOAT, list(expected.shape))
        for out_name, (_lname, expected) in decode_targets.items()
    ]
    sub_graph = helper.make_graph(decode_nodes, "tricast_weight_decode", [], outputs, inits)
    sub_model = helper.make_model(
        sub_graph, opset_imports=list(model_proto.opset_import), ir_version=model_proto.ir_version
    )
    onnx.checker.check_model(sub_model)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = torch.get_num_threads()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(sub_model.SerializeToString(), opts, providers=["CPUExecutionProvider"])
    got_list = sess.run([name for name in decode_targets], {})

    mismatched: list[str] = []
    for (_out_name, (layer_name, expected)), got in zip(decode_targets.items(), got_list, strict=True):
        got_t = torch.from_numpy(got)
        if not same_bits(got_t, expected):
            mismatched.append(layer_name)
    return mismatched


def _initializer_bytes(model_proto: Any) -> int:
    total = 0
    for init in model_proto.graph.initializer:
        if init.raw_data:
            total += len(init.raw_data)
        else:
            # Fallback estimate for typed repeated fields
            elems = int(np.prod(init.dims)) if init.dims else 1
            total += elems * 4
    return total


def _save_onnx(model_proto: Any, out_path: Path, external_data_threshold: int) -> list[Path]:
    import onnx

    out_path.parent.mkdir(parents=True, exist_ok=True)
    data_path = out_path.with_name(out_path.name + ".data")
    use_external = _initializer_bytes(model_proto) >= external_data_threshold
    if use_external:
        onnx.save_model(
            model_proto,
            str(out_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_path.name,
            size_threshold=1024,
        )
        onnx.checker.check_model(str(out_path))
        return [out_path, data_path] if data_path.is_file() else [out_path]
    onnx.checker.check_model(model_proto)
    onnx.save(model_proto, str(out_path))
    return [out_path]


def _check_fallbacks(captured: list[_CapturedLayer], allow: bool) -> None:
    if not captured:
        raise ValueError("no quantized layers to export")
    for cap in captured:
        if any(enc.kind == "dense" for enc in cap.encoded):
            cap.fallback = cap.fallback or "stored as dense weight"
            if not allow:
                raise ValueError(
                    f"{cap.name}: dense fallback is not a true-bitwidth export ({cap.fallback}); "
                    "to retain this layer explicitly use allow_dense_fallback=True or --allow-dense-fallback"
                )


def _new_output(out_path: Path) -> None:
    for suffix in ("", ".data", ".manifest.json"):
        path = out_path.with_name(out_path.name + suffix)
        if os.path.lexists(path):
            raise FileExistsError(f"{path} already exists; choose a new export path")


def _finish_export(
    proto: Any,
    out_path: Path,
    external_data_threshold: int,
    stored: dict[str, dict[str, Any]],
    decode_targets: dict[str, tuple[str, torch.Tensor]],
    kernel: KernelMode,
    feeds: dict[str, np.ndarray],
    ref_cpu: torch.Tensor,
    verify_ort: bool,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Verify the staged graph and its sidecar before publishing to new paths."""
    import onnx
    import onnxruntime as ort

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_path.parent) as tmp:
        staged = Path(tmp) / out_path.name
        files = _save_onnx(proto, staged, external_data_threshold)
        ort_check = None
        if verify_ort:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = torch.get_num_threads()
            sess = ort.InferenceSession(str(staged), opts, providers=["CPUExecutionProvider"])
            got = torch.from_numpy(sess.run(None, feeds)[0]).float()
            if (
                got.shape != ref_cpu.shape
                or not torch.isfinite(got).all()
                or not torch.isfinite(ref_cpu).all()
            ):
                raise ValueError("exported or reference output is non-finite or has a mismatched shape")
            diff = (got - ref_cpu).abs()
            ort_check = {
                "max_abs_diff": float(diff.max()),
                "mean_abs_diff": float(diff.mean()),
                "cosine_similarity": float(
                    F.cosine_similarity(got.flatten().double(), ref_cpu.flatten().double(), dim=0)
                ),
            }

        files_meta = {p.name: {"bytes": p.stat().st_size, "sha256": _sha256_file(p)} for p in files}
        low_bit = [info for info in stored.values() if "fallback" not in info]
        encodings: dict[str, int] = {}
        for info in stored.values():
            key = info["encoding"]
            encodings[key] = encodings.get(key, 0) + 1
        manifest = {
            "schema": SCHEMA,
            **provenance,
            "kernel": kernel,
            "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__,
            "opset": 21,
            "ir_version": int(proto.ir_version),
            "files": files_meta,
            "bytes": sum(f["bytes"] for f in files_meta.values()),
            "quantized_layers": len(low_bit),
            "quantized_params": sum(i["params"] for i in low_bit),
            "quantized_weight_bytes": sum(i["weight_bytes"] for i in low_bit),
            "encodings": encodings,
            "fallback_layers": {n: i["fallback"] for n, i in stored.items() if "fallback" in i},
            "checks": {
                "decoded_equals_tricast_weight": {
                    "checked_layers": len(decode_targets),
                    "matmul_nbits_layers": len(stored) - len(decode_targets),
                    "matmul_nbits_not_bit_exact": sorted(
                        n for n, i in stored.items() if i.get("bit_exact_weight") is False
                    ),
                    "mismatched": [],
                },
                "ort_forward": ort_check,
            },
            "arithmetic": "runtime MatMul/Conv; chip-specific accumulation is not emulated",
            "layers": stored,
        }
        report = staged.with_name(staged.name + ".manifest.json")
        report.write_text(
            json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=1) + "\n", encoding="utf-8"
        )
        # The .onnx becomes visible only after its data file is ready; the manifest is the final marker.
        published: list[Path] = []
        try:
            for source in [*files[1:], staged, report]:
                dest = out_path.parent / source.name
                os.link(source, dest)
                published.append(dest)
        except OSError:
            for dest in published:
                dest.unlink()
            raise
    return manifest


# ---------- public API ----------
def export_onnx(
    model: nn.Module,
    out_path: str | Path,
    *,
    sample_inputs: tuple[torch.Tensor, ...] | torch.Tensor | None = None,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    kernel: KernelMode = "standard",
    external_data_threshold: int = _EXTERNAL_THRESHOLD_DEFAULT,
    verify_ort: bool = True,
    allow_dense_fallback: bool = False,
    live_encoded: dict[str, Encoded] | None = None,
    recipe_text: str | None = None,
    model_id: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Export a patched (and calibrated) PyTorch ``model`` to ``out_path`` with true bit-width weights.

    Does not mutate ``model``. Writes ``<out_path>.manifest.json`` alongside ``out_path`` and returns it.
    """
    if kernel not in ("standard", "matmul_nbits"):
        raise ValueError(f"kernel must be 'standard' or 'matmul_nbits', got {kernel!r}")
    if any(hasattr(mod, "_tricast_attention") for mod in model.modules()):
        raise ValueError("attention arithmetic is not exported to ONNX (weights only)")
    if getattr(model, "_tricast_kv_patched", False):
        raise ValueError("a KV-cache spec is not exported to ONNX (weights only)")

    import onnx

    out_path = Path(out_path)
    _new_output(out_path)
    captured = _capture_patched_model(model, live_encoded=live_encoded)
    _check_fallbacks(captured, allow_dense_fallback)
    causal_lm = _is_hf_causal_lm(model)

    if sample_inputs is None:
        if not causal_lm:
            raise ValueError("sample_inputs is required when exporting a non-CausalLM nn.Module")
        ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        sample_tuple: tuple[torch.Tensor, ...] = (ids, torch.ones_like(ids))
    elif isinstance(sample_inputs, torch.Tensor):
        sample_tuple = (sample_inputs,)
    else:
        sample_tuple = tuple(sample_inputs)

    cpu_inputs = tuple(t.detach().cpu() for t in sample_tuple)
    if causal_lm and input_names is None:
        input_names = ["input_ids", "attention_mask"]
        output_names = ["logits"]
        dynamic_axes = {
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        }
    else:
        input_names = input_names or [f"input_{i}" for i in range(len(cpu_inputs))]
        output_names = output_names or ["output_0"]

    # Reference output on the original patched model before temporary swap
    orig_device = next((p.device for p in model.parameters()), torch.device("cpu"))
    training = {mod: mod.training for mod in model.modules()}
    if any(p.device != orig_device for p in model.parameters()):
        raise ValueError("the trace expects a model on one device; sharded models are not supported")
    model.eval()
    saved_modules: list[tuple[nn.Module, str, nn.Module]] = []
    cfg = getattr(model, "config", None)
    change_attn = causal_lm and cfg is not None and hasattr(cfg, "_attn_implementation")
    saved_attn_impl = cfg._attn_implementation if change_attn else None
    trace_moved = False
    try:
        with torch.no_grad():
            dev_inputs = tuple(t.to(orig_device) for t in cpu_inputs)
            if causal_lm:
                ref_out = model(input_ids=dev_inputs[0], attention_mask=dev_inputs[1], use_cache=False).logits
            else:
                raw_ref = model(*dev_inputs)
                ref_out = raw_ref[0] if isinstance(raw_ref, tuple) else raw_ref
            ref_cpu = ref_out.detach().float().cpu()
        if change_attn:
            cfg._attn_implementation = "eager"
        for cap in captured:
            parent_name, _, child_name = cap.name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            orig_mod = getattr(parent, child_name)
            saved_modules.append((parent, child_name, orig_mod))
            out_fmt = getattr(getattr(cap.spec, "mma", None), "out_format", None)
            out_dtype = _OUT_DTYPES.get(getattr(out_fmt, "name", None))
            if cap.op_kind == "linear":
                trace_mod: nn.Module = _TraceLinear(cap.target_weight, cap.bias, cap.transform, out_dtype)
            else:
                assert cap.conv_ref is not None
                trace_mod = _TraceConv2d(cap.conv_ref, cap.target_weight, out_dtype)
            setattr(parent, child_name, trace_mod)

        # EmuLinear._apply requantizes RTN weights (including stochastic rounding).
        # Detach these original modules before any device transfer and restore them last.
        if orig_device.type != "cpu":
            trace_moved = True
            model.to("cpu")
        trace_root: nn.Module = _CausalLMExportWrapper(model) if causal_lm else model
        init_prefix = "model." if causal_lm else ""

        with tempfile.TemporaryDirectory() as tmp_dir:
            float_onnx = Path(tmp_dir) / "traced.float.onnx"
            torch.onnx.export(
                trace_root,
                cpu_inputs,
                str(float_onnx),
                opset_version=20,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                dynamo=False,
            )
            proto = onnx.load(str(float_onnx))
    finally:
        if trace_moved:
            # Only ordinary modules were moved; leave the original quantized modules untouched.
            for parent, child_name, _orig in saved_modules:
                setattr(parent, child_name, nn.Identity())
            model.to(orig_device)
        for parent, child_name, orig_mod in saved_modules:
            setattr(parent, child_name, orig_mod)
        if change_attn:
            cfg._attn_implementation = saved_attn_impl
        for mod, flag in training.items():
            mod.training = flag

    stored, decode_targets = _rewrite_weights(proto, captured, init_prefix, kernel)
    mismatched = _verify_decoded_weights(proto, decode_targets)
    if mismatched:
        raise RuntimeError(
            f"{len(mismatched)} ONNX layers do not decode bit-for-bit to TriCast's weight, "
            f"e.g. {mismatched[0]}"
        )

    feeds = {name: t.numpy() for name, t in zip(input_names, cpu_inputs, strict=True)} if verify_ort else {}
    return _finish_export(
        proto,
        out_path,
        external_data_threshold,
        stored,
        decode_targets,
        kernel,
        feeds,
        ref_cpu,
        verify_ort,
        {
            "model": model_id,
            "revision": revision,
            "recipe_sha256": hashlib.sha256(recipe_text.encode()).hexdigest()
            if recipe_text is not None
            else None,
        },
    )


def _checkpoint_storage(proto: Any, tensors: dict[str, torch.Tensor], tied: dict[str, str]) -> dict[str, Any]:
    """Store untouched checkpoint tensors once in their source dtype, without changing FP32 arithmetic.

    A trace may materialize a transposed Linear weight under an anonymous initializer name. Resolve it
    through the named module's actual weight input, never by finding another tensor of the same shape.
    """
    import ml_dtypes
    from onnx import TensorProto, numpy_helper

    gb = _GraphBuilder(proto)
    by_output = {out: node for node in proto.graph.node for out in node.output}

    def resolve(value: str, transposed: bool) -> tuple[str, bool] | None:
        while value not in gb.initializers:
            node = by_output.get(value)
            if node is None:
                return None
            if node.op_type == "Transpose":
                perm = next((list(a.ints) for a in node.attribute if a.name == "perm"), None)
                if perm not in (None, [1, 0]):
                    return None
                transposed = not transposed
            elif node.op_type not in ("Identity", "Cast"):
                return None
            value = node.input[0]
        return value, transposed

    def source_refs(key: str) -> set[tuple[str, bool]]:
        refs = set()
        named = resolve("model." + key, False)
        if named is not None:
            refs.add(named)
        if key.endswith(".weight"):
            module = key.removesuffix(".weight")
            for node in proto.graph.node:
                # TorchScript node scopes keep either dotted module names or nested path components.
                scope = node.name.rpartition("/")[0].replace("/", ".").strip(".")
                if scope != "model." + module and scope != module:
                    continue
                if node.op_type == "Gather":
                    ref = resolve(node.input[0], False)
                elif node.op_type == "MatMul":
                    ref = resolve(node.input[1], True)
                elif node.op_type == "Gemm":
                    trans_b = next((attr.i for attr in node.attribute if attr.name == "transB"), 0)
                    ref = resolve(node.input[1], not trans_b)
                else:
                    continue
                if ref is not None:
                    refs.add(ref)
        return refs

    groups: dict[str, list[str]] = {}
    for name in tensors:
        groups.setdefault(tied.get(name, name), []).append(name)
    unmapped = []
    rewritten: dict[str, tuple[str, bool]] = {}
    removed: set[str] = set()
    tied_payloads = {}
    saved = 0
    for source, keys in groups.items():
        weight = tensors[source].detach().cpu()
        refs: set[tuple[str, bool]] = set()
        for name in keys:
            matched = source_refs(name)
            if not matched:
                if len(keys) > 1:
                    raise ValueError(f"{name}: cannot locate declared tied checkpoint weight in ONNX")
                unmapped.append(name)
            refs.update(matched)
        if not refs:
            continue
        # For unrelated, already shared constants (e.g. all-one RMSNorm vectors), retain the exporter's
        # identity chain. Only keys explicitly tied in the package may create new sharing here.
        if any(init in rewritten for init, _ in refs):
            if len(keys) > 1:
                raise ValueError(f"{source}: tied source initializer is shared outside its checkpoint group")
            unmapped.append(source)
            continue
        if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            unmapped.extend(k for k in keys if k not in unmapped)
            continue
        for init, transposed in refs:
            array = numpy_helper.to_array(gb.initializers[init])
            # Compare along the checkpoint's rows in small chunks: embeddings can be several GB.
            if transposed:
                array = array.T
            if array.shape != tuple(weight.shape) or array.dtype != np.dtype("float32"):
                raise ValueError(
                    f"{source}: unexpected traced initializer {init} ({array.shape}, {array.dtype})"
                )
            width = max(1, weight.numel() // max(1, weight.shape[0])) if weight.ndim else 1
            source_rows = weight.reshape(-1, width)
            rows = array.reshape(source_rows.shape)
            step = max(1, (1 << 20) // width)
            for offset in range(0, rows.shape[0], step):
                expected = source_rows[offset : offset + step].float().numpy()
                if not np.array_equal(rows[offset : offset + step].view(np.uint32), expected.view(np.uint32)):
                    raise ValueError(f"{source}: initializer {init} does not match the stored checkpoint")
        if weight.dtype == torch.float32 and len(refs) == 1:
            if len(keys) > 1:
                tied_payloads[source] = next(iter(refs))[0]
            continue
        dense = weight.contiguous()
        raw = (
            dense.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
            if dense.dtype == torch.bfloat16
            else dense.numpy()
        )
        name = gb.fresh(source + "__checkpoint")
        init = numpy_helper.from_array(raw, name)
        proto.graph.initializer.append(init)
        cast = name
        if weight.dtype != torch.float32:
            cast = gb.fresh(source + "__fp32")
            gb.add_node("Cast", [name], [cast], to=TensorProto.FLOAT)
        for old, transposed in refs:
            saved += len(gb.initializers[old].raw_data)
            removed.add(old)
            if transposed:
                gb.add_node("Transpose", [cast], [old], perm=[1, 0])
            else:
                gb.add_node("Identity", [cast], [old])
            rewritten[old] = (source, transposed)
        saved -= len(init.raw_data)
        if len(keys) > 1:
            tied_payloads[source] = name
    # Do not copy every large initializer simply to remove the old ones.
    for i in range(len(proto.graph.initializer) - 1, -1, -1):
        if proto.graph.initializer[i].name in removed:
            del proto.graph.initializer[i]
    for node in reversed(gb.prepended_nodes):
        proto.graph.node.insert(0, node)
    return {
        "saved_initializer_bytes": saved,
        "tied": tied,
        "tied_initializers": tied_payloads,
        "unmapped": sorted(set(unmapped)),
    }


def export_package_onnx(
    package_dir: str | Path,
    out_path: str | Path,
    *,
    kernel: KernelMode = "standard",
    external_data_threshold: int = _EXTERNAL_THRESHOLD_DEFAULT,
    verify_ort: bool = True,
    allow_dense_fallback: bool = False,
) -> dict[str, Any]:
    """Convert an existing ``tricast.pack`` directory directly to a true-bitwidth ONNX graph without
    re-running calibration or downloading the original BF16 checkpoint."""
    import onnx
    import yaml
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.modeling_utils import no_init_weights

    from .pack import SCHEMA as PACK_SCHEMA
    from .pack import PackedLinear, _load_remaining_state, decode
    from .recipe import load_recipe

    if kernel not in ("standard", "matmul_nbits"):
        raise ValueError(f"unsupported ONNX kernel {kernel!r}")
    out_path = Path(out_path)
    _new_output(out_path)
    package = Path(package_dir)
    package_manifest = json.loads((package / MANIFEST).read_text(encoding="utf-8"))
    for filename in (CONFIG, WEIGHTS, "config.json"):
        entry = package_manifest["files"].get(filename)
        path = package / filename
        if (
            not entry
            or not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or _sha256_file(path) != entry["sha256"]
        ):
            raise ValueError(f"{package}: missing or mismatched package file {filename}")
    config = json.loads((package / CONFIG).read_text(encoding="utf-8"))
    if config.get("schema") != PACK_SCHEMA or package_manifest.get("schema") != PACK_SCHEMA:
        raise ValueError(f"{package}: not a {PACK_SCHEMA} package")
    if hashlib.sha256(config["recipe"]["yaml"].encode()).hexdigest() != config["recipe"]["sha256"]:
        raise ValueError(f"{package}: recipe SHA-256 mismatch")
    hf_config = AutoConfig.from_pretrained(package, local_files_only=True)
    hf_config._attn_implementation = "eager"
    with no_init_weights(), torch.device("cpu"):
        model = AutoModelForCausalLM.from_config(hf_config, torch_dtype=torch.float32)
    tensors = load_file(str(package / WEIGHTS), device="cpu")
    recipe_text = config["recipe"]["yaml"]
    recipe = load_recipe(yaml.safe_load(recipe_text))
    if recipe.kv is not None:
        raise ValueError(f"{package}: a KV-cache spec is not exported to ONNX")

    decoder = decoder_layers(model)
    captured: list[_CapturedLayer] = []
    packed_layers: dict[str, tuple[dict[str, Any], dict[str, torch.Tensor], Any, torch.Tensor | None]] = {}
    for name, entry in config["layers"].items():
        spec = recipe.spec_for(name, decoder=decoder)
        if spec is None:
            raise ValueError(f"{name}: the stored recipe does not select this layer")
        _check_layer_supported(name, spec)
        stored_t = {
            key[len(name) + 1 :]: tensors.pop(key) for key in list(tensors) if key.startswith(name + ".")
        }
        bias = stored_t.pop("bias", None)
        diag = stored_t.pop("diag", None)
        packed_layers[name] = (entry, stored_t, spec, bias)

        target_w = decode(entry, stored_t, spec)
        tr_info = entry.get("transform") or {"kind": "none"}
        transform = LinearTransform(
            tr_info["kind"], tr_info.get("block", 0), None if diag is None else diag.float()
        )
        meta = {k: v for k, v in entry.items() if k not in ("kind", "shape", "transform", "bias", "fallback")}
        enc = Encoded(entry["kind"], (int(entry["shape"][0]), int(entry["shape"][1])), stored_t, meta)
        captured.append(
            _CapturedLayer(
                name=name,
                op_kind="linear",
                spec=spec,
                encoded=[enc],
                target_weight=target_w,
                fallback=entry.get("fallback"),
                transform=transform,
                bias=None if bias is None else bias.float(),
            )
        )

    _check_fallbacks(captured, allow_dense_fallback)

    # First load into PackedLinear to compute exact reference logits, then swap to _TraceLinear for export
    for name, (entry, stored_t, spec, bias) in packed_layers.items():
        tr_info = entry.get("transform") or {"kind": "none"}
        cap_match = next(c for c in captured if c.name == name)
        t_with_diag = dict(stored_t)
        if cap_match.transform.diag is not None:
            t_with_diag["diag"] = cap_match.transform.diag
        parent_name, _, child_name = name.rpartition(".")
        setattr(
            model.get_submodule(parent_name),
            child_name,
            PackedLinear(name, entry, t_with_diag, spec, bias, mode="load"),
        )
    _load_remaining_state(model, tensors, config, package)
    model.eval()

    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    mask = torch.ones_like(ids)
    with torch.no_grad():
        ref_cpu = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.detach().float().cpu()

    for cap in captured:
        parent_name, _, child_name = cap.name.rpartition(".")
        out_fmt = getattr(getattr(cap.spec, "mma", None), "out_format", None)
        out_dtype = _OUT_DTYPES.get(getattr(out_fmt, "name", None))
        setattr(
            model.get_submodule(parent_name),
            child_name,
            _TraceLinear(cap.target_weight, cap.bias, cap.transform, out_dtype),
        )

    out_path = Path(out_path)
    with tempfile.TemporaryDirectory() as tmp_dir:
        float_onnx = Path(tmp_dir) / "pkg.float.onnx"
        torch.onnx.export(
            _CausalLMExportWrapper(model),
            (ids, mask),
            str(float_onnx),
            opset_version=20,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch", 1: "seq"},
            },
            do_constant_folding=True,
            dynamo=False,
        )
        proto = onnx.load(str(float_onnx))

    checkpoint = _checkpoint_storage(proto, tensors, config.get("tied") or {})
    stored, decode_targets = _rewrite_weights(proto, captured, "model.", kernel)
    mismatched = _verify_decoded_weights(proto, decode_targets)
    if mismatched:
        raise RuntimeError(
            f"{len(mismatched)} ONNX layers do not decode bit-for-bit to package weight, e.g. {mismatched[0]}"
        )
    return _finish_export(
        proto,
        out_path,
        external_data_threshold,
        stored,
        decode_targets,
        kernel,
        {"input_ids": ids.numpy(), "attention_mask": mask.numpy()},
        ref_cpu,
        verify_ort,
        {
            "model": config["model"],
            "revision": config["revision"],
            "recipe_sha256": config["recipe"]["sha256"],
            "source_package": str(package),
            "checkpoint_storage": checkpoint,
        },
    )


def build_onnx(
    model_id: str,
    revision: str | None,
    recipe_path: Path,
    out_path: Path,
    *,
    device: str = "cpu",
    kernel: KernelMode = "standard",
    hessian_max_bytes: int | None = None,
    allow_dense_fallback: bool = False,
) -> dict[str, Any]:
    """Load ``model_id``, patch and calibrate with ``recipe_path`` while capturing live low-bit codes, and
    export to ``out_path``."""
    from . import calibrate, load_recipe, patch_model
    from .eval.runner import _load_model
    from .pack import Capture

    _new_output(Path(out_path))
    torch.manual_seed(42)
    recipe_text = recipe_path.read_text(encoding="utf-8")
    recipe = load_recipe(str(recipe_path))
    if recipe.kv is not None:
        raise ValueError("a KV-cache spec is not exported to ONNX (weights only)")
    dtype = "bfloat16" if device.startswith("cuda") else "float32"
    model, tokenizer = _load_model(model_id, dtype, device, revision=revision)
    actual_rev = getattr(model.config, "_commit_hash", None) or revision
    patch_model(model, recipe)
    capture = Capture(model)
    capture.requantize_uncalibrated()
    if recipe.needs_calibration:
        kwargs = {} if hessian_max_bytes is None else {"hessian_max_bytes": hessian_max_bytes}
        calibrate(model, recipe, tokenizer=tokenizer, device=device, **kwargs)
    capture.requantize_uncalibrated()
    capture.close()
    manifest = export_onnx(
        model,
        out_path,
        kernel=kernel,
        live_encoded=capture.encoded,
        recipe_text=recipe_text,
        model_id=model_id,
        revision=actual_rev,
        allow_dense_fallback=allow_dense_fallback,
    )
    return manifest


def run_onnx(
    onnx_path: str | Path,
    *,
    tokenizer_source: str | Path | None = None,
    prompts: list[str] | None = None,
    max_new_tokens: int = 32,
    threads: int | None = None,
) -> dict[str, Any]:
    """Run greedy generation or a smoke forward pass on a CausalLM ONNX file in ``onnxruntime``."""
    import onnxruntime as ort

    onnx_path = Path(onnx_path)
    opts = ort.SessionOptions()
    if threads is not None:
        opts.intra_op_num_threads = threads
    started = time.time()
    sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    result: dict[str, Any] = {
        "onnx": str(onnx_path),
        "load_seconds": round(time.time() - started, 2),
        "answers": [],
    }
    if not prompts:
        ids = np.array([[1, 2, 3, 4]], dtype=np.int64)
        mask = np.ones_like(ids)
        logits = sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        result["logits_shape"] = list(logits.shape)
        return result

    if tokenizer_source is None:
        raise ValueError("--tokenizer is required when --prompt is given")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    for prompt in prompts:
        if getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            ids_t = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        else:
            ids_t = tokenizer(prompt, return_tensors="pt")["input_ids"]
        cur = ids_t.numpy().astype(np.int64)
        prompt_len = cur.shape[1]
        eos_id = getattr(tokenizer, "eos_token_id", None)
        for _ in range(max_new_tokens):
            mask = np.ones_like(cur)
            logits = sess.run(None, {"input_ids": cur, "attention_mask": mask})[0]
            next_id = int(np.argmax(logits[0, -1]))
            cur = np.concatenate([cur, np.array([[next_id]], dtype=np.int64)], axis=1)
            if eos_id is not None and next_id == eos_id:
                break
        decoded = tokenizer.decode(cur[0, prompt_len:].tolist(), skip_special_tokens=True)
        result["answers"].append({"prompt": prompt, "text": decoded})
    return result


# ---------- CLI ----------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="quantize a Hugging Face causal LM with a recipe and export to ONNX")
    b.add_argument("--model", required=True)
    b.add_argument("--revision")
    b.add_argument("--recipe", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--device", default="cpu")
    b.add_argument("--kernel", choices=("standard", "matmul_nbits"), default="standard")
    b.add_argument("--hessian-max-bytes", type=int)

    fp = sub.add_parser("from-pack", help="convert a tricast.pack directory directly to ONNX")
    fp.add_argument("--package", type=Path, required=True)
    fp.add_argument("--out", type=Path, required=True)
    fp.add_argument("--kernel", choices=("standard", "matmul_nbits"), default="standard")

    r = sub.add_parser("run", help="run inference on an exported ONNX causal LM in onnxruntime")
    r.add_argument("--onnx", type=Path, required=True)
    r.add_argument("--tokenizer", type=str)
    r.add_argument("--prompt", action="append", default=[])
    r.add_argument("--max-new-tokens", type=int, default=32)
    r.add_argument("--threads", type=int)

    for cmd in (b, fp):
        cmd.add_argument("--allow-dense-fallback", action="store_true", help="allow and report dense layers")
    args = parser.parse_args(argv)
    if args.command == "build":
        manifest = build_onnx(
            args.model,
            args.revision,
            args.recipe,
            args.out,
            device=args.device,
            kernel=args.kernel,
            hessian_max_bytes=args.hessian_max_bytes,
            allow_dense_fallback=args.allow_dense_fallback,
        )
        print(
            json.dumps(
                {k: manifest[k] for k in ("bytes", "encodings", "fallback_layers", "checks")},
                ensure_ascii=False,
            )
        )
        print(f"ONNX_EXPORT_DONE out={args.out} bytes={manifest['bytes']}", flush=True)
        return 0
    if args.command == "from-pack":
        manifest = export_package_onnx(
            args.package, args.out, kernel=args.kernel, allow_dense_fallback=args.allow_dense_fallback
        )
        print(
            json.dumps(
                {k: manifest[k] for k in ("bytes", "encodings", "fallback_layers", "checks")},
                ensure_ascii=False,
            )
        )
        print(f"ONNX_FROM_PACK_DONE out={args.out} bytes={manifest['bytes']}", flush=True)
        return 0
    out = run_onnx(
        args.onnx,
        tokenizer_source=args.tokenizer,
        prompts=args.prompt,
        max_new_tokens=args.max_new_tokens,
        threads=args.threads,
    )
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print("ONNX_RUN_DONE", flush=True)
    return 0


# Kept side-effect-free for import; encode_scale_tensor / narrow_float_tensor re-used by callers if needed.
__all__ = [
    "SCHEMA",
    "KernelMode",
    "build_onnx",
    "encode_scale_tensor",
    "export_onnx",
    "export_package_onnx",
    "main",
    "narrow_float_tensor",
    "run_onnx",
]

if __name__ == "__main__":
    sys.exit(main())
