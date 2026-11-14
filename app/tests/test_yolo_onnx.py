"""The detector's ONNX files (scripts/quantizer/yolo_tune.py onnx): each quantized conv weight is stored as
integer codes with one scale per output channel and decoded by DequantizeLinear, and onnxruntime then gives
the same outputs as the float graph it came from. A two-conv network on CPU; needs onnx, onnxruntime and
ml_dtypes (skipped without them)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("ml_dtypes")

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("yolo_tune", ROOT / "scripts" / "quantizer" / "yolo_tune.py")
YT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = YT
SPEC.loader.exec_module(YT)

from tricast.quant.api import quantize  # noqa: E402

METHODS = {"0": "int4", "2": "int8"}  # conv index in the Sequential -> method; the 1x1 conv stays int8
X = torch.randn(2, 3, 16, 16, generator=torch.Generator().manual_seed(42))


def _network() -> torch.nn.Module:
    torch.manual_seed(42)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.SiLU(), torch.nn.Conv2d(8, 4, 1)
    ).eval()


def _quantized(net: torch.nn.Module) -> dict[str, tuple]:
    """Put each conv's fake-quantized weight in place and return its integer codes and scales."""
    weights = {}
    for index, method in METHODS.items():
        conv = net[int(index)]
        rows = conv.weight.detach().reshape(conv.weight.shape[0], -1)
        qt = quantize(rows, YT.weight_spec(method), backend="reference")
        codes = qt.values.to(torch.int64).reshape(conv.weight.shape)
        scale = qt.scale.reshape(-1).float()
        with torch.no_grad():
            conv.weight.copy_(codes.float() * scale.reshape(-1, 1, 1, 1))
        weights[f"{index}.weight"] = (codes.numpy(), scale.numpy(), YT.METHODS[method])
    return weights


def _export(net: torch.nn.Module, path: Path) -> None:
    torch.onnx.export(
        net,
        torch.zeros(1, 3, 16, 16),
        str(path),
        opset_version=20,
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes={"images": {0: "batch"}},
        dynamo=False,
    )


def _run(path: Path) -> torch.Tensor:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return torch.from_numpy(session.run(None, {"images": X.numpy()})[0])


def _decoded(model) -> dict:
    """Run only the DequantizeLinear nodes of a graph in onnxruntime: weight name -> the FP32 array."""
    from onnx import helper

    nodes = [node for node in model.graph.node if node.op_type == "DequantizeLinear"]
    needed = {name for node in nodes for name in node.input}
    graph = helper.make_graph(
        nodes,
        "decode",
        [],
        [helper.make_tensor_value_info(node.output[0], onnx.TensorProto.FLOAT, None) for node in nodes],
        [init for init in model.graph.initializer if init.name in needed],
    )
    decode = helper.make_model(graph, opset_imports=model.opset_import, ir_version=model.ir_version)
    session = ort.InferenceSession(decode.SerializeToString(), providers=["CPUExecutionProvider"])
    return dict(zip([node.output[0] for node in nodes], session.run(None, {}), strict=True))


def test_integer_weights_give_the_float_graphs_outputs(tmp_path: Path) -> None:
    net = _network()
    weights = _quantized(net)
    float_path, int_path = tmp_path / "net.float.onnx", tmp_path / "net.onnx"
    _export(net, float_path)
    stored = YT.int_weights_onnx(float_path, int_path, weights)
    assert stored == {
        "0.weight": {"bits": 4, "dtype": "int4", "params": 8 * 3 * 3 * 3},
        "2.weight": {"bits": 8, "dtype": "int8", "params": 4 * 8},
    }
    model = onnx.load(str(int_path))
    types = {init.name: init.data_type for init in model.graph.initializer}
    assert types["0.weight__q"] == onnx.TensorProto.INT4 and types["2.weight__q"] == onnx.TensorProto.INT8
    assert "0.weight" not in types and "2.weight" not in types  # only the codes are stored
    dequant = [node for node in model.graph.node if node.op_type == "DequantizeLinear"]
    assert sorted(node.output[0] for node in dequant) == ["0.weight", "2.weight"]
    assert all(attr.i == 0 for node in dequant for attr in node.attribute if attr.name == "axis")
    # onnxruntime's DequantizeLinear rebuilds every exported FP32 weight bit for bit
    exported = {
        init.name: onnx.numpy_helper.to_array(init) for init in onnx.load(str(float_path)).graph.initializer
    }
    for name, decoded in _decoded(model).items():
        assert decoded.dtype == exported[name].dtype and (decoded == exported[name]).all(), name
    # the same weights; a convolution whose weight comes from a node instead of an initializer may take
    # another kernel (another order of the FP32 sums), so the outputs agree to rounding, not bit for bit
    ours, theirs = _run(int_path), _run(float_path)
    assert (ours - theirs).abs().max() <= 1e-5 * theirs.abs().max()
    with torch.no_grad():
        assert torch.allclose(ours, net(X), atol=1e-5)


def test_codes_that_do_not_rebuild_the_weight_are_refused(tmp_path: Path) -> None:
    net = _network()
    weights = _quantized(net)
    float_path = tmp_path / "net.float.onnx"
    _export(net, float_path)
    codes, scale, bits = weights["0.weight"]
    bad = codes.copy()
    bad.flat[0] = bad.flat[0] + 1 if bad.flat[0] < 7 else bad.flat[0] - 1
    with pytest.raises(YT.TuneError, match="not the exported weight"):
        YT.int_weights_onnx(float_path, tmp_path / "bad.onnx", {**weights, "0.weight": (bad, scale, bits)})
    with pytest.raises(YT.TuneError, match="no ONNX initializer"):
        YT.int_weights_onnx(float_path, tmp_path / "bad.onnx", {"9.weight": (codes, scale, bits)})
    assert not (tmp_path / "bad.onnx").exists()
