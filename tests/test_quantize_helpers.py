"""Regression tests for INT8 quantization helpers (shape pin + opset upgrade).

These guard the fixes found during the stock-weights dry run:
  * dynamic input dims broke symbolic shape inference, and
  * per-channel QDQ needs opset >= 13 (the DequantizeLinear `axis` attr).
"""

import pytest

onnx = pytest.importorskip("onnx")

from src.optimization.quantize_int8 import _fix_input_shape  # noqa: E402


def _tiny_dynamic_model(opset: int):
    """A 1-node Identity model with a dynamic-batch float input at given opset."""
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3, "h", "w"])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 3, "h", "w"])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "g", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


def test_fix_input_shape_pins_static_dims(tmp_path):
    model = _tiny_dynamic_model(opset=13)
    out = tmp_path / "static.onnx"
    _fix_input_shape(model, imgsz=640, out_path=out)

    loaded = onnx.load(str(out))
    dims = loaded.graph.input[0].type.tensor_type.shape.dim
    values = [d.dim_value for d in dims]
    assert values == [1, 3, 640, 640]
    # no symbolic (dim_param) dimensions remain on the pinned axes
    assert all(d.dim_param == "" for d in dims)


def test_fix_input_shape_upgrades_opset_to_13(tmp_path):
    model = _tiny_dynamic_model(opset=12)
    out = tmp_path / "static12.onnx"
    _fix_input_shape(model, imgsz=320, out_path=out)

    loaded = onnx.load(str(out))
    opset = max(op.version for op in loaded.opset_import if op.domain in ("", "ai.onnx"))
    assert opset >= 13


def test_fix_input_shape_keeps_high_opset(tmp_path):
    model = _tiny_dynamic_model(opset=17)
    out = tmp_path / "static17.onnx"
    _fix_input_shape(model, imgsz=640, out_path=out)

    loaded = onnx.load(str(out))
    opset = max(op.version for op in loaded.opset_import if op.domain in ("", "ai.onnx"))
    assert opset == 17
