# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for browser mixed-FP16 ONNX conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.transformers.onnx_model import OnnxModel

from ardy.browser.precision import (
    IEEE_FP16_MAX_FINITE,
    IEEE_FP16_MIN_POSITIVE_SUBNORMAL,
    MIXED_FP16_FP32_OP_TYPES_V3,
    MIXED_FP16_POLICIES,
    convert_browser_onnx_to_mixed_fp16,
    deduplicate_identical_converter_casts,
    resolve_fp32_node_names,
    stable_topological_sort,
    validate_fp32_source_graph,
    validate_no_external_data,
    validate_no_storage_only_fp16_casts,
    validate_production_policy_coverage,
)


def _model_with_nodes(nodes, *, output_name: str):
    graph = helper.make_graph(
        nodes,
        "policy-test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 2])],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
    )


def test_text_policy_keeps_the_complete_condition_graph_in_fp32():
    nodes = [
        helper.make_node("Identity", ["input"], ["hidden"], name="/backbone/output/Identity"),
        helper.make_node("Cast", ["input"], ["mask_cast"], name="/Cast", to=TensorProto.FLOAT),
        helper.make_node("Sub", ["mask_cast", "input"], ["mask_delta"], name="/Sub"),
        helper.make_node("Mul", ["mask_delta", "input"], ["mask"], name="/Mul"),
        helper.make_node(
            "MatMul",
            ["hidden", "hidden"],
            ["scores"],
            name="/backbone/encoder/layer.0/attention/self/MatMul",
        ),
        helper.make_node(
            "Mul",
            ["scores", "input"],
            ["scaled_scores"],
            name="/backbone/encoder/layer.0/attention/self/Mul",
        ),
        helper.make_node(
            "Add",
            ["scaled_scores", "mask"],
            ["masked"],
            name="/backbone/encoder/layer.0/attention/self/Add",
        ),
        helper.make_node(
            "Softmax",
            ["masked"],
            ["probabilities"],
            name="/backbone/encoder/layer.0/attention/self/Softmax",
        ),
        helper.make_node(
            "MatMul",
            ["probabilities", "hidden"],
            ["values"],
            name="/backbone/encoder/layer.0/attention/self/MatMul_1",
        ),
        helper.make_node("ReduceSum", ["hidden"], ["pooled"], name="/ReduceSum"),
        helper.make_node(
            "LayerNormalization",
            ["pooled"],
            ["adapted"],
            name="/adapter/adapter.0/LayerNormalization",
        ),
        helper.make_node("Gemm", ["adapted"], ["root"], name="/root_head/Gemm"),
        helper.make_node("Gemm", ["adapted"], ["body"], name="/body_head/Gemm"),
        helper.make_node("Add", ["root", "body"], ["output"], name="/head_merge/Add"),
    ]
    model = _model_with_nodes(nodes, output_name="output")

    selected = set(resolve_fp32_node_names(model, "text_encoder"))

    assert "/backbone/encoder/layer.0/attention/self/MatMul" in selected
    assert "/backbone/encoder/layer.0/attention/self/Mul" in selected
    assert "/backbone/encoder/layer.0/attention/self/Add" in selected
    assert "/backbone/encoder/layer.0/attention/self/Softmax" in selected
    assert "/backbone/encoder/layer.0/attention/self/MatMul_1" in selected
    assert "/Cast" in selected
    assert "/Sub" in selected
    assert "/Mul" in selected
    assert "/ReduceSum" in selected
    assert "/adapter/adapter.0/LayerNormalization" in selected
    assert "/root_head/Gemm" in selected
    assert "/body_head/Gemm" in selected
    assert "LayerNormalization" in MIXED_FP16_POLICIES["text_encoder"].fp32_op_types
    assert MIXED_FP16_POLICIES["text_encoder"].fp32_op_types == MIXED_FP16_FP32_OP_TYPES_V3


def test_text_policy_keeps_a_disconnected_score_chain_in_fp32():
    nodes = [
        helper.make_node(
            "MatMul",
            ["input", "input"],
            ["scores"],
            name="/backbone/encoder/layer.0/attention/self/MatMul",
        ),
        helper.make_node(
            "Mul",
            ["scores", "input"],
            ["scaled_scores"],
            name="/backbone/encoder/layer.0/attention/self/Mul",
        ),
        helper.make_node(
            "Add",
            ["scores", "input"],
            ["masked"],
            name="/backbone/encoder/layer.0/attention/self/Add",
        ),
        helper.make_node(
            "Softmax",
            ["masked"],
            ["output"],
            name="/backbone/encoder/layer.0/attention/self/Softmax",
        ),
    ]
    model = _model_with_nodes(nodes, output_name="output")

    selected = set(resolve_fp32_node_names(model, "text_encoder"))

    assert selected == {node.name for node in nodes}


def test_denoiser_policy_keeps_the_complete_autoregressive_graph_in_fp32():
    nodes = [
        helper.make_node("Concat", ["input", "input"], ["prepared"], name="/Concat"),
        helper.make_node(
            "MatMul",
            ["prepared", "prepared"],
            ["scores"],
            name="/denoiser/root_model/seqTransEncoder/layers.0/MatMul_1",
        ),
        helper.make_node(
            "Add",
            ["scores", "scores"],
            ["masked"],
            name="/denoiser/root_model/seqTransEncoder/layers.0/Add_2",
        ),
        helper.make_node(
            "Softmax",
            ["masked"],
            ["attended"],
            name="/denoiser/root_model/seqTransEncoder/layers.0/Softmax",
        ),
        helper.make_node(
            "MatMul",
            ["attended", "attended"],
            ["root"],
            name="/denoiser/root_model/output_linear/MatMul",
        ),
        helper.make_node(
            "Add",
            ["root", "root"],
            ["core_output"],
            name="/denoiser/Reshape_12",
        ),
        helper.make_node("Sub", ["core_output", "core_output"], ["delta"], name="/Sub"),
        helper.make_node("Add", ["core_output", "delta"], ["output"], name="/Add_1"),
    ]
    model = _model_with_nodes(nodes, output_name="output")

    selected = set(resolve_fp32_node_names(model, "denoiser"))

    assert "/Concat" in selected
    assert "/denoiser/root_model/seqTransEncoder/layers.0/MatMul_1" in selected
    assert "/denoiser/root_model/seqTransEncoder/layers.0/Add_2" in selected
    assert "/denoiser/root_model/seqTransEncoder/layers.0/Softmax" in selected
    assert "/denoiser/root_model/output_linear/MatMul" in selected
    assert "/denoiser/Reshape_12" in selected
    assert "/Sub" in selected
    assert "/Add_1" in selected


def test_decoder_policy_keeps_wrapper_geometry_and_qk_but_not_output_projection():
    nodes = [
        helper.make_node("Add", ["input", "input"], ["prepared"], name="/Add"),
        helper.make_node(
            "MatMul",
            ["prepared", "prepared"],
            ["projected"],
            name="/decoder/input_proj/MatMul",
        ),
        helper.make_node(
            "MatMul",
            ["projected", "projected"],
            ["scores"],
            name="/decoder/seqTransEncoder/layers.0/self_attn/MatMul_1",
        ),
        helper.make_node(
            "Add",
            ["scores", "scores"],
            ["masked"],
            name="/decoder/seqTransEncoder/layers.0/self_attn/Add_3",
        ),
        helper.make_node(
            "Softmax",
            ["masked"],
            ["attended"],
            name="/decoder/seqTransEncoder/layers.0/self_attn/Softmax",
        ),
        helper.make_node(
            "MatMul",
            ["attended", "attended"],
            ["decoded"],
            name="/decoder/output_proj/MatMul",
        ),
        helper.make_node("ReduceL2", ["decoded"], ["output"], name="/ReduceL2"),
    ]
    model = _model_with_nodes(nodes, output_name="output")

    selected = set(resolve_fp32_node_names(model, "decoder"))

    assert "/Add" in selected
    assert "/decoder/input_proj/MatMul" not in selected
    assert "/decoder/seqTransEncoder/layers.0/self_attn/MatMul_1" in selected
    assert "/decoder/seqTransEncoder/layers.0/self_attn/Add_3" in selected
    assert "/decoder/seqTransEncoder/layers.0/self_attn/Softmax" in selected
    assert "/decoder/output_proj/MatMul" not in selected
    assert "/ReduceL2" in selected


def test_production_coverage_guard_rejects_partial_policy_matches():
    model = _model_with_nodes(
        [
            helper.make_node(
                "Softmax",
                ["input"],
                ["output"],
                name="/backbone/encoder/layer.0/attention/self/Softmax",
            )
        ],
        output_name="output",
    )

    with pytest.raises(
        ValueError,
        match=r"attention_islands: expected 6, found 0.*layer_normalizations: expected 15, found 0",
    ):
        validate_production_policy_coverage(model, "text_encoder")


def test_converter_cast_deduplication_repairs_ssa_and_stable_topology():
    consumer = helper.make_node("Add", ["cast", "cast"], ["output"], name="consumer")
    cast_a = helper.make_node("Cast", ["input"], ["cast"], name="converter_cast", to=TensorProto.FLOAT)
    cast_b = helper.make_node("Cast", ["input"], ["cast"], name="converter_cast", to=TensorProto.FLOAT)
    graph = helper.make_graph(
        [consumer, cast_a, cast_b],
        "cast-dedup",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT16, [1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
    )
    model = OnnxModel(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]))

    assert deduplicate_identical_converter_casts(model) == 1
    stable_topological_sort(model)

    assert [node.name for node in model.model.graph.node] == ["converter_cast", "consumer"]
    onnx.checker.check_model(model.model, full_check=True)


def _write_convertible_model(path: Path) -> None:
    weight = numpy_helper.from_array(
        np.asarray([[1.0, -2.0], [3.0, 0.5]], dtype=np.float32),
        name="weight",
    )
    scale = numpy_helper.from_array(np.ones(2, dtype=np.float32), name="scale")
    bias = numpy_helper.from_array(np.zeros(2, dtype=np.float32), name="bias")
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["input", "weight"], ["hidden"], name="/decoder/input_proj/MatMul"),
            helper.make_node(
                "LayerNormalization",
                ["hidden", "scale", "bias"],
                ["output"],
                name="/decoder/norm/LayerNormalization",
                axis=-1,
                epsilon=1e-5,
            ),
        ],
        "mixed-fp16",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
        [weight, scale, bias],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, path)


def test_mixed_fp16_conversion_is_deterministic_and_keeps_float_io(tmp_path: Path):
    source = tmp_path / "source.onnx"
    output_a = tmp_path / "output-a.onnx"
    output_b = tmp_path / "output-b.onnx"
    _write_convertible_model(source)

    report_a = convert_browser_onnx_to_mixed_fp16(
        source,
        output_a,
        graph_name="decoder",
        validate_production_policy=False,
    )
    report_b = convert_browser_onnx_to_mixed_fp16(
        source,
        output_b,
        graph_name="decoder",
        validate_production_policy=False,
    )

    converted = onnx.load(output_a, load_external_data=False)
    assert output_a.read_bytes() == output_b.read_bytes()
    assert converted.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert converted.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
    initializer_types = {item.name: item.data_type for item in converted.graph.initializer}
    assert initializer_types["weight"] == TensorProto.FLOAT16
    assert initializer_types["scale"] == TensorProto.FLOAT
    assert initializer_types["bias"] == TensorProto.FLOAT
    assert report_a.to_dict() == report_b.to_dict()
    producer_by_output = {output: node for node in converted.graph.node for output in node.output}
    assert not any(
        node.op_type == "Cast"
        and node.input
        and (producer := producer_by_output.get(node.input[0])) is not None
        and producer.op_type == "Cast"
        for node in converted.graph.node
    )
    assert report_a.graph_inputs == {"input": "float"}
    assert report_a.graph_outputs == {"output": "float"}
    assert report_a.output_initializers.count_by_dtype == {"float": 2, "float16": 1}
    assert report_a.policy_id == "decoder-qk-norm-io-geometry-v3"
    assert report_a.conversion_mode == "mixed-fp16"
    assert report_a.source_sha256 != report_a.output_sha256
    assert IEEE_FP16_MIN_POSITIVE_SUBNORMAL == 5.960464477539063e-08
    assert IEEE_FP16_MAX_FINITE == 65504.0


def test_denoiser_conversion_keeps_compute_and_initializers_in_fp32(tmp_path: Path):
    source = tmp_path / "source.onnx"
    output = tmp_path / "output.onnx"
    weight = numpy_helper.from_array(
        np.asarray([[1.0, -2.0], [3.0, 0.5]], dtype=np.float32),
        name="weight",
    )
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["input", "weight"], ["hidden"], name="/denoiser/MatMul"),
            helper.make_node("Add", ["hidden", "input"], ["output"], name="/denoiser/Add"),
        ],
        "fp32-denoiser",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
        [weight],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), source)

    report = convert_browser_onnx_to_mixed_fp16(
        source,
        output,
        graph_name="denoiser",
        validate_production_policy=False,
    )

    converted = onnx.load(output, load_external_data=False)
    assert output.read_bytes() == source.read_bytes()
    assert {initializer.data_type for initializer in converted.graph.initializer} == {TensorProto.FLOAT}
    assert not any(node.op_type == "Cast" for node in converted.graph.node)
    assert report.explicit_fp32_node_count == len(converted.graph.node)
    assert report.output_initializers.count_by_dtype == {"float": 1}
    assert report.policy_id == "denoiser-continuation-fp32-v3"
    assert report.conversion_mode == "fp32-identity"
    assert report.source_sha256 == report.output_sha256


def test_fp32_identity_validation_rejects_reduced_precision_tensors_and_casts():
    half_weight = numpy_helper.from_array(
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16),
        name="half_weight",
    )
    graph = helper.make_graph(
        [],
        "half-initializer",
        [],
        [],
        [half_weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    with pytest.raises(ValueError, match=r"FP32 source contains float16 tensor 'half_weight'"):
        validate_fp32_source_graph(model, "denoiser")

    graph = helper.make_graph(
        [
            helper.make_node(
                "Cast",
                ["input"],
                ["output"],
                name="/denoiser/Cast",
                to=TensorProto.FLOAT16,
            ),
        ],
        "half-cast",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    with pytest.raises(ValueError, match=r"FP32 source declares float16 value 'output'"):
        validate_fp32_source_graph(model, "denoiser")


def test_storage_only_fp16_initializer_cast_is_rejected():
    half_weight = numpy_helper.from_array(
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16),
        name="half_weight",
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "Cast",
                ["half_weight"],
                ["weight"],
                name="restore-weight",
                to=TensorProto.FLOAT,
            ),
            helper.make_node("MatMul", ["input", "weight"], ["output"], name="compute"),
        ],
        "storage-only-half",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
        [half_weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    with pytest.raises(ValueError, match=r"storage-only reduced-precision initializer 'half_weight'"):
        validate_no_storage_only_fp16_casts(model, "denoiser")


def test_external_tensor_data_is_rejected(tmp_path: Path):
    source = tmp_path / "external.onnx"
    weight = numpy_helper.from_array(np.ones((2, 2), dtype=np.float32), name="weight")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["output"], name="/decoder/input_proj/MatMul")],
        "external-data",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save_model(
        model,
        source,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="external.weights",
        size_threshold=0,
    )

    unloaded = onnx.load(source, load_external_data=False)
    with pytest.raises(ValueError, match="external tensor data"):
        validate_no_external_data(unloaded)
    with pytest.raises(ValueError, match="external tensor data"):
        convert_browser_onnx_to_mixed_fp16(source, tmp_path / "output.onnx", graph_name="decoder")
