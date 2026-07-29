# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Mixed-FP16 conversion for the browser ONNX graphs.

The browser keeps its public tensor contracts in FP32.  The text encoder and
autoregressive denoiser stay FP32 because continuation-rollout ablations show
that their FP16 errors compound across generated windows.  The structured
decoder uses FP16 linear layers around numerically sensitive FP32 regions.

The policy selectors intentionally use exported module names rather than node
indices.  This keeps the policy reviewable and resilient to harmless exporter
changes while still failing ONNX validation if the resulting graph is invalid.
"""

from __future__ import annotations

import hashlib
import heapq
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import onnx
from onnx import GraphProto, ModelProto, NodeProto, TensorProto
from onnxruntime.transformers.onnx_model import OnnxModel

BrowserGraphName = Literal["text_encoder", "denoiser", "decoder"]
ConversionMode = Literal["mixed-fp16", "fp32-identity"]

# The complete representable range is important here.  In particular, using
# the common converter's 1e-7/1e4 defaults needlessly clips valid weights.
IEEE_FP16_MIN_POSITIVE_SUBNORMAL = 2.0**-24
IEEE_FP16_MAX_FINITE = 65504.0
MIXED_FP16_POLICY_VERSION = 3
_FLOATING_TENSOR_TYPES = frozenset(
    data_type
    for name in (
        "FLOAT",
        "FLOAT16",
        "DOUBLE",
        "BFLOAT16",
        "FLOAT8E4M3FN",
        "FLOAT8E4M3FNUZ",
        "FLOAT8E5M2",
        "FLOAT8E5M2FNUZ",
        "FLOAT4E2M1",
    )
    if (data_type := getattr(TensorProto, name, None)) is not None
)

# Keep this list fixed as part of the policy contract.  ORT's
# DEFAULT_OP_BLOCK_LIST has changed between releases; importing it would make
# an identical export command produce a different precision policy after an
# otherwise harmless dependency update.
MIXED_FP16_FP32_OP_TYPES_V3 = (
    "ArrayFeatureExtractor",
    "Binarizer",
    "CastMap",
    "CategoryMapper",
    "DictVectorizer",
    "FeatureVectorizer",
    "Imputer",
    "LabelEncoder",
    "LinearClassifier",
    "LinearRegressor",
    "Normalizer",
    "OneHotEncoder",
    "RandomUniformLike",
    "SVMClassifier",
    "SVMRegressor",
    "Scaler",
    "TreeEnsembleClassifier",
    "TreeEnsembleRegressor",
    "TreeEnsemble",
    "ZipMap",
    "NonMaxSuppression",
    "TopK",
    "RoiAlign",
    "Range",
    "CumSum",
    "Min",
    "Max",
    "Upsample",
    "LayerNormalization",
)


@dataclass(frozen=True)
class GraphPrecisionPolicy:
    """One explicit FP32-retention policy used during FP16 conversion."""

    graph_name: BrowserGraphName
    policy_id: str
    conversion_mode: ConversionMode
    description: str
    fp32_op_types: tuple[str, ...]
    select_fp32_nodes: Callable[[GraphProto], set[str]]
    production_coverage: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class InitializerPrecisionStats:
    """Logical initializer storage grouped by ONNX element type."""

    count_by_dtype: dict[str, int]
    bytes_by_dtype: dict[str, int]
    total_count: int
    total_bytes: int


@dataclass(frozen=True)
class MixedPrecisionReport:
    """JSON-serializable summary of one converted graph."""

    schema_version: int
    graph_name: BrowserGraphName
    policy_id: str
    conversion_mode: ConversionMode
    source_sha256: str
    output_sha256: str
    source_size_bytes: int
    output_size_bytes: int
    size_reduction_bytes: int
    size_reduction_fraction: float
    source_node_count: int
    output_node_count: int
    output_cast_node_count: int
    fp32_policy_node_count: int
    explicit_fp32_node_count: int
    fp32_policy_nodes_by_op_type: dict[str, int]
    deduplicated_cast_node_count: int
    source_initializers: InitializerPrecisionStats
    output_initializers: InitializerPrecisionStats
    graph_inputs: dict[str, str]
    graph_outputs: dict[str, str]

    def to_dict(self) -> dict:
        """Return a plain dictionary suitable for a manifest or JSON report."""
        return asdict(self)


def _node_basename(node: NodeProto) -> str:
    return node.name.rsplit("/", 1)[-1]


def _is_transformer_attention_node(
    node: NodeProto,
    *,
    path_fragment: str,
    exact_basenames: frozenset[str],
) -> bool:
    return path_fragment in node.name and _node_basename(node) in exact_basenames


def _producer_by_output(graph: GraphProto) -> dict[str, NodeProto]:
    producers: dict[str, NodeProto] = {}
    for node in graph.node:
        for output_name in node.output:
            if not output_name:
                continue
            if output_name in producers:
                raise ValueError(f"ONNX graph has multiple producers for {output_name!r}.")
            producers[output_name] = node
    return producers


def _upstream_nodes_outside_prefix(
    graph: GraphProto,
    initial_value_names: list[str],
    *,
    stop_prefixes: tuple[str, ...],
) -> set[str]:
    """Find named upstream nodes, stopping at known module boundaries."""
    producers = _producer_by_output(graph)
    selected: set[str] = set()
    pending = list(initial_value_names)
    visited_values: set[str] = set()
    while pending:
        value_name = pending.pop()
        if not value_name or value_name in visited_values:
            continue
        visited_values.add(value_name)
        producer = producers.get(value_name)
        if producer is None or producer.name.startswith(stop_prefixes):
            continue
        if not producer.name:
            raise ValueError("FP32 policy traversal encountered an unnamed ONNX node.")
        selected.add(producer.name)
        pending.extend(producer.input)
    return selected


def _text_pooling_nodes(graph: GraphProto) -> set[str]:
    """Resolve the pooling subgraph between the backbone and adapter by names."""
    adapter_inputs: list[str] = []
    for node in graph.node:
        if not node.name.startswith("/adapter/"):
            continue
        adapter_inputs.extend(node.input)

    # Stop at both sides of the named boundary.  Adapter-internal inputs occur
    # when all adapter nodes are used as traversal seeds; stopping at /adapter/
    # prevents those paths from leaking through the region.
    return _upstream_nodes_outside_prefix(
        graph,
        adapter_inputs,
        stop_prefixes=("/backbone/", "/adapter/"),
    )


@dataclass(frozen=True)
class _TextAttentionIsland:
    """One structurally verified score-calculation island."""

    score_matmul: NodeProto
    scale_mul: NodeProto
    mask_add: NodeProto
    softmax: NodeProto
    mask_input: str
    mask_producer: NodeProto | None

    @property
    def nodes(self) -> tuple[NodeProto, NodeProto, NodeProto, NodeProto]:
        return self.score_matmul, self.scale_mul, self.mask_add, self.softmax


def _text_attention_islands(graph: GraphProto) -> list[_TextAttentionIsland]:
    """Find MatMul -> Mul(scale) -> Add(mask) -> Softmax by data flow."""
    producers = _producer_by_output(graph)
    islands: list[_TextAttentionIsland] = []
    for softmax in graph.node:
        if (
            softmax.op_type != "Softmax"
            or not softmax.name.endswith("/attention/self/Softmax")
            or len(softmax.input) != 1
        ):
            continue

        attention_prefix = softmax.name.removesuffix("/Softmax")
        mask_add = producers.get(softmax.input[0])
        if (
            mask_add is None
            or mask_add.op_type != "Add"
            or mask_add.name != f"{attention_prefix}/Add"
            or len(mask_add.input) != 2
        ):
            continue

        scale_candidates = [
            producers[input_name]
            for input_name in mask_add.input
            if input_name in producers
            and producers[input_name].op_type == "Mul"
            and producers[input_name].name == f"{attention_prefix}/Mul"
        ]
        if len(scale_candidates) != 1:
            continue
        scale_mul = scale_candidates[0]

        score_candidates = [
            producers[input_name]
            for input_name in scale_mul.input
            if input_name in producers
            and producers[input_name].op_type == "MatMul"
            and producers[input_name].name == f"{attention_prefix}/MatMul"
        ]
        if len(score_candidates) != 1:
            continue
        score_matmul = score_candidates[0]

        scale_outputs = set(scale_mul.output)
        mask_inputs = [input_name for input_name in mask_add.input if input_name not in scale_outputs]
        if len(mask_inputs) != 1:
            continue
        islands.append(
            _TextAttentionIsland(
                score_matmul=score_matmul,
                scale_mul=scale_mul,
                mask_add=mask_add,
                softmax=softmax,
                mask_input=mask_inputs[0],
                mask_producer=producers.get(mask_inputs[0]),
            )
        )
    return islands


def _select_text_fp32_nodes(graph: GraphProto) -> set[str]:
    # Even sub-millimetre text-condition perturbations can send deterministic
    # autoregressive rollouts down a different trajectory after several
    # windows.  The isolated condition encoder therefore stays byte-identical
    # to its FP32 export.
    return {node.name for node in graph.node}


def _denoiser_cfg_nodes(graph: GraphProto) -> set[str]:
    return _upstream_nodes_outside_prefix(
        graph,
        [output.name for output in graph.output],
        stop_prefixes=("/denoiser/",),
    )


def _select_denoiser_fp32_nodes(graph: GraphProto) -> set[str]:
    # The denoiser is autoregressive: every generated window becomes history
    # for the next.  Five-window ablations found that even selective FP16
    # compute accumulated large position and rotation errors.  Blocking every
    # named node also keeps its initializers in FP32 because conversion uses
    # force_fp16_initializers=False.
    return {node.name for node in graph.node}


def _select_decoder_fp32_nodes(graph: GraphProto) -> set[str]:
    selected: set[str] = set()
    for node in graph.node:
        if not node.name.startswith("/decoder/"):
            selected.add(node.name)
        if _is_transformer_attention_node(
            node,
            path_fragment="/self_attn/",
            exact_basenames=frozenset({"MatMul_1", "Add_3", "Softmax"}),
        ):
            selected.add(node.name)
    return selected


def _named_attention_coverage(
    graph: GraphProto,
    *,
    path_fragment: str,
    exact_basenames: frozenset[str],
) -> tuple[int, int]:
    groups: dict[str, set[str]] = defaultdict(set)
    candidate_count = 0
    for node in graph.node:
        if not _is_transformer_attention_node(
            node,
            path_fragment=path_fragment,
            exact_basenames=exact_basenames,
        ):
            continue
        candidate_count += 1
        groups[node.name.rsplit("/", 1)[0]].add(_node_basename(node))
    complete_islands = sum(basenames == exact_basenames for basenames in groups.values())
    return complete_islands, candidate_count


def _collect_production_policy_coverage(
    graph: GraphProto,
    graph_name: BrowserGraphName,
) -> dict[str, int]:
    layer_normalizations = sum(node.op_type == "LayerNormalization" for node in graph.node)
    if graph_name == "text_encoder":
        islands = _text_attention_islands(graph)
        mask_subgraph_nodes: set[str] = set()
        for island in islands:
            mask_subgraph_nodes.update(
                _upstream_nodes_outside_prefix(
                    graph,
                    [island.mask_input],
                    stop_prefixes=("/backbone/",),
                )
            )
        return {
            "graph_nodes": len(graph.node),
            "attention_islands": len(islands),
            "attention_island_nodes": len({node.name for island in islands for node in island.nodes}),
            "attention_mask_producers": len(
                {island.mask_producer.name for island in islands if island.mask_producer is not None}
            ),
            "attention_mask_subgraph_nodes": len(mask_subgraph_nodes),
            "attention_softmax_candidates": sum(
                node.op_type == "Softmax" and node.name.endswith("/attention/self/Softmax") for node in graph.node
            ),
            "layer_normalizations": layer_normalizations,
            "pooling_nodes": len(_text_pooling_nodes(graph)),
        }

    if graph_name == "denoiser":
        attention_islands, attention_nodes = _named_attention_coverage(
            graph,
            path_fragment="/seqTransEncoder/layers.",
            exact_basenames=frozenset({"MatMul_1", "Add_2", "Softmax"}),
        )
        return {
            "graph_nodes": len(graph.node),
            "attention_islands": attention_islands,
            "attention_island_nodes": attention_nodes,
            "cfg_wrapper_nodes": len(_denoiser_cfg_nodes(graph)),
            "layer_normalizations": layer_normalizations,
            "output_head_nodes": sum(
                node.name.startswith("/denoiser/root_model/output_linear/")
                or node.name.startswith("/denoiser/body_model/output_linear/")
                for node in graph.node
            ),
        }

    attention_islands, attention_nodes = _named_attention_coverage(
        graph,
        path_fragment="/self_attn/",
        exact_basenames=frozenset({"MatMul_1", "Add_3", "Softmax"}),
    )
    return {
        "attention_islands": attention_islands,
        "attention_island_nodes": attention_nodes,
        "layer_normalizations": layer_normalizations,
        "wrapper_nodes": sum(not node.name.startswith("/decoder/") for node in graph.node),
    }


MIXED_FP16_POLICIES: dict[BrowserGraphName, GraphPrecisionPolicy] = {
    "text_encoder": GraphPrecisionPolicy(
        graph_name="text_encoder",
        policy_id="text-continuation-fp32-v3",
        conversion_mode="fp32-identity",
        description=(
            "The complete text-condition graph and all floating-point initializers "
            "remain FP32 because condition perturbations compound in autoregressive "
            "motion rollouts."
        ),
        fp32_op_types=MIXED_FP16_FP32_OP_TYPES_V3,
        select_fp32_nodes=_select_text_fp32_nodes,
        production_coverage=(
            ("graph_nodes", 511),
            ("attention_islands", 6),
            ("attention_island_nodes", 24),
            ("attention_mask_producers", 1),
            ("attention_mask_subgraph_nodes", 9),
            ("attention_softmax_candidates", 6),
            ("layer_normalizations", 15),
            ("pooling_nodes", 44),
        ),
    ),
    "denoiser": GraphPrecisionPolicy(
        graph_name="denoiser",
        policy_id="denoiser-continuation-fp32-v3",
        conversion_mode="fp32-identity",
        description=(
            "The complete autoregressive denoiser graph and all floating-point "
            "initializers remain FP32 because FP16 errors compound across "
            "continuation windows."
        ),
        fp32_op_types=MIXED_FP16_FP32_OP_TYPES_V3,
        select_fp32_nodes=_select_denoiser_fp32_nodes,
        production_coverage=(
            ("graph_nodes", 1500),
            ("attention_islands", 16),
            ("attention_island_nodes", 48),
            ("cfg_wrapper_nodes", 17),
            ("layer_normalizations", 32),
            ("output_head_nodes", 4),
        ),
    ),
    "decoder": GraphPrecisionPolicy(
        graph_name="decoder",
        policy_id="decoder-qk-norm-io-geometry-v3",
        conversion_mode="mixed-fp16",
        description=(
            "FP32 self-attention score MatMul_1/Add_3/Softmax, every "
            "LayerNormalization, and all wrapper preprocessing and postprocessing "
            "outside the /decoder/ neural module."
        ),
        fp32_op_types=MIXED_FP16_FP32_OP_TYPES_V3,
        select_fp32_nodes=_select_decoder_fp32_nodes,
        production_coverage=(
            ("attention_islands", 8),
            ("attention_island_nodes", 24),
            ("layer_normalizations", 16),
            ("wrapper_nodes", 734),
        ),
    ),
}


def get_mixed_fp16_policy(graph_name: BrowserGraphName) -> GraphPrecisionPolicy:
    """Return the reviewed policy for a browser graph."""
    try:
        return MIXED_FP16_POLICIES[graph_name]
    except KeyError as error:
        choices = ", ".join(sorted(MIXED_FP16_POLICIES))
        raise ValueError(f"Unknown browser graph {graph_name!r}; expected one of: {choices}.") from error


def validate_production_policy_coverage(
    model: ModelProto,
    graph_name: BrowserGraphName,
) -> dict[str, int]:
    """Fail if an exported production graph no longer matches its FP32 policy.

    Node names are part of this optimization contract.  Silent partial matches
    would be more dangerous than an export failure: a renamed attention score
    node could otherwise fall back to FP16 without any ablation coverage.
    """
    policy = get_mixed_fp16_policy(graph_name)
    expected = dict(policy.production_coverage)
    observed = _collect_production_policy_coverage(model.graph, graph_name)
    mismatches = {
        key: (expected_value, observed.get(key, 0))
        for key, expected_value in expected.items()
        if observed.get(key, 0) != expected_value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: expected {expected_value}, found {observed_value}"
            for key, (expected_value, observed_value) in sorted(mismatches.items())
        )
        raise ValueError(f"{graph_name} mixed-FP16 policy coverage mismatch ({policy.policy_id}): {details}.")
    return observed


def resolve_fp32_node_names(
    model: ModelProto,
    graph_name: BrowserGraphName,
) -> tuple[str, ...]:
    """Resolve and validate exact node names passed to ORT's converter."""
    policy = get_mixed_fp16_policy(graph_name)
    graph_names = [node.name for node in model.graph.node]
    unnamed_count = sum(not name for name in graph_names)
    if unnamed_count:
        raise ValueError(
            f"{graph_name} has {unnamed_count} unnamed ONNX nodes; "
            "name-based mixed-precision policy cannot be applied safely."
        )
    duplicate_names = sorted(name for name, count in Counter(graph_names).items() if count > 1)
    if duplicate_names:
        raise ValueError(
            f"{graph_name} has duplicate ONNX node names, including {duplicate_names[0]!r}; "
            "name-based mixed-precision policy is ambiguous."
        )

    selected = policy.select_fp32_nodes(model.graph)
    unknown = selected.difference(graph_names)
    if unknown:
        raise ValueError(f"{graph_name} FP32 policy selected unknown node {min(unknown)!r}.")
    return tuple(sorted(selected))


def _cast_semantic_key(node: NodeProto) -> tuple:
    return (
        node.domain,
        tuple(node.input),
        tuple(node.output),
        tuple(attribute.SerializeToString(deterministic=True) for attribute in node.attribute),
    )


def _deduplicate_identical_cast_nodes_in_graph(graph: GraphProto) -> int:
    """Remove duplicate converter Cast producers while rejecting real SSA bugs."""
    producer_by_output: dict[str, NodeProto] = {}
    kept_nodes: list[NodeProto] = []
    removed = 0

    for node in graph.node:
        duplicate_producers: list[NodeProto] = []
        duplicate_producer_ids: set[int] = set()
        for output_name in node.output:
            if not output_name or output_name not in producer_by_output:
                continue
            producer = producer_by_output[output_name]
            if id(producer) not in duplicate_producer_ids:
                duplicate_producers.append(producer)
                duplicate_producer_ids.add(id(producer))
        if duplicate_producers:
            if (
                node.op_type != "Cast"
                or len(duplicate_producers) != 1
                or any(producer.op_type != "Cast" for producer in duplicate_producers)
            ):
                duplicate_name = next(output for output in node.output if output in producer_by_output)
                raise ValueError(f"ONNX SSA violation has non-Cast producers for {duplicate_name!r}.")
            existing = next(iter(duplicate_producers))
            if _cast_semantic_key(existing) != _cast_semantic_key(node):
                duplicate_name = next(output for output in node.output if output in producer_by_output)
                raise ValueError(f"ONNX SSA violation has different Cast producers for {duplicate_name!r}.")
            removed += 1
            continue

        kept_nodes.append(node)
        for output_name in node.output:
            if output_name:
                producer_by_output[output_name] = node

    if removed:
        graph.ClearField("node")
        graph.node.extend(kept_nodes)

    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                removed += _deduplicate_identical_cast_nodes_in_graph(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    removed += _deduplicate_identical_cast_nodes_in_graph(subgraph)
    return removed


def deduplicate_identical_converter_casts(model: OnnxModel) -> int:
    """Repair identical Cast nodes emitted for adjacent FP32-blocked nodes."""
    return _deduplicate_identical_cast_nodes_in_graph(model.model.graph)


def _stable_topological_sort_graph(graph: GraphProto) -> None:
    """Kahn-sort a graph, using original node order as the stable tie-breaker."""
    nodes = list(graph.node)
    producer_index: dict[str, int] = {}
    for index, node in enumerate(nodes):
        for output_name in node.output:
            if not output_name:
                continue
            if output_name in producer_index:
                raise ValueError(f"Cannot topologically sort duplicate ONNX value {output_name!r}.")
            producer_index[output_name] = index

    dependencies: list[set[int]] = []
    consumers: dict[int, list[int]] = defaultdict(list)
    for index, node in enumerate(nodes):
        node_dependencies = {
            producer_index[input_name]
            for input_name in node.input
            if input_name in producer_index and producer_index[input_name] != index
        }
        dependencies.append(node_dependencies)
        for dependency in node_dependencies:
            consumers[dependency].append(index)

    ready = [index for index, node_dependencies in enumerate(dependencies) if not node_dependencies]
    heapq.heapify(ready)
    sorted_indices: list[int] = []
    while ready:
        index = heapq.heappop(ready)
        sorted_indices.append(index)
        for consumer_index in consumers.get(index, ()):
            dependencies[consumer_index].discard(index)
            if not dependencies[consumer_index]:
                heapq.heappush(ready, consumer_index)

    if len(sorted_indices) != len(nodes):
        unresolved = [nodes[index].name for index, deps in enumerate(dependencies) if deps]
        raise ValueError(f"ONNX graph is cyclic; unresolved node: {unresolved[0]!r}.")

    graph.ClearField("node")
    graph.node.extend(nodes[index] for index in sorted_indices)
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                _stable_topological_sort_graph(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    _stable_topological_sort_graph(subgraph)


def stable_topological_sort(model: OnnxModel) -> None:
    """Deterministically restore producer-before-consumer ONNX node order."""
    _stable_topological_sort_graph(model.model.graph)


def _iter_graph_tensors(graph: GraphProto):
    yield from graph.initializer
    for sparse_initializer in graph.sparse_initializer:
        yield sparse_initializer.values
        yield sparse_initializer.indices
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                yield attribute.t
            elif attribute.type == onnx.AttributeProto.TENSORS:
                yield from attribute.tensors
            elif attribute.type == onnx.AttributeProto.GRAPH:
                yield from _iter_graph_tensors(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    yield from _iter_graph_tensors(subgraph)


def validate_no_external_data(model: ModelProto) -> None:
    """Require a genuinely single-file model suitable for a browser pack."""
    for tensor in _iter_graph_tensors(model.graph):
        if tensor.data_location == TensorProto.EXTERNAL or tensor.external_data:
            label = tensor.name or "<unnamed tensor>"
            raise ValueError(f"Browser ONNX graph references external tensor data: {label!r}.")


def _iter_graphs(graph: GraphProto):
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from _iter_graphs(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    yield from _iter_graphs(subgraph)


def _cast_target(node: NodeProto) -> int | None:
    if node.op_type != "Cast":
        return None
    return next(
        (int(attribute.i) for attribute in node.attribute if attribute.name == "to"),
        None,
    )


def validate_fp32_source_graph(model: ModelProto, graph_name: BrowserGraphName) -> None:
    """Require all floating tensors and declared values in an FP32 export to be FLOAT."""
    for tensor in _iter_graph_tensors(model.graph):
        if tensor.data_type in _FLOATING_TENSOR_TYPES and tensor.data_type != TensorProto.FLOAT:
            label = tensor.name or "<unnamed tensor>"
            raise ValueError(f"{graph_name} FP32 source contains {_dtype_name(tensor.data_type)} tensor {label!r}.")

    for graph in _iter_graphs(model.graph):
        for value in (*graph.input, *graph.output, *graph.value_info):
            data_type = value.type.tensor_type.elem_type
            if data_type in _FLOATING_TENSOR_TYPES and data_type != TensorProto.FLOAT:
                raise ValueError(f"{graph_name} FP32 source declares {_dtype_name(data_type)} value {value.name!r}.")
        for node in graph.node:
            target = _cast_target(node)
            if target in _FLOATING_TENSOR_TYPES and target != TensorProto.FLOAT:
                raise ValueError(f"{graph_name} FP32 source Cast {node.name!r} targets {_dtype_name(target)}.")


def validate_no_storage_only_fp16_casts(model: ModelProto, graph_name: BrowserGraphName) -> None:
    """Reject FP16-stored weights that are immediately restored for FP32 compute."""
    reduced_initializers = {
        initializer.name
        for initializer in model.graph.initializer
        if initializer.data_type in _FLOATING_TENSOR_TYPES and initializer.data_type != TensorProto.FLOAT
    }
    for node in model.graph.node:
        if node.input and node.input[0] in reduced_initializers and _cast_target(node) == TensorProto.FLOAT:
            raise ValueError(
                f"{graph_name} contains storage-only reduced-precision initializer {node.input[0]!r} cast back to FP32."
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _io_contract(graph: GraphProto) -> tuple[tuple[tuple[str, bytes], ...], tuple[tuple[str, bytes], ...]]:
    inputs = tuple((value.name, value.type.SerializeToString(deterministic=True)) for value in graph.input)
    outputs = tuple((value.name, value.type.SerializeToString(deterministic=True)) for value in graph.output)
    return inputs, outputs


def _validate_io_contract(
    source_contract: tuple[tuple[tuple[str, bytes], ...], tuple[tuple[str, bytes], ...]],
    converted_graph: GraphProto,
) -> None:
    converted_contract = _io_contract(converted_graph)
    if converted_contract != source_contract:
        raise ValueError("Mixed-FP16 conversion changed the browser graph I/O contract.")


def _dtype_name(data_type: int) -> str:
    try:
        return TensorProto.DataType.Name(data_type).lower()
    except ValueError:
        return f"unknown_{data_type}"


_ELEMENT_BYTES = {
    TensorProto.FLOAT: 4,
    TensorProto.UINT8: 1,
    TensorProto.INT8: 1,
    TensorProto.UINT16: 2,
    TensorProto.INT16: 2,
    TensorProto.INT32: 4,
    TensorProto.INT64: 8,
    TensorProto.BOOL: 1,
    TensorProto.FLOAT16: 2,
    TensorProto.DOUBLE: 8,
    TensorProto.UINT32: 4,
    TensorProto.UINT64: 8,
    TensorProto.COMPLEX64: 8,
    TensorProto.COMPLEX128: 16,
    TensorProto.BFLOAT16: 2,
}


def _logical_tensor_bytes(tensor: TensorProto) -> int:
    element_count = 1
    for dimension in tensor.dims:
        element_count *= dimension
    element_bytes = _ELEMENT_BYTES.get(tensor.data_type)
    if element_bytes is None:
        return len(tensor.raw_data)
    return element_count * element_bytes


def _initializer_precision_stats(model: ModelProto) -> InitializerPrecisionStats:
    count_by_dtype: Counter[str] = Counter()
    bytes_by_dtype: Counter[str] = Counter()
    for initializer in model.graph.initializer:
        dtype = _dtype_name(initializer.data_type)
        count_by_dtype[dtype] += 1
        bytes_by_dtype[dtype] += _logical_tensor_bytes(initializer)
    return InitializerPrecisionStats(
        count_by_dtype=dict(sorted(count_by_dtype.items())),
        bytes_by_dtype=dict(sorted(bytes_by_dtype.items())),
        total_count=sum(count_by_dtype.values()),
        total_bytes=sum(bytes_by_dtype.values()),
    )


def _graph_io_types(values) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        tensor_type = value.type.tensor_type
        result[value.name] = _dtype_name(tensor_type.elem_type)
    return result


def convert_browser_onnx_to_mixed_fp16(
    source_path: Path,
    output_path: Path,
    *,
    graph_name: BrowserGraphName,
    validate_production_policy: bool = True,
) -> MixedPrecisionReport:
    """Convert one self-contained browser graph to the reviewed mixed-FP16 policy.

    ``source_path`` and ``output_path`` may be the same.  The converted graph is
    written atomically only after ONNX checker, I/O, SSA, topology, and
    external-data validation succeeds.  ``validate_production_policy=False`` is
    reserved for synthetic unit graphs and deliberate ablation candidates.
    """
    source_path = Path(source_path)
    output_path = Path(output_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Browser ONNX graph not found: {source_path}")

    source_model = onnx.load(source_path, load_external_data=False)
    validate_no_external_data(source_model)
    onnx.checker.check_model(source_model, full_check=True)
    validate_fp32_source_graph(source_model, graph_name)
    source_contract = _io_contract(source_model.graph)
    source_initializers = _initializer_precision_stats(source_model)
    source_size_bytes = source_path.stat().st_size
    source_sha256 = _sha256(source_path)
    source_node_count = len(source_model.graph.node)

    policy = get_mixed_fp16_policy(graph_name)
    if validate_production_policy:
        validate_production_policy_coverage(source_model, graph_name)
    explicit_fp32_nodes = resolve_fp32_node_names(source_model, graph_name)
    keep_graph_fp32 = policy.conversion_mode == "fp32-identity"
    if keep_graph_fp32 and len(explicit_fp32_nodes) != source_node_count:
        raise ValueError(
            f"{graph_name} FP32-identity policy selected {len(explicit_fp32_nodes)} of {source_node_count} graph nodes."
        )
    fp32_policy_nodes = [
        node
        for node in source_model.graph.node
        if node.name in explicit_fp32_nodes or node.op_type in policy.fp32_op_types
    ]

    converted_model = OnnxModel(source_model)
    if keep_graph_fp32:
        # Do not round-trip an all-FP32 graph through the converter.  Even when
        # every node is blocked, it inserts boundary metadata and rewrites
        # existing Cast nodes, increasing the denoiser by roughly 0.3 MB.
        deduplicated_cast_count = 0
    else:
        converted_model.convert_float_to_float16(
            use_symbolic_shape_infer=False,
            min_positive_val=IEEE_FP16_MIN_POSITIVE_SUBNORMAL,
            max_finite_val=IEEE_FP16_MAX_FINITE,
            keep_io_types=True,
            op_block_list=list(policy.fp32_op_types),
            node_block_list=list(explicit_fp32_nodes),
            force_fp16_initializers=False,
        )
        deduplicated_cast_count = deduplicate_identical_converter_casts(converted_model)
        converted_model.initialize(converted_model.model)
        stable_topological_sort(converted_model)

    validate_no_external_data(converted_model.model)
    _validate_io_contract(source_contract, converted_model.model.graph)
    onnx.checker.check_model(converted_model.model, full_check=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        if keep_graph_fp32:
            shutil.copyfile(source_path, temporary_path)
        else:
            converted_model.save_model_to_file(
                temporary_path.as_posix(),
                use_external_data_format=False,
            )
        persisted_model = onnx.load(temporary_path, load_external_data=False)
        validate_no_external_data(persisted_model)
        _validate_io_contract(source_contract, persisted_model.graph)
        onnx.checker.check_model(persisted_model, full_check=True)
        if keep_graph_fp32:
            validate_fp32_source_graph(persisted_model, graph_name)
        else:
            validate_no_storage_only_fp16_casts(persisted_model, graph_name)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    output_size_bytes = output_path.stat().st_size
    output_sha256 = _sha256(output_path)
    if keep_graph_fp32 and source_sha256 != output_sha256:
        raise RuntimeError(f"{graph_name} FP32-identity conversion was not byte-identical.")
    output_initializers = _initializer_precision_stats(persisted_model)
    output_node_count = len(persisted_model.graph.node)
    reduction_bytes = source_size_bytes - output_size_bytes
    return MixedPrecisionReport(
        schema_version=1,
        graph_name=graph_name,
        policy_id=policy.policy_id,
        conversion_mode=policy.conversion_mode,
        source_sha256=source_sha256,
        output_sha256=output_sha256,
        source_size_bytes=source_size_bytes,
        output_size_bytes=output_size_bytes,
        size_reduction_bytes=reduction_bytes,
        size_reduction_fraction=reduction_bytes / source_size_bytes,
        source_node_count=source_node_count,
        output_node_count=output_node_count,
        output_cast_node_count=sum(node.op_type == "Cast" for node in persisted_model.graph.node),
        fp32_policy_node_count=len(fp32_policy_nodes),
        explicit_fp32_node_count=len(explicit_fp32_nodes),
        fp32_policy_nodes_by_op_type=dict(sorted(Counter(node.op_type for node in fp32_policy_nodes).items())),
        deduplicated_cast_node_count=deduplicated_cast_count,
        source_initializers=source_initializers,
        output_initializers=output_initializers,
        graph_inputs=_graph_io_types(persisted_model.graph.input),
        graph_outputs=_graph_io_types(persisted_model.graph.output),
    )


__all__ = [
    "IEEE_FP16_MAX_FINITE",
    "IEEE_FP16_MIN_POSITIVE_SUBNORMAL",
    "MIXED_FP16_FP32_OP_TYPES_V3",
    "MIXED_FP16_POLICIES",
    "MIXED_FP16_POLICY_VERSION",
    "BrowserGraphName",
    "ConversionMode",
    "GraphPrecisionPolicy",
    "InitializerPrecisionStats",
    "MixedPrecisionReport",
    "convert_browser_onnx_to_mixed_fp16",
    "deduplicate_identical_converter_casts",
    "get_mixed_fp16_policy",
    "resolve_fp32_node_names",
    "stable_topological_sort",
    "validate_fp32_source_graph",
    "validate_no_external_data",
    "validate_no_storage_only_fp16_casts",
    "validate_production_policy_coverage",
]
