# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only shared-expert contract tests.

This fixture intentionally uses only the Python standard library. It extracts
pure helpers and inspects call wiring from the GPU test/selector sources without
importing torch, aiter, ROCm, or FlyDSL.
"""

from __future__ import annotations

import ast
import os
import random
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_TEST_PATH = REPO_ROOT / "my_code" / "test_flydsl_grouped_gemm_gfx1250.py"
SELECTOR_PATH = REPO_ROOT / "aiter" / "ops" / "flydsl" / "grouped_moe_gfx1250.py"
SHARED_EXPERTS_ENV = "AITER_MOE_NUM_SHARED_EXPERTS"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in parsed source")


def _exec_selected(
    path: Path,
    *,
    functions: set[str],
    assignments: set[str] = frozenset(),
    namespace: dict | None = None,
) -> dict:
    """Execute selected pure nodes in original source order."""
    tree = _parse(path)
    body: list[ast.stmt] = ast.parse(
        "from __future__ import annotations"
    ).body
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            body.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {
                target.id for target in targets if isinstance(target, ast.Name)
            }
            if names & assignments:
                body.append(node)
    selected = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    result = {} if namespace is None else dict(namespace)
    exec(compile(selected, str(path), "exec"), result)  # noqa: S102
    return result


def _named_calls(function: ast.FunctionDef, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == name
    ]


class _ActivationType:
    Silu = object()


class SharedGeometryFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        namespace = _exec_selected(
            GPU_TEST_PATH,
            functions={"_resolve_moe_geometry"},
        )
        cls.resolve = staticmethod(namespace["_resolve_moe_geometry"])

    def test_physical_e65_topk9_resolves_to_routed_e64_topk8(self):
        self.assertEqual(self.resolve(65, 9, 1), (64, 8))

    def test_shared_zero_preserves_geometry(self):
        self.assertEqual(self.resolve(64, 8, 0), (64, 8))

    def test_invalid_shared_geometry_is_rejected(self):
        for args in ((65, 9, -1), (65, 9, 2), (1, 2, 1), (65, 1, 1), (3, 4, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.resolve(*args)


class SelectorFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        namespace = _exec_selected(
            SELECTOR_PATH,
            functions={
                "_routed_geometry_from_metadata",
                "_make_tdm_token_profile",
                "_find_routed_e64_topk8_token_profile",
            },
            assignments={
                "_SHARED_EXPERTS_ENV",
                "_E64_TOPK8_TOKEN_PROFILES",
                "_E64_TOPK8_TOKEN_INTERVALS",
            },
            namespace={"os": os, "ActivationType": _ActivationType},
        )
        cls.find_profile = staticmethod(
            namespace["_find_routed_e64_topk8_token_profile"]
        )

    @staticmethod
    def _kwargs(experts: int, topk: int, tokens: int = 1536) -> dict:
        return {
            "tokens": tokens,
            "experts": experts,
            "topk": topk,
            "model_dim": 7168,
            "inter_dim": 2048,
            "activation": _ActivationType.Silu,
            "is_grouped_a4w4": True,
            "use_bias": False,
            "is_ep": False,
            "enable_ep_scatter": False,
        }

    @staticmethod
    def _exact_t256(profile: dict) -> dict:
        keys = (
            "tile_m",
            "tile_m2",
            "n_warp",
            "fused_gemm1_n_warp",
            "n_warp2",
            "num_buffers",
            "partial_m_direct_scale",
            "partial_m_drain_output",
            "allow_partial_m_exact",
        )
        return {key: profile[key] for key in keys}

    def test_t1536_requires_explicit_shared_metadata(self):
        expected = {
            "tile_m": 256,
            "tile_m2": 256,
            "n_warp": 2,
            "fused_gemm1_n_warp": 4,
            "n_warp2": 4,
            "num_buffers": 4,
            "partial_m_direct_scale": 0,
            "partial_m_drain_output": 0,
            "allow_partial_m_exact": 1,
        }
        with patch.dict(os.environ, {SHARED_EXPERTS_ENV: "0"}):
            self.assertIsNone(self.find_profile(**self._kwargs(65, 9)))
            self.assertEqual(
                self._exact_t256(self.find_profile(**self._kwargs(64, 8))),
                expected,
            )
        with patch.dict(os.environ, {SHARED_EXPERTS_ENV: "1"}):
            profile = self.find_profile(**self._kwargs(65, 9))
            self.assertEqual(self._exact_t256(profile), expected)
            self.assertIsNone(self.find_profile(**self._kwargs(66, 9)))
            self.assertIsNone(self.find_profile(**self._kwargs(65, 10)))

    def test_selector_metadata_and_geometry_are_strict(self):
        for raw in ("", "-1", "2", "not-an-int"):
            with self.subTest(raw=raw), patch.dict(
                os.environ, {SHARED_EXPERTS_ENV: raw}
            ), self.assertRaises(ValueError):
                self.find_profile(**self._kwargs(65, 9))
        with patch.dict(os.environ, {SHARED_EXPERTS_ENV: "1"}):
            for experts, topk in ((1, 9), (65, 1), (3, 5)):
                with self.subTest(experts=experts, topk=topk), self.assertRaises(
                    ValueError
                ):
                    self.find_profile(**self._kwargs(experts, topk))

    def test_aligned_override_uses_routed_geometry(self):
        with patch.dict(os.environ, {SHARED_EXPERTS_ENV: "1"}):
            profile = self.find_profile(**self._kwargs(65, 9, tokens=6144))
        self.assertEqual(profile["tile_m"], 256)
        self.assertEqual(profile["n_warp2"], 2)
        self.assertEqual(profile["partial_m_direct_scale"], 0)
        self.assertEqual(profile["partial_m_drain_output"], 0)


class PhysicalRoutingFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = _parse(GPU_TEST_PATH)

    def test_make_topk_uses_only_routed_score_columns_then_appends_shared(self):
        function = _function(self.tree, "_make_topk")
        score_call = _named_calls(function, "_make_routing_score")
        self.assertEqual(len(score_call), 1)
        self.assertEqual(
            [ast.unparse(arg) for arg in score_call[0].args],
            ["tokens", "routed_experts", "routed_topk"],
        )

        fused_call = _named_calls(function, "fused_topk")
        self.assertEqual(len(fused_call), 1)
        self.assertEqual(
            [ast.unparse(arg) for arg in fused_call[0].args],
            ["hidden_states", "score", "routed_topk", "True"],
        )

        assignments = {
            target.id: node.value
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertEqual(ast.unparse(assignments["shared_id"]), "routed_experts")
        self.assertEqual(
            ast.unparse(assignments["physical_ids"]),
            "torch.cat((topk_id, shared_ids), dim=1)",
        )
        self.assertEqual(
            ast.unparse(assignments["physical_weights"]),
            "torch.cat((topk_w, shared_weights), dim=1)",
        )
        self.assertEqual(
            ast.unparse(assignments["shared_weights"].func),
            "torch.ones",
        )

    def test_dut_reference_weights_and_counts_use_physical_routes(self):
        function = _function(self.tree, "_run_grouped_via_fused_moe")
        make_topk = _named_calls(function, "_make_topk")
        self.assertEqual(len(make_topk), 1)
        self.assertEqual(
            [ast.unparse(arg) for arg in make_topk[0].args],
            ["hidden", "experts", "topk"],
        )
        self.assertEqual(
            ast.unparse(
                next(
                    keyword.value
                    for keyword in make_topk[0].keywords
                    if keyword.arg == "shared_experts"
                )
            ),
            "shared_experts",
        )

        fused_moe = _named_calls(function, "fused_moe")
        reference = _named_calls(function, "_torch_moe_ref")
        self.assertEqual(len(fused_moe), 1)
        self.assertEqual(len(reference), 1)
        self.assertEqual(
            [ast.unparse(arg) for arg in fused_moe[0].args[3:5]],
            ["topk_w", "topk_id"],
        )
        self.assertEqual(
            [ast.unparse(arg) for arg in reference[0].args[-2:]],
            ["topk_w", "topk_id"],
        )

        for call in _named_calls(function, "_pattern_packed"):
            self.assertEqual(ast.unparse(call.args[0]), "experts")
        for call in _named_calls(function, "init_weight_scales"):
            self.assertEqual(ast.unparse(call.args[0]), "experts")

        bincount = _named_calls(function, "torch.bincount")
        self.assertEqual(len(bincount), 1)
        self.assertEqual(
            ast.unparse(
                next(
                    keyword.value
                    for keyword in bincount[0].keywords
                    if keyword.arg == "minlength"
                )
            ),
            "experts",
        )

        run_moe = _function(self.tree, "run_moe")
        summary = _named_calls(run_moe, "_build_effective_summary")
        self.assertEqual(len(summary), 1)
        summary_keywords = {
            keyword.arg: ast.unparse(keyword.value) for keyword in summary[0].keywords
        }
        self.assertEqual(summary_keywords["experts"], "experts")
        self.assertEqual(summary_keywords["tokens"], "tokens")
        self.assertEqual(summary_keywords["topk"], "topk")

    def test_random_and_balanced_physical_route_contract(self):
        tokens = 16
        for balanced in (False, True):
            rng = random.Random(0)
            routed = [
                (
                    [(token * 8 + slot) % 64 for slot in range(8)]
                    if balanced
                    else rng.sample(range(64), 8)
                )
                for token in range(tokens)
            ]
            physical_ids = [row + [64] for row in routed]
            physical_weights = [[1.0 / 8] * 8 + [1.0] for _ in range(tokens)]
            with self.subTest(balanced=balanced):
                self.assertTrue(
                    all(len(set(row[:8])) == 8 for row in physical_ids)
                )
                self.assertTrue(
                    all(all(0 <= expert < 64 for expert in row[:8]) for row in physical_ids)
                )
                self.assertTrue(all(row[-1] == 64 for row in physical_ids))
                self.assertTrue(all(row[-1] == 1.0 for row in physical_weights))
                self.assertEqual(
                    sum(expert == 64 for row in physical_ids for expert in row),
                    tokens,
                )


if __name__ == "__main__":
    unittest.main()
