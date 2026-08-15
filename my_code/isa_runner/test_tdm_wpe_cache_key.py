#!/usr/bin/env python3
"""CPU-only tests for the TDM waves-per-EU JIT cache-key plumbing."""

import ast
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


KERNEL_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "aiter"
    / "ops"
    / "flydsl"
    / "kernels"
    / "mxfp4_preshuffle_gfx1250_tdm.py"
)


class _Constexpr:
    def __class_getitem__(cls, item):
        return object


def _kernel_tree():
    return ast.parse(KERNEL_SOURCE.read_text(encoding="utf-8"))


def _load_cpu_testable_symbols():
    wanted = {
        "_normalize_waves_per_eu",
        "_waves_per_eu_value_attrs",
        "launch_gemm_a8w4_tdm",
    }
    definitions = [
        node
        for node in _kernel_tree().body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    ]
    namespace = {
        "Constexpr": _Constexpr,
        "fx": SimpleNamespace(
            Tensor=object,
            Pointer=object,
            Int32=object,
            Stream=object,
            Float32=object,
        ),
        "os": os,
    }
    exec(compile(ast.Module(definitions, type_ignores=[]), KERNEL_SOURCE, "exec"), namespace)
    return namespace


class TdmWavesPerEuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.symbols = _load_cpu_testable_symbols()

    def test_normalization_matrix(self):
        normalize = self.symbols["_normalize_waves_per_eu"]
        cases = {
            "1": (1, 1),
            "1,1": (1, 1),
            " 1 , 2 ": (1, 2),
            "2": (2, 2),
            "2,4": (2, 4),
            "": (0, 0),
            "0": (0, 0),
            "off": (0, 0),
            "OFF": (0, 0),
            "none": (0, 0),
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize(raw), expected)

    def test_normalization_preserves_validation_messages(self):
        normalize = self.symbols["_normalize_waves_per_eu"]
        with self.assertRaisesRegex(
            ValueError,
            "AITER_TDM_WAVES_PER_EU must be N, min,max, or 0/off/none",
        ):
            normalize("1,2,3")
        for raw in ("0,1", "2,1", "-1"):
            with self.subTest(raw=raw), self.assertRaisesRegex(
                ValueError,
                "AITER_TDM_WAVES_PER_EU requires 1 <= min <= max",
            ):
                normalize(raw)

    def test_value_attrs_matrix(self):
        value_attrs = self.symbols["_waves_per_eu_value_attrs"]
        self.assertEqual(value_attrs(0, 0), {})
        self.assertEqual(
            value_attrs(1, 1),
            {"passthrough": [["amdgpu-waves-per-eu", "1,1"]]},
        )
        self.assertEqual(
            value_attrs(2, 4),
            {"passthrough": [["amdgpu-waves-per-eu", "2,4"]]},
        )

    def test_public_wrapper_reads_environment_on_every_call(self):
        launch = self.symbols["launch_gemm_a8w4_tdm"]
        jit_mock = Mock()
        launch.__globals__["_launch_gemm_a8w4_tdm_jit"] = jit_mock
        required_args = [
            object()
            for parameter in inspect.signature(launch).parameters.values()
            if parameter.default is inspect.Parameter.empty
        ]

        with patch.dict(os.environ, {}, clear=True):
            launch(*required_args)
            for raw in ("1", "1,1", "0", "off", "none", "2", "1,2"):
                os.environ["AITER_TDM_WAVES_PER_EU"] = raw
                launch(*required_args)

        normalized = [
            (
                call.kwargs["waves_per_eu_min"],
                call.kwargs["waves_per_eu_max"],
            )
            for call in jit_mock.call_args_list
        ]
        self.assertEqual(
            normalized,
            [
                (1, 1),
                (1, 1),
                (1, 1),
                (0, 0),
                (0, 0),
                (0, 0),
                (2, 2),
                (1, 2),
            ],
        )

    def test_public_signature_keeps_existing_defaults(self):
        parameters = inspect.signature(
            self.symbols["launch_gemm_a8w4_tdm"]
        ).parameters
        self.assertEqual(
            list(parameters),
            [
                "arg_c",
                "arg_a",
                "arg_b",
                "arg_scale_a",
                "arg_scale_b",
                "i32_m",
                "stream",
                "N",
                "K",
                "tile_m",
                "tile_n",
                "tile_k",
                "m_warp",
                "n_warp",
                "out_is_f16",
                "num_buffers",
                "a_is_fp4",
                "arg_m_tile_map",
                "n_experts",
                "stage1_act",
                "has_bias",
                "arg_bias",
                "f32_swiglu_limit",
                "stage1_quant_out",
                "quant_wmma_rep",
                "arg_quant_scale",
            ],
        )
        self.assertEqual(parameters["stage1_quant_out"].default, 0)
        self.assertEqual(parameters["quant_wmma_rep"].default, 1)
        self.assertIsNone(parameters["arg_quant_scale"].default)
        self.assertNotIn("waves_per_eu_min", parameters)
        self.assertNotIn("waves_per_eu_max", parameters)

    def test_jit_body_uses_constexpr_values_not_environment(self):
        functions = {
            node.name: node
            for node in _kernel_tree().body
            if isinstance(node, ast.FunctionDef)
        }
        public = functions["launch_gemm_a8w4_tdm"]
        jit_body = functions["_launch_gemm_a8w4_tdm_jit"]

        self.assertEqual(public.decorator_list, [])
        self.assertEqual(
            [ast.unparse(item) for item in jit_body.decorator_list],
            ["flyc.jit"],
        )
        annotations = {
            argument.arg: ast.unparse(argument.annotation)
            for argument in jit_body.args.args
            if argument.annotation is not None
        }
        self.assertEqual(annotations["waves_per_eu_min"], "Constexpr[int]")
        self.assertEqual(annotations["waves_per_eu_max"], "Constexpr[int]")
        self.assertFalse(
            any(
                isinstance(node, ast.Constant)
                and node.value == "AITER_TDM_WAVES_PER_EU"
                for node in ast.walk(jit_body)
            )
        )
        self.assertTrue(
            any(
                isinstance(node, ast.Constant)
                and node.value == "AITER_TDM_WAVES_PER_EU"
                for node in ast.walk(public)
            )
        )


if __name__ == "__main__":
    unittest.main()
