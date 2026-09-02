#!/usr/bin/env python3
"""CPU-only tests for the optional MoE GEMM1 C++ launch backend."""

from __future__ import annotations

import hashlib
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import gemm_batch_isa_runner as batch  # noqa: E402
import gemm_isa_runner as single  # noqa: E402
import moe_pipeline_timing_context as timing_context  # noqa: E402
import moe_cpp_backend as backend  # noqa: E402


class LogicalTrafficAccountingTest(unittest.TestCase):
    def test_store_detector_ignores_comments_metadata_and_lds(self):
        source = """
        ; tensor_store_from_lds s[0:3], s[4:11]
        // buffer_store_b128 v[0:3], v0, s[0:3], 0
        # global_store_dword v0, v1, off
        .name: fake_buffer_store_symbol
        ds_store_b128 v0, v[0:3]
        scratch_store_dword v0, v1, off
        """
        detection = single.detect_global_output_stores(source)
        self.assertFalse(detection.writes_output)
        self.assertEqual(detection.summary, "none")

    def test_store_detector_recognizes_supported_global_families(self):
        for mnemonic in (
            "tensor_store_from_lds s[0:3], s[4:11]",
            "buffer_store_b128 v[0:3], v0, s[0:3], 0",
            "global_store_dword v0, v1, off",
            "flat_store_dword v[0:1], v2",
            "image_store v0, v[1:4], s[0:7] dmask:0x1",
            "buffer_atomic_add v0, v1, s[0:3], 0",
        ):
            with self.subTest(mnemonic=mnemonic):
                detection = single.detect_global_output_stores(mnemonic)
                self.assertTrue(detection.writes_output)
                self.assertEqual(detection.summary, mnemonic.split()[0])

    def test_unknown_store_fails_closed(self):
        with self.assertRaisesRegex(
            single.GemmIsaRunnerError,
            "unclassified store/atomic",
        ):
            single.detect_global_output_stores("mystery_store_dword v0, v1")

    def test_real_full_and_loadonly_sources(self):
        my_code = _HERE.parent
        cases = (
            ("moe_gemm1_a4w4_v0.s", True, "tensor_store_from_lds"),
            ("moe_gemm1_a4w4_v0_loadonly.s", False, "none"),
            (
                "f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps.s",
                True,
                "tensor_store_from_lds",
            ),
            (
                "f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps_v20.s",
                True,
                "tensor_store_from_lds",
            ),
            (
                "f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps/"
                "f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps_"
                "loadonly_v17.s",
                False,
                "none",
            ),
            ("MAB/mab_tdm_gemm_full_batch.s", True, "buffer_store_b128"),
            (
                "MAB/mab_tdm_gemm_full_batch_loadonly.s",
                False,
                "none",
            ),
        )
        for relative, expected, mnemonic in cases:
            with self.subTest(source=relative):
                source = (my_code / relative).read_text(
                    encoding="utf-8",
                    errors="replace",
                )
                detection = single.detect_global_output_stores(source)
                self.assertEqual(detection.writes_output, expected)
                self.assertEqual(detection.summary, mnemonic)

    def test_full_loadonly_traffic_diff_is_exact_output(self):
        full = single.GlobalOutputStoreDetection(("tensor_store_from_lds",))
        loadonly = single.GlobalOutputStoreDetection(())
        useful = single.calculate_moe_a4w4_bf16_useful_bytes(
            valid_rows=3072,
            active_experts=96,
            n=6144,
            k=7168,
            output_n=3072,
        )
        self.assertEqual(useful.a_payload_bytes, 11_010_048)
        self.assertEqual(useful.a_scale_bytes, 688_128)
        self.assertEqual(useful.b_payload_bytes, 2_113_929_216)
        self.assertEqual(useful.b_scale_bytes, 132_120_576)
        self.assertEqual(useful.output_payload_bytes, 18_874_368)
        self.assertEqual(useful.read_bytes, 2_257_747_968)
        full_traffic = single.make_logical_traffic(
            read_bytes=useful.read_bytes,
            output_bytes_if_stored=useful.output_payload_bytes,
            store_detection=full,
        )
        loadonly_traffic = single.make_logical_traffic(
            read_bytes=useful.read_bytes,
            output_bytes_if_stored=useful.output_payload_bytes,
            store_detection=loadonly,
        )
        self.assertEqual(full_traffic.write_bytes, 18_874_368)
        self.assertEqual(loadonly_traffic.write_bytes, 0)
        self.assertEqual(
            full_traffic.total_bytes - loadonly_traffic.total_bytes,
            18_874_368,
        )
        # Timing method/context/backend never enter the traffic calculation.
        for _timing, _context, _backend in (
            ("profiler", "standalone", "python"),
            ("cuda-event", "standalone", "python"),
            ("profiler", "moe-pipeline", "python"),
            ("profiler", "moe-pipeline", "cpp"),
        ):
            self.assertEqual(full_traffic.total_bytes, 2_276_622_336)


class MoeCppBackendCliTest(unittest.TestCase):
    def test_single_timing_method_and_inmoe_cli(self):
        parser = single._build_parser()
        default_args = parser.parse_args(["--isa", "kernel.s"])
        event_args = parser.parse_args(
            ["--isa", "kernel.s", "--timing-method", "cuda-event"]
        )
        inmoe_args = parser.parse_args(["--isa", "kernel.s", "--inmoe"])

        self.assertEqual(default_args.timing_method, "profiler")
        self.assertEqual(event_args.timing_method, "cuda-event")
        self.assertFalse(default_args.inmoe)
        self.assertTrue(inmoe_args.inmoe)
        self.assertFalse(default_args.cudagh)
        self.assertTrue(
            parser.parse_args(["--isa", "kernel.s", "--cudagh"]).cudagh
        )
        self.assertFalse(hasattr(default_args, "cpp"))
        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["--isa", "kernel.s", "--timing-method", "invalid"]
            )
        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["--isa", "kernel.s", "--timing-context", "moe-pipeline"]
            )
        with self.assertRaisesRegex(
            single.GemmIsaRunnerError,
            "incompatible with --inmoe",
        ):
            single.validate_timing_method_context("cuda-event", True)
        with self.assertRaisesRegex(
            single.GemmIsaRunnerError,
            "only valid with --timing-method profiler",
        ):
            single.validate_timing_method_context(
                "cuda-event",
                False,
                True,
            )
        single.validate_single_cudagh_backend(False)
        with self.assertRaisesRegex(
            single.GemmIsaRunnerError,
            "empty CUDA Graph",
        ):
            single.validate_single_cudagh_backend(True)

    def test_cpp_flag_default_and_opt_in(self):
        parser = batch._build_parser()
        default_args = parser.parse_args(["--isa", "kernel.s"])
        cpp_args = parser.parse_args(["--isa", "kernel.s", "--cpp"])

        self.assertFalse(default_args.cpp)
        self.assertTrue(cpp_args.cpp)
        batch.validate_cudagh_backend(False, False)
        batch.validate_cudagh_backend(True, True)
        with self.assertRaisesRegex(
            single.GemmIsaRunnerError,
            "--cudagh requires --cpp",
        ):
            batch.validate_cudagh_backend(True, False)

    def test_removed_launch_backend_cli_is_rejected(self):
        parser = batch._build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["--isa", "kernel.s", "--launch-backend", "cpp"]
            )

    def test_inmoe_flag_and_removed_timing_context_cli(self):
        parser = batch._build_parser()
        self.assertFalse(parser.parse_args(["--isa", "kernel.s"]).inmoe)
        self.assertTrue(
            parser.parse_args(["--isa", "kernel.s", "--inmoe"]).inmoe
        )
        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["--isa", "kernel.s", "--timing-context", "moe-pipeline"]
            )

    def test_batch_inherits_single_timing_options_once(self):
        parser = batch._build_parser()
        self.assertEqual(
            sum(action.dest == "timing_method" for action in parser._actions),
            1,
        )
        self.assertEqual(
            sum(action.dest == "inmoe" for action in parser._actions),
            1,
        )
        self.assertEqual(
            sum(action.dest == "cudagh" for action in parser._actions),
            1,
        )


class GenericInmoeContextTest(unittest.TestCase):
    def test_context_accepts_arbitrary_target_and_keeps_random_y1(self):
        from my_code.isa_runner import moe_gemm1_pipeline_launch_compare as compare

        launches = []

        class Scalar:
            def item(self):
                return compare.EXPERTS * compare.TILE_M

        class Pointer:
            def data_ptr(self):
                return 0xBEEF

        class State:
            def __init__(self, _torch, _device):
                self.moe_out = object()
                self.contiguous_m_t = Scalar()
                self.random_y1 = Pointer()

            def assert_external_runner_isolation(self, tensors):
                self.tensors = tensors
                return {
                    "cross_allocation_pairs_checked": len(tensors) * 6,
                    "verified": True,
                }

            def warm(self, _backend, iterations):
                self.warm_iterations = iterations

            def verify_external_kernel_order(self, launch, symbol, backend):
                launch()
                return {
                    "target_symbol": symbol,
                    "backend": backend,
                    "verified": True,
                }

            def random_y1_sha256(self):
                return "stable-random-y1"

            def run_with_external_gemm1(self, launch):
                launch()
                return self.moe_out

            def output_sample_sha256(self):
                return "pipeline-output"

            def close(self):
                self.closed = True

        class Trace:
            def to_dict(self, orient):
                self.orient = orient
                return [
                    {
                        "name": "external_64x256",
                        "device_type": "CUDA",
                    },
                    {
                        "name": "a8w4_tdm_fp4_mock_K3072_e96",
                        "device_type": "CUDA",
                    },
                ]

        state_holder = {}

        def state_factory(*args):
            state_holder["state"] = State(*args)
            return state_holder["state"]

        def raw_launch():
            launches.append("target")
            return object()

        def run_perftest(callable_, **kwargs):
            self.assertEqual(kwargs["warmup"], 0)
            self.assertEqual(kwargs["reported_warmup"], 3)
            output = callable_()
            return (
                output,
                100.0,
                {
                    "name": "external_64x256",
                    "cnt": 7,
                    "warmup": 3,
                    "device_time_sum": 70.0,
                    "device_time_avg": 10.0,
                    "device_type": "CUDA",
                    "device_index": 0,
                    "source": "profiler",
                },
                Trace(),
            )

        def select_timing(_trace, **kwargs):
            self.assertIn("K3072", kwargs["symbol"])
            return {
                "name": kwargs["symbol"],
                "cnt": 7,
                "device_time_avg": 5.0,
            }

        with patch.object(compare, "FixedMoePipeline", side_effect=state_factory):
            timing = timing_context.run_inmoe_context(
                torch_module=object(),
                device=0,
                symbol="external_64x256",
                target_shape=(64, 256, 512),
                target_tensors={"only_one_tensor": object()},
                raw_target_launch=raw_launch,
                profiler_target_launch=raw_launch,
                target_warmup=3,
                iters=7,
                backend_label="python",
                synchronize=lambda: None,
                mark_stream_synchronized=lambda: None,
                run_target_perftest=run_perftest,
                select_target_profiler_timing=select_timing,
                test_graph=False,
                timing_source="pipeline profiler",
            )

        self.assertEqual(timing["device_time_avg"], 10.0)
        self.assertEqual(timing["pipeline_us"], 100.0)
        self.assertTrue(timing["random_y1_unchanged"])
        self.assertEqual(timing["isolation_pairs_checked"], 6)
        # Three target-only warmups, one order proof, one formal iteration.
        self.assertEqual(len(launches), 5)


class MoeCppBackendTargetValidationTest(unittest.TestCase):
    def test_cpp_backend_accepts_only_exact_target_and_symbol(self):
        target = batch.moe_cpp_target_isa()
        batch.validate_cpp_backend_target(
            target,
            batch.MOE_GEMM1_WPT4_KERNEL_SYMBOL,
            "moe-gemm1",
        )

        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / target.name
            copied.write_bytes(target.read_bytes())
            with self.assertRaisesRegex(
                batch.single.GemmIsaRunnerError,
                "restricted to the exact source",
            ):
                batch.validate_cpp_backend_target(
                    copied,
                    batch.MOE_GEMM1_WPT4_KERNEL_SYMBOL,
                    "moe-gemm1",
                )

        with self.assertRaisesRegex(
            batch.single.GemmIsaRunnerError,
            "restricted to the exact source",
        ):
            batch.validate_cpp_backend_target(
                target,
                batch.MOE_GEMM1_KERNEL_SYMBOL,
                "moe-gemm1",
            )


class MoeCppBackendKeyAndCacheTest(unittest.TestCase):
    def test_build_key_is_stable_and_all_content_changes_invalidate(self):
        isa_digest = hashlib.sha256(b"isa-v1").hexdigest()
        cpp_digest = hashlib.sha256(b"cpp-v1").hexdigest()
        base = {
            "isa_sha256": isa_digest,
            "cpp_sha256": cpp_digest,
            "symbol": batch.MOE_GEMM1_WPT4_KERNEL_SYMBOL,
            "arch": "gfx1250",
            "flags": ["-O3"],
        }
        first = backend.make_build_key("test", base)
        self.assertEqual(first, backend.make_build_key("test", dict(base)))

        for field, value in (
            ("isa_sha256", hashlib.sha256(b"isa-v2").hexdigest()),
            ("cpp_sha256", hashlib.sha256(b"cpp-v2").hexdigest()),
            ("symbol", "different_exact_symbol"),
            ("arch", "gfx9999"),
            ("flags", ["-O2"]),
        ):
            changed = dict(base)
            changed[field] = value
            with self.subTest(field=field):
                self.assertNotEqual(
                    first,
                    backend.make_build_key("test", changed),
                )

    def test_co_hash_changes_extension_key(self):
        common = {
            "cpp_sha256": hashlib.sha256(b"fixed cpp").hexdigest(),
            "python_abi": "cpython-312",
            "torch_abi": "torch-rocm",
        }
        one = {**common, "co_sha256": hashlib.sha256(b"co-one").hexdigest()}
        two = {**common, "co_sha256": hashlib.sha256(b"co-two").hexdigest()}
        self.assertNotEqual(
            backend.make_build_key("cpp_extension", one),
            backend.make_build_key("cpp_extension", two),
        )

    def test_real_isa_and_cpp_byte_changes_change_keys(self):
        isa = batch.moe_cpp_target_isa().read_bytes()
        cpp = backend._CPP_SOURCE.read_bytes()
        isa_key = backend.make_build_key(
            "code_object",
            {"isa_sha256": hashlib.sha256(isa).hexdigest()},
        )
        cpp_key = backend.make_build_key(
            "cpp_extension",
            {"cpp_sha256": hashlib.sha256(cpp).hexdigest()},
        )

        self.assertNotEqual(
            isa_key,
            backend.make_build_key(
                "code_object",
                {"isa_sha256": hashlib.sha256(isa + b"\n").hexdigest()},
            ),
        )
        self.assertNotEqual(
            cpp_key,
            backend.make_build_key(
                "cpp_extension",
                {"cpp_sha256": hashlib.sha256(cpp + b"\n").hexdigest()},
            ),
        )

    def test_manifest_and_hashes_control_cache_hit(self):
        with tempfile.TemporaryDirectory() as temporary:
            entry = Path(temporary) / "entry"
            entry.mkdir()
            (entry / "kernel.o").write_bytes(b"object")
            (entry / "kernel.co").write_bytes(b"code object")
            inputs = {"isa_sha256": hashlib.sha256(b"isa").hexdigest()}
            key = backend.make_build_key("code_object", inputs)
            backend._write_cache_manifest(
                entry,
                key=key,
                inputs=inputs,
                artifact_names=("kernel.o", "kernel.co"),
            )

            hit = backend._cache_entry_status(
                entry,
                key,
                ("kernel.o", "kernel.co"),
            )
            self.assertTrue(hit.hit, hit.reason)

            (entry / "kernel.co").write_bytes(b"tampered co")
            miss = backend._cache_entry_status(
                entry,
                key,
                ("kernel.o", "kernel.co"),
            )
            self.assertFalse(miss.hit)
            self.assertRegex(miss.reason, "size changed|sha256 changed")

    def test_rendered_cpp_identity_tracks_both_keys(self):
        template = backend._CPP_SOURCE.read_bytes()
        rendered = backend._render_cpp_source(
            template,
            extension_key="e" * 64,
            code_object_sha256="c" * 64,
        )
        self.assertIn(b"e" * 64, rendered)
        self.assertIn(b"c" * 64, rendered)
        self.assertNotIn(b"@MOE_CPP_EXTENSION_BUILD_KEY@", rendered)
        self.assertNotIn(b"@MOE_CODE_OBJECT_SHA256@", rendered)


class MoeCppBackendTimingTest(unittest.TestCase):
    def test_cudagh_is_forwarded_and_source_is_graph_specific(self):
        class TraceFrame:
            def to_dict(self, orient):
                self.records = [
                    {
                        "name": batch.MOE_GEMM1_WPT4_KERNEL_SYMBOL,
                        "cnt": 100,
                        "device_time_sum": 3200.0,
                        "device_time_avg": 32.0,
                        "device_type": "CUDA",
                        "device_index": "0",
                    }
                ]
                return self.records

            def to_string(self, index=True):
                return repr(self.to_dict("records"))

        output = object()
        run_perftest = Mock(return_value=(output, 32.0, TraceFrame()))
        actual_output, row = batch._run_moe_cpp_perftest(
            run_perftest,
            Mock(),
            symbol=batch.MOE_GEMM1_WPT4_KERNEL_SYMBOL,
            warmup=7,
            iters=100,
            device=0,
            test_graph=True,
        )

        self.assertIs(actual_output, output)
        self.assertTrue(run_perftest.call_args.kwargs["testGraph"])
        self.assertEqual(
            row["source"],
            single.RUN_PERFTEST_GRAPH_TIMING_SOURCE,
        )
        self.assertTrue(row["test_graph"])

    def test_run_perftest_is_formal_summary_source(self):
        class TraceFrame:
            def to_dict(self, orient):
                assert orient == "records"
                self.records = [
                    {
                        "name": batch.MOE_GEMM1_WPT4_KERNEL_SYMBOL,
                        "cnt": 91,
                        "device_time_sum": 3390.25,
                        "device_time_avg": 37.2554945054945,
                        "device_type": "CUDA",
                        "device_index": "0",
                    }
                ]
                return self.records

            def to_string(self, index=True):
                return repr(self.to_dict("records"))

        output = object()
        trace_df = TraceFrame()
        run_perftest = Mock(return_value=(output, 37.25, trace_df))
        launch = Mock()

        actual_output, row = batch._run_moe_cpp_perftest(
            run_perftest,
            launch,
            symbol=batch.MOE_GEMM1_WPT4_KERNEL_SYMBOL,
            warmup=7,
            iters=100,
            device=0,
        )

        self.assertIs(actual_output, output)
        run_perftest.assert_called_once_with(
            launch,
            num_warmup=7,
            num_iters=100,
            testGraph=False,
            return_trace_df=True,
            num_rotate_args=1,
        )
        self.assertEqual(row["name"], batch.MOE_GEMM1_WPT4_KERNEL_SYMBOL)
        self.assertEqual(row["cnt"], 91)
        self.assertEqual(row["device_time_avg"], 37.2554945054945)
        self.assertEqual(row["device_time_sum"], 3390.25)
        self.assertEqual(row["source"], batch.RUN_PERFTEST_TIMING_SOURCE)


class MoeCppPipelineAdapterContractTest(unittest.TestCase):
    def _valid_case(self):
        return {
            "experts": 96,
            "tokens": 512,
            "topk": 6,
            "model_dim": 7168,
            "inter_dim": 3072,
            "data_format": "a4w4",
            "activation": "silu",
            "use_bias": False,
            "expert_balance": True,
            "num_expert_activated": 0,
            "tile_m_override": "64",
            "swiglu_limit": 7.0,
            "situ_beta": 4.0,
            "situ_linear_beta": 25.0,
        }

    def test_pipeline_backend_env_is_explicit_and_strict(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(backend.selected_pipeline_launch_backend())
        with patch.dict(
            os.environ,
            {backend.PIPELINE_LAUNCH_BACKEND_ENV: "cpp"},
            clear=True,
        ):
            self.assertEqual(
                backend.selected_pipeline_launch_backend(),
                backend.PIPELINE_LAUNCH_BACKEND_CPP,
            )
        with patch.dict(
            os.environ,
            {backend.PIPELINE_LAUNCH_BACKEND_ENV: "ctypes"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "must be unset"):
                backend.selected_pipeline_launch_backend()

    def test_pipeline_case_rejects_any_non_target_setting(self):
        valid = self._valid_case()
        backend.validate_pipeline_gemm1_case(**valid)

        for field, value, message in (
            ("experts", 95, "experts"),
            ("tokens", 511, "tokens"),
            ("data_format", "a8w4", "data_format"),
            ("activation", "swiglu", "activation"),
            ("use_bias", True, "use_bias"),
            ("expert_balance", False, "expert_balance"),
            ("tile_m_override", None, "AITER_TDM_TILE_M"),
            ("situ_beta", 1.0, "situ_beta"),
        ):
            changed = dict(valid)
            changed[field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, message):
                    backend.validate_pipeline_gemm1_case(**changed)

    def test_pipeline_kernarg_uses_real_descriptor_aliases(self):
        payload = backend.pack_pipeline_moe_kernargs(
            packer=batch.pack_moe_kernargs,
            ptr_c=0x1000,
            ptr_a=0x2000,
            ptr_b=0x3000,
            ptr_scale_a=0x4000,
            ptr_scale_b=0x5000,
            ptr_m_tile_map=0x6000,
            c_shape=(1, 9216, 3072),
            c_strides=(9216 * 3072, 3072),
            sa_shape=(1, 2304, 224),
            sa_strides=(2304 * 224, 224),
            sb_size0=33_030_144,
            i32_m=9216,
            i32_n=6144,
            swiglu_limit=7.0,
            situ_beta=4.0,
            situ_linear_beta=25.0,
        )

        self.assertEqual(len(payload), batch.MOE_KERNARG_SIZE)
        self.assertEqual(struct.unpack_from("<Q", payload, 120)[0], 0x2000)
        self.assertEqual(struct.unpack_from("<Q", payload, 128)[0], 0x1000)
        self.assertEqual(struct.unpack_from("<I", payload, 72)[0], 224)
        self.assertEqual(struct.unpack_from("<I", payload, 104)[0], 33_030_144)
        self.assertEqual(struct.unpack_from("<I", payload, 140)[0], 9216)
        self.assertEqual(struct.unpack_from("<I", payload, 144)[0], 3072)
        self.assertEqual(struct.unpack_from("<f", payload, 172)[0], 7.0)
        self.assertEqual(struct.unpack_from("<f", payload, 176)[0], 4.0)
        self.assertEqual(struct.unpack_from("<f", payload, 180)[0], 25.0)


if __name__ == "__main__":
    unittest.main()
