#!/usr/bin/env python3
"""Exercise the actual guarded dispatch block without importing a GPU runtime."""

import ast
import json
import os
from pathlib import Path
from unittest.mock import patch


def main():
    path = Path(__file__).resolve().parents[1] / "aiter/ops/flydsl/grouped_gemm_mxfp4.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    branch = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                  and node.test.id == "use_fused_persistent"
                  and any(isinstance(n, ast.Name) and n.id == "tuned_nonbalanced"
                          for n in ast.walk(node)))
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name in ("_select_bool_env", "_select_tdm_b_th")]
    code = compile(ast.fix_missing_locations(ast.Module(body=helpers + branch.body, type_ignores=[])),
                   str(path), "exec")

    def dispatch(**changes):
        state = dict(tile_m=256, n_experts=65, contiguous_m=30464,
                     partial_m_direct_scale=0, auto_balanced_m192=0, balanced_m192=-1,
                     _balanced_expert_routing_enabled=lambda: False,
                     persistent_kwargs={}, tdm_b_th=0, os=os)
        state.update(changes)
        exec(code, state)
        return state["persistent_kwargs"]

    keys = ("persistent_n_tasks", "staged_quant_scale", "row_tail_predication",
            "half_a_prefetch", "half_scale_prefetch", "row_tail_a_groups",
            "pair_a_prefetch")
    checks = []
    with patch.dict(os.environ, {}, clear=True):
        assert tuple(dispatch()[k] for k in keys) == (1, 1, 3, 1, 1, 2, 1)
        assert dispatch()["task_m_order"] == 1
        assert dispatch()["task_m_group"] == 8
        assert dispatch()["runtime_n_experts"] == 65
        checks.append("E65 target defaults")
        e64 = dispatch(n_experts=64, contiguous_m=28672, tdm_b_th=6)
        assert tuple(e64[k] for k in keys) == (1, 1, 3, 1, 1, 2, 1)
        assert e64["task_m_order"] == 1 and e64["task_m_group"] == 8
        assert e64["runtime_n_experts"] == 64
        checks.append("E64 target shares the runtime-count schedule")
        for changes in (dict(n_experts=64), dict(n_experts=66), dict(tile_m=192),
                        dict(contiguous_m=30208), dict(contiguous_m=30720),
                        dict(contiguous_m=57856), dict(partial_m_direct_scale=1),
                        dict(auto_balanced_m192=1), dict(balanced_m192=1),
                        dict(_balanced_expert_routing_enabled=lambda: True)):
            assert tuple(dispatch(**changes)[k] for k in keys) == (4, 0, 0, 0, 0, 0, 0), changes
            assert dispatch(**changes)["task_m_order"] == 0, changes
            assert dispatch(**changes)["task_m_group"] == 16, changes
            assert dispatch(**changes)["runtime_n_experts"] == 0, changes
            checks.append(changes.keys().__iter__().__next__())
    with patch.dict(os.environ, {"AITER_GFX1250_GEMM1_N_TASKS": "4"}, clear=True):
        assert tuple(dispatch()[k] for k in keys) == (4, 0, 0, 0, 0, 0, 0)
        assert dispatch()["task_m_order"] == 0
        assert dispatch()["task_m_group"] == 16
        assert dispatch()["runtime_n_experts"] == 0
        checks.append("explicit original fallback")
    with patch.dict(os.environ, {"AITER_GFX1250_GEMM1_ROW_TAIL_PREDICATION": "1",
                               "AITER_GFX1250_GEMM1_HALF_A_PREFETCH": "0"}, clear=True):
        assert tuple(dispatch()[k] for k in keys) == (1, 1, 1, 0, 0, 0, 0)
        assert dispatch()["task_m_order"] == 0
        assert dispatch()["task_m_group"] == 16
        checks.append("explicit rowtail/ScaleA control")
    for env, expected in (
        ({"AITER_GFX1250_GEMM1_HALF_A_PREFETCH": "0"}, (1, 1, 3, 0, 0, 0, 0)),
        ({"AITER_GFX1250_GEMM1_ROW_TAIL_PREDICATION": "1"}, (1, 1, 1, 1, 0, 0, 0)),
        ({"AITER_GFX1250_GEMM1_ROW_TAIL_PREDICATION": "0"}, (1, 1, 0, 1, 0, 0, 0)),
        ({"AITER_GFX1250_GEMM1_HALF_SCALE_PREFETCH": "0"}, (1, 1, 3, 1, 0, 0, 0)),
        ({"AITER_GFX1250_GEMM1_ROW_TAIL_A_GROUPS": "4",
          "AITER_GFX1250_GEMM1_PAIR_A_PREFETCH": "0"}, (1, 1, 3, 1, 1, 4, 0)),
        ({"AITER_GFX1250_GEMM1_HALF_SCALE_PREFETCH": "0",
          "AITER_GFX1250_GEMM1_ROW_TAIL_A_GROUPS": "0",
          "AITER_GFX1250_GEMM1_PAIR_A_PREFETCH": "0"}, (1, 1, 3, 1, 0, 0, 0)),
    ):
        with patch.dict(os.environ, env, clear=True):
            assert tuple(dispatch()[k] for k in keys) == expected, env
            assert dispatch()["task_m_order"] == int(expected[2] == 3), env
            assert dispatch()["task_m_group"] == (8 if expected[2] == 3 else 16), env
            checks.append("override dependencies: " + ",".join(env))
    with patch.dict(os.environ, {"AITER_GFX1250_GEMM1_TASK_M_ORDER": "0"}, clear=True):
        assert tuple(dispatch()[k] for k in keys) == (1, 1, 3, 1, 1, 2, 1)
        assert dispatch()["task_m_order"] == 0
        assert dispatch()["task_m_group"] == 16
        checks.append("explicit original M-task order")
    with patch.dict(os.environ, {"AITER_GFX1250_GEMM1_TASK_M_GROUP": "16"}, clear=True):
        assert dispatch()["task_m_group"] == 16
        checks.append("explicit original M swizzle group")
    for name, value in (("AITER_GFX1250_GEMM1_UNIFIED_EXPERTS", "0"),
                        ("AITER_TDM_B_TH", "6")):
        with patch.dict(os.environ, {name: value}, clear=True):
            assert dispatch()["runtime_n_experts"] == 0
            e64 = dispatch(n_experts=64, contiguous_m=28672, tdm_b_th=6)
            assert e64["runtime_n_experts"] == 0
            assert tuple(e64[k] for k in keys) == (4, 0, 0, 0, 0, 0, 0)
            checks.append("explicit legacy selector: " + name)
    print(json.dumps({"checks": checks, "passed": True, "gpu_dispatch": False}, indent=2))


if __name__ == "__main__":
    main()
