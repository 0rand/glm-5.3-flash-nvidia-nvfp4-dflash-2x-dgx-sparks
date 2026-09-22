import importlib
import io
import json
import os
import pathlib
import sys
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout

ROOT = pathlib.Path(__file__).resolve().parents[2]
GLM = ROOT
sys.path.insert(0, str(GLM))


# Top-level fixture so inspect.getsource() can recover compilable source.
torch = types.SimpleNamespace(
    int8="int8", zeros=lambda *args, **kwargs: "ordinary"
)


def fixture_allocate(size, dtype, device):
    buf_size = size
    buf = torch.zeros(buf_size, dtype=torch.int8, device=device)
    return buf


class DisplayKvGlmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = importlib.import_module("display_kv_glm")

    def setUp(self):
        self.m._reset_for_tests()
        os.environ["DISPLAY_KV_EXPECTED_TOTAL_BYTES"] = "13690208256"
        os.environ.pop("DISPLAY_KV_MIN_TOTAL_BYTES", None)
        os.environ.pop("DISPLAY_KV_MAX_ORDINARY_BYTES", None)

    def tearDown(self):
        self.m._reset_for_tests()
        for key in (
            "DISPLAY_KV_EXPECTED_TOTAL_BYTES",
            "DISPLAY_KV_MIN_TOTAL_BYTES",
            "DISPLAY_KV_MAX_ORDINARY_BYTES",
        ):
            os.environ.pop(key, None)

    def test_pool_layout_adds_display_suffix_without_extra_ordinary_ram(self):
        base = 11 * 1024**3
        total = base + self.m.DISPLAY_BYTES
        layout = self.m.pool_layout(total)
        self.assertEqual(layout.ordinary_bytes, base)
        self.assertEqual(layout.display_bytes, self.m.DISPLAY_BYTES)
        self.assertEqual(layout.total_bytes, total)
        self.assertEqual(layout.backing_bytes, total)
        self.assertEqual(layout.padding_bytes, 0)

    def test_pool_layout_rejects_request_that_cannot_include_display_suffix(self):
        with self.assertRaisesRegex(ValueError, "larger than display reserve"):
            self.m.pool_layout(self.m.DISPLAY_BYTES)

    def test_runtime_guards_preserve_11gb_ordinary_pin_with_planner_alignment(self):
        os.environ.pop("DISPLAY_KV_EXPECTED_TOTAL_BYTES", None)
        os.environ["DISPLAY_KV_MIN_TOTAL_BYTES"] = "10240000001"
        os.environ["DISPLAY_KV_MAX_ORDINARY_BYTES"] = "10240000000"

        exact_budget = self.m.pool_layout(12_119_048_192)
        self.m._validate_layout(exact_budget)
        self.assertEqual(exact_budget.ordinary_bytes, 10_240_000_000)

        # Planner alignment may request fewer, non-64-KiB bytes. Round only the
        # owned backing up and keep the exact vLLM-visible view length.
        shortfall = 12_345
        unaligned = self.m.pool_layout(12_119_048_192 - shortfall)
        self.m._validate_layout(unaligned)
        self.assertEqual(unaligned.ordinary_bytes, 10_240_000_000)
        self.assertEqual(unaligned.padding_bytes, shortfall)
        self.assertEqual(unaligned.backing_bytes, 12_119_048_192)

        aligned = self.m.pool_layout(12_119_048_192 - self.m.QUANTUM)
        self.m._validate_layout(aligned)
        self.assertLess(aligned.ordinary_bytes, 10_240_000_000)

        with self.assertRaisesRegex(RuntimeError, "ordinary-RAM guard"):
            self.m._validate_layout(
                self.m.pool_layout(12_119_048_192 + self.m.QUANTUM)
            )

    def test_diagnostics_never_pollute_stdout(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            self.m._emit({"stage": "test"})
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(json.loads(stderr.getvalue()), {"stage": "test"})

    def test_compile_allocator_replaces_exactly_one_torch_zeros(self):
        calls = []

        def replacement(size, dtype, device):
            calls.append((size, dtype, device))
            return "external"

        patched = self.m.compile_allocator(fixture_allocate, replacement)
        self.assertEqual(patched(123, "ignored", "cuda:0"), "external")
        self.assertEqual(calls, [(123, "int8", "cuda:0")])

    def test_compile_allocator_fails_closed_when_source_drifts(self):
        def drifted(size, dtype, device):
            return size

        with self.assertRaisesRegex(RuntimeError, "exactly once"):
            self.m.compile_allocator(drifted, lambda *a: None)

    def test_real_allocation_sets_expected_size_only_inside_context(self):
        cfg = types.SimpleNamespace(
            kv_cache_tensors=[types.SimpleNamespace(size=13_690_208_256)]
        )
        self.assertIsNone(self.m.expected_allocation_size())
        with self.m.real_allocation(cfg):
            self.assertEqual(self.m.expected_allocation_size(), 13_690_208_256)
            # Simulate the allocator creating its process-lifetime owner.
            self.m._owner = object()
        self.assertIsNone(self.m.expected_allocation_size())

    def test_wrap_worker_initialization_activates_real_allocation_context(self):
        seen = []

        class Worker:
            def initialize_from_config(self, cfg):
                seen.append(self_module.expected_allocation_size())
                # Simulate allocate_kv_cache creating the owner.
                self_module._owner = object()
                return "ok"

        self_module = self.m
        original = Worker.initialize_from_config
        Worker.initialize_from_config = self.m.wrap_initialize_from_config(original)
        cfg = types.SimpleNamespace(
            kv_cache_tensors=[types.SimpleNamespace(size=13_690_208_256)]
        )
        self.assertEqual(Worker().initialize_from_config(cfg), "ok")
        self.assertEqual(seen, [13_690_208_256])
        self.assertIsNone(self.m.expected_allocation_size())


if __name__ == "__main__":
    unittest.main(verbosity=2)
