"""Regression tests: AdaMuon_adv must not crash on BF16 params with FP32 grads.

Reproduces the reported crash:

    RuntimeError: expected dtype struct c10::BFloat16 for `end` but got dtype float
        adv_optm/util/Muon_AuxAdam.py, _adam_step_parameter
        exp_avg.lerp_(grad, 1.0 - beta1_adam)

Root cause (fixed in commits 8a68555 / cbbfe37):
- `get_state()` returned the *stored* tensor as-is for 'auto'/'fp32', so with a
  BF16 parameter the momentum accumulator `exp_avg` was BF16.
- `upcast_grad_for_precision()` did not upcast 'auto' gradients, so with FP32
  gradients (mixed precision, as in SD/LoRA training) `lerp_` received a BF16
  `self` and an FP32 `end`.

The current source upcasts both the state working tensor and the gradient to
fp32. These tests assert the full optimizer step runs under the exact
mixed-precision configuration that used to crash: the AuxAdam (Adam) path, the
Muon path, mixed param groups, and the torch.compile variants.

Compiled-path workaround: torch._dynamo cannot trace a graph input whose .grad
dtype differs from its own, so step_parameter passes a storage-sharing detached
view to the compiled step and apply_parameter_update receives `state`
explicitly (see TestCompiledBf16Fp32Grad). The sibling Muon_adv optimizer
shares the same fix (TestSiblingMuonAdvCompiled).
"""

import os
import sys
import unittest

import torch

# Make the package importable when running from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adv_optm.optim.AdaMuon_adv import AdaMuon_adv  # noqa: E402
from adv_optm.util.state_util import get_state, upcast_grad_for_precision  # noqa: E402

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# The compiled tests exercise many distinct input signatures of the shared
# torch._dynamo step wrapper (torch/_dynamo/external_utils.py `inner`) in one
# process. Dynamo's default per-frame recompile cache (8) is too small and
# fullgraph=True escalates the overflow to FailOnRecompileLimitHit. Raising the
# cache limit is torch's documented remedy and mirrors what real compiled
# training setups with many layer shapes require.
torch._dynamo.config.cache_size_limit = 64


class AdaMuonBf16Fp32GradTestCase(unittest.TestCase):
    """Base helpers for the mixed-precision regression suite."""

    @staticmethod
    def make_param(shape, use_muon, grad_dtype=torch.float32):
        """BF16 parameter with a configurable gradient dtype.

        Default (grad_dtype=torch.float32) reproduces the mixed-precision setup
        from the reported crash. PyTorch forbids assigning an FP32 grad directly
        to a BF16 leaf, so we mimic the production mechanism: the gradient is
        produced in FP32 while the parameter *data* is cast to BF16 (e.g. weights
        cast after backward in a full-BF16 / mixed-precision setup).
        """
        p = torch.nn.Parameter(torch.randn(*shape, device=DEVICE, dtype=torch.float32) * 0.1)
        p.grad = torch.randn(*shape, device=DEVICE, dtype=torch.float32) * 0.1
        p.data = p.data.to(torch.bfloat16)
        if grad_dtype != torch.float32:
            p.grad = p.grad.to(grad_dtype)
        assert p.dtype == torch.bfloat16 and p.grad.dtype == grad_dtype
        if use_muon:
            p._is_muon = True
        return p

    @staticmethod
    def opt_kwargs(**overrides):
        kwargs = dict(
            lr=1e-3,
            betas=(0.95, 0.95),
            weight_decay=0.0,
            rms_rescaling=True,
            ns_steps=5,
            stochastic_rounding=True,
        )
        kwargs.update(overrides)
        return kwargs


class TestAdamPathBf16Fp32Grad(AdaMuonBf16Fp32GradTestCase):
    """The exact traceback scenario: AuxAdam path (use_muon=False)."""

    def test_2d_bf16_param_fp32_grad_runs(self):
        # 2D tensor forced onto the Adam path, adam_state_precision='auto'.
        p = self.make_param((16, 16), use_muon=False)
        grad_before = p.grad.clone()
        opt = AdaMuon_adv([p], **self.opt_kwargs(use_muon=False, adam_state_precision="auto"))
        before = p.detach().float().clone()

        opt.step()
        opt.step()  # second step exercises the set_state round-trip

        self.assertTrue(torch.isfinite(p.detach()).all())
        self.assertFalse(torch.allclose(p.detach().float(), before), "BF16 Adam param should update")
        # p.grad must not be mutated by the step.
        torch.testing.assert_close(p.grad, grad_before)

    def test_1d_bf16_param_fp32_grad_runs(self):
        # 1D tensor (e.g. a bias) forced onto the Adam path.
        p = self.make_param((64,), use_muon=False)
        opt = AdaMuon_adv([p], **self.opt_kwargs(use_muon=False, adam_state_precision="auto"))
        opt.step()
        opt.step()
        self.assertTrue(torch.isfinite(p.detach()).all())

    def test_adam_states_stored_bf16_compute_fp32(self):
        # Storage stays BF16 (memory savings) but the working tensors are fp32.
        p = self.make_param((16, 16), use_muon=False)
        opt = AdaMuon_adv([p], **self.opt_kwargs(use_muon=False, adam_state_precision="auto"))
        st = opt.state[p]
        self.assertEqual(st["exp_avg"].dtype, torch.bfloat16)
        self.assertEqual(st["exp_avg_sq"].dtype, torch.bfloat16)
        self.assertEqual(opt.param_groups[0]["adam_actual_state_precision"], "auto")

        exp_avg = get_state(st, "exp_avg", "auto")
        self.assertEqual(exp_avg.dtype, torch.float32, "get_state('auto') must upcast to fp32")
        grad = upcast_grad_for_precision(p.grad, st, "auto")
        self.assertEqual(grad.dtype, torch.float32, "'auto' grad must be upcast to fp32")

        # Guard the exact crash site: lerp_ must receive two fp32 operands.
        exp_avg.lerp_(grad, 0.1)
        self.assertEqual(exp_avg.dtype, torch.float32)

        opt.step()
        self.assertTrue(torch.isfinite(p.detach()).all())
        # Storage remains BF16 after the set_state round-trip.
        self.assertEqual(st["exp_avg"].dtype, torch.bfloat16)
        self.assertEqual(st["exp_avg_sq"].dtype, torch.bfloat16)


class TestMuonPathBf16Fp32Grad(AdaMuonBf16Fp32GradTestCase):
    """Companion scenario: Muon path (use_muon=True) with the same mixed dtypes."""

    def test_2d_bf16_param_fp32_grad_runs(self):
        p = self.make_param((16, 16), use_muon=True)
        grad_before = p.grad.clone()
        opt = AdaMuon_adv([p], **self.opt_kwargs(use_muon=True, state_precision="auto"))
        before = p.detach().float().clone()

        opt.step()
        opt.step()

        self.assertTrue(torch.isfinite(p.detach()).all())
        self.assertFalse(torch.allclose(p.detach().float(), before), "BF16 Muon param should update")
        torch.testing.assert_close(p.grad, grad_before)

    def test_1d_bf16_param_fp32_grad_runs(self):
        p = self.make_param((64,), use_muon=True)
        opt = AdaMuon_adv([p], **self.opt_kwargs(use_muon=True, state_precision="auto"))
        opt.step()
        opt.step()
        self.assertTrue(torch.isfinite(p.detach()).all())


class TestMixedGroupsBf16Fp32Grad(AdaMuonBf16Fp32GradTestCase):
    """Realistic SD/LoRA layout: 2D matrices on Muon, 1D tensors on Adam."""

    def test_mixed_groups_run(self):
        p_muon = self.make_param((32, 32), use_muon=True)
        p_adam = self.make_param((32,), use_muon=False)

        opt = AdaMuon_adv(
            [
                {"params": [p_muon], "use_muon": True},
                {"params": [p_adam], "use_muon": False},
            ],
            **self.opt_kwargs(adam_state_precision="auto", state_precision="auto"),
        )
        self.assertTrue(opt.state[p_muon]["is_muon"])
        self.assertFalse(opt.state[p_adam]["is_muon"])

        opt.step()
        opt.step()
        self.assertTrue(torch.isfinite(p_muon.detach()).all())
        self.assertTrue(torch.isfinite(p_adam.detach()).all())
        self.assertEqual(opt.state[p_muon]["momentum_buffer"].dtype, torch.bfloat16)
        self.assertEqual(opt.state[p_adam]["exp_avg"].dtype, torch.bfloat16)


class TestCompiledBf16Fp32Grad(AdaMuonBf16Fp32GradTestCase):
    """Compiled-path behavior under mixed precision (validated on CUDA).

    torch._dynamo refuses to trace a graph input tensor whose .grad dtype
    differs from its own dtype ("Inconsistent dtype between tensor and its
    gradient", an FSDP-related fake-tensor check in torch/_dynamo/variables/
    builder.py; a compiled vanilla AdamW hits the same wall). The optimizers
    work around it at the call boundary in step_parameter:
      - pass a storage-sharing detached view (no .grad attribute) as the
        compiled step's parameter input when grad dtype != param dtype;
      - apply_parameter_update receives `state` explicitly so the in-graph
        self.state[p] lookup (which would re-trigger the check by wrapping the
        mismatched key) is eliminated.

    These tests cover both the matching-dtype runs (normal autocast) and the
    previously-broken mismatched-dtype runs (the reported crash config).
    """

    def test_compiled_adam_bf16_matching_dtype_runs(self):
        # Normal autocast: BF16 param + BF16 grad through the compiled AuxAdam path.
        p = self.make_param((16, 16), use_muon=False, grad_dtype=torch.bfloat16)
        opt = AdaMuon_adv(
            [p],
            **self.opt_kwargs(use_muon=False, adam_state_precision="auto", compiled_optimizer=True),
        )
        opt.step()
        opt.step()
        self.assertTrue(torch.isfinite(p.detach()).all())
        self.assertEqual(opt.state[p]["exp_avg"].dtype, torch.bfloat16)

    def test_compiled_muon_bf16_matching_dtype_runs(self):
        p = self.make_param((16, 16), use_muon=True, grad_dtype=torch.bfloat16)
        opt = AdaMuon_adv(
            [p],
            **self.opt_kwargs(use_muon=True, state_precision="auto", compiled_optimizer=True),
        )
        opt.step()
        opt.step()
        self.assertTrue(torch.isfinite(p.detach()).all())

    def test_compiled_bf16_sr_matching_dtype_runs(self):
        # bf16_sr state precision (stochastic rounding) + compiled + BF16.
        p = self.make_param((16, 16), use_muon=True, grad_dtype=torch.bfloat16)
        opt = AdaMuon_adv(
            [p],
            **self.opt_kwargs(use_muon=True, state_precision="bf16_sr", compiled_optimizer=True),
        )
        opt.step()
        opt.step()
        self.assertTrue(torch.isfinite(p.detach()).all())

    def test_compiled_int8_sr_matching_dtype_runs(self):
        # int8_sr quantized states (LoRA-style VRAM saving) + compiled + BF16.
        p = self.make_param((16, 16), use_muon=True, grad_dtype=torch.bfloat16)
        opt = AdaMuon_adv(
            [p],
            **self.opt_kwargs(use_muon=True, state_precision="int8_sr", compiled_optimizer=True),
        )
        opt.step()
        opt.step()
        self.assertTrue(torch.isfinite(p.detach()).all())

    def test_compiled_1d_adam_bf16_matching_dtype_runs(self):
        p = self.make_param((64,), use_muon=False, grad_dtype=torch.bfloat16)
        opt = AdaMuon_adv(
            [p],
            **self.opt_kwargs(use_muon=False, adam_state_precision="auto", compiled_optimizer=True),
        )
        opt.step()
        opt.step()
        self.assertTrue(torch.isfinite(p.detach()).all())

    def test_compiled_adam_mismatched_dtype_runs(self):
        # The reported crash config (BF16 param + FP32 grad) through the compiled
        # AuxAdam path must now run thanks to the detached-input workaround.
        p = self.make_param((16, 16), use_muon=False)
        grad_before = p.grad.clone()
        opt = AdaMuon_adv(
            [p],
            **self.opt_kwargs(use_muon=False, adam_state_precision="auto", compiled_optimizer=True),
        )
        before = p.detach().float().clone()
        opt.step()
        opt.step()
        self.assertTrue(torch.isfinite(p.detach()).all())
        self.assertFalse(torch.allclose(p.detach().float(), before), "BF16 Adam param should update")
        torch.testing.assert_close(p.grad, grad_before, msg="p.grad must be untouched")
        self.assertEqual(opt.state[p]["exp_avg"].dtype, torch.bfloat16)

    def test_compiled_muon_mismatched_dtype_runs(self):
        p = self.make_param((16, 16), use_muon=True)
        grad_before = p.grad.clone()
        opt = AdaMuon_adv(
            [p],
            **self.opt_kwargs(use_muon=True, state_precision="auto", compiled_optimizer=True),
        )
        before = p.detach().float().clone()
        opt.step()
        opt.step()
        self.assertTrue(torch.isfinite(p.detach()).all())
        self.assertFalse(torch.allclose(p.detach().float(), before), "BF16 Muon param should update")
        torch.testing.assert_close(p.grad, grad_before, msg="p.grad must be untouched")
        self.assertEqual(opt.state[p]["momentum_buffer"].dtype, torch.bfloat16)


class TestSiblingMuonAdvCompiled(AdaMuonBf16Fp32GradTestCase):
    """Muon_adv shares the Muon_AuxAdam / apply_parameter_update fix; the
    compiled + mismatched-dtype scenario must run there as well."""

    def test_compiled_muon_adv_mismatched_dtype_runs(self):
        from adv_optm.optim.Muon_adv import Muon_adv

        p = self.make_param((16, 16), use_muon=True)
        grad_before = p.grad.clone()
        opt = Muon_adv([p], lr=1e-3, use_muon=True, compiled_optimizer=True)
        before = p.detach().float().clone()
        opt.step()
        opt.step()
        self.assertTrue(torch.isfinite(p.detach()).all())
        self.assertFalse(torch.allclose(p.detach().float(), before), "BF16 Muon_adv param should update")
        torch.testing.assert_close(p.grad, grad_before, msg="p.grad must be untouched")


if __name__ == "__main__":
    unittest.main(verbosity=2)
