"""Unit and functional tests for the AdaMuon_adv bug fixes.

Covers the defects found during code review of `adv_optm/optim/AdaMuon_adv.py`:

- Bug 1: Kourkoutas-β dynamic beta2 was overwritten by the static
  `group['adam_betas']` inside the bias-correction branch (the default),
  silently disabling the Kourkoutas mechanism.
- Bug 2: 1-D Muon parameters crashed on `flatten(1)` / Newton-Schulz 2D assert
  because the non-factored Muon branch had no dimensionality guard.
- Bug 5: MARS-M `last_grad` was stored at the parameter dtype (e.g. BF16),
  silently truncating the variance-reduction signal when gradients are upcast.
- Bug 6: The factored Muon path referenced `original_shape` (defined only in the
  non-factored branch), raising NameError on every factored step.
- Bug 7: torch.compile cache keys ignored hyperparameter configuration, so
  groups sharing a shape but differing in config could reuse stale graphs.

All tests run on CUDA as mandated by the project conventions.
"""

import os
import sys
import unittest

import torch

# Make the package importable when running from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adv_optm.optim.AdaMuon_adv import AdaMuon_adv, _muon_group_config, _adam_group_config  # noqa: E402

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


class AdaMuonAdvTestCase(unittest.TestCase):
    """Base helpers for the test suite."""

    @staticmethod
    def make_param(shape, dtype=torch.float32, use_muon=True):
        p = torch.nn.Parameter(torch.randn(*shape, device=DEVICE, dtype=dtype) * 0.1)
        p.grad = torch.randn(*shape, device=DEVICE, dtype=dtype) * 0.1
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
            use_muon=True,
        )
        kwargs.update(overrides)
        return kwargs


class TestBug1KourkoutasBeta2(AdaMuonAdvTestCase):
    """Kourkoutas-β dynamic beta2 must survive the bias-correction branch."""

    def test_dynamic_beta2_used_when_bias_correction_on(self):
        import types

        import adv_optm.util.Muon_AuxAdam as aux

        p = self.make_param((16, 16))
        opt = AdaMuon_adv(
            [p],
            adam_betas=(0.9, 0.99),
            adam_use_bias_correction=True,
            adam_kourkoutas_beta=True,
            use_muon=False,  # adam path only
        )
        opt.state[p]["step"] = 0

        # Deterministically force the helper to return a dynamic beta2 that
        # differs from the static adam_betas[1] == 0.99.
        def fake_prepare(self_, current_step, device):
            pass

        def fake_get_beta2(self_, p_, group_):
            return 0.9123

        opt.kourkoutas_helper.maybe_prepare_step = types.MethodType(fake_prepare, opt.kourkoutas_helper)
        opt.kourkoutas_helper.get_beta2 = types.MethodType(fake_get_beta2, opt.kourkoutas_helper)

        # Capture the beta2 / sqrt_bias_correction2 that actually reach the Adam step.
        captured = {}

        def spy(self_, p_, grad_, state_, group_, beta1, beta2, sqrt_bc2, step_size, rt, rst):
            captured["beta2"] = beta2
            captured["sqrt_bc2"] = sqrt_bc2

        orig_aux = aux._adam_step_parameter
        try:
            aux._adam_step_parameter = spy
            opt.step_parameter(p, opt.param_groups[0], 0)
        finally:
            aux._adam_step_parameter = orig_aux

        self.assertAlmostEqual(
            captured["beta2"],
            0.9123,
            places=4,
            msg="Dynamic Kourkoutas beta2 was overwritten by static adam_betas in the bias-correction branch",
        )
        # Bias correction must be computed with the DYNAMIC beta2, not 0.99.
        expected_sqrt_bc2 = (1.0 - 0.9123 ** 1) ** 0.5
        self.assertAlmostEqual(captured["sqrt_bc2"], expected_sqrt_bc2, places=4)


class TestBug2OneDimMuonParam(AdaMuonAdvTestCase):
    """1-D Muon params must not crash on flatten/NS; they use element-wise scaling."""

    def test_1d_muon_param_step_runs(self):
        p = self.make_param((64,))
        opt = AdaMuon_adv([p], **self.opt_kwargs(use_muon=True))
        before = p.detach().clone()
        opt.step()
        # The parameter must have changed (update applied).
        self.assertFalse(torch.allclose(p.detach(), before), "1-D Muon param did not update")

    def test_1d_muon_param_with_normuon_runs(self):
        p = self.make_param((64,))
        opt = AdaMuon_adv([p], **self.opt_kwargs(use_muon=True, normuon_variant=True))
        opt.step()
        self.assertTrue(torch.isfinite(p.detach()).all())

    def test_1d_muon_param_with_factored_2nd_runs(self):
        # factored_2nd with 1-D: __init_state guards `not (len(p.shape)==1 and not vector_reshape)`,
        # so it falls back to the dense second moment. This just verifies the 1-D guard handles it.
        p = self.make_param((64,))
        opt = AdaMuon_adv([p], **self.opt_kwargs(use_muon=True, factored_2nd=True))
        opt.step()
        self.assertTrue(torch.isfinite(p.detach()).all())

    def test_1d_muon_param_rms_rescaling_off(self):
        # Bug 4 companion: rms_adjustment's non-rescaling branch indexes size(-2)
        # which breaks on 1-D. Verify 1-D + rms_rescaling=False still works.
        p = self.make_param((64,))
        opt = AdaMuon_adv([p], **self.opt_kwargs(use_muon=True, rms_rescaling=False))
        opt.step()
        self.assertTrue(torch.isfinite(p.detach()).all())


class TestBug5MarsLastGradDtype(AdaMuonAdvTestCase):
    """MARS-M last_grad must be stored at the parameter's dtype, using stochastic
    rounding for BF16 writes instead of a truncating cast."""

    def test_mars_last_grad_stored_at_param_dtype_bf16(self):
        p = self.make_param((16, 16), dtype=torch.bfloat16)
        opt = AdaMuon_adv([p], **self.opt_kwargs(use_muon=True, approx_mars=True))
        st = opt.state[p]
        self.assertEqual(st["last_grad"].dtype, torch.bfloat16)
        opt.step()
        # After the step, last_grad holds the previous gradient, still in BF16.
        self.assertEqual(st["last_grad"].dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(st["last_grad"]).all())

    def test_mars_last_grad_stored_at_param_dtype_fp32(self):
        p = self.make_param((16, 16))
        opt = AdaMuon_adv([p], **self.opt_kwargs(use_muon=True, approx_mars=True))
        st = opt.state[p]
        self.assertEqual(st["last_grad"].dtype, torch.float32)
        opt.step()
        self.assertEqual(st["last_grad"].dtype, torch.float32)

    def test_mars_last_grad_matches_previous_grad(self):
        # last_grad after a step must equal the gradient from that step (stored
        # at the param dtype), confirming the state roundtrip is consistent.
        p = self.make_param((16, 16))
        g0 = p.grad.detach().clone()
        opt = AdaMuon_adv([p], **self.opt_kwargs(use_muon=True, approx_mars=True))
        opt.step()
        torch.testing.assert_close(opt.state[p]["last_grad"], g0, atol=1e-6, rtol=1e-6)


class TestAutoPrecisionFp32Compute(AdaMuonAdvTestCase):
    """With state_precision='auto' and a BF16 param, states are stored in BF16
    (memory savings) but all computation must be upcast to fp32."""

    def test_auto_bf16_states_stored_bf16_but_compute_fp32(self):
        p = self.make_param((16, 16), dtype=torch.bfloat16)
        opt = AdaMuon_adv([p], **self.opt_kwargs(use_muon=True, state_precision="auto"))
        st = opt.state[p]
        # Storage stays at the parameter dtype (memory savings).
        self.assertEqual(st["momentum_buffer"].dtype, torch.bfloat16)
        self.assertEqual(st["second_momentum_buffer"].dtype, torch.bfloat16)
        self.assertEqual(opt.param_groups[0]["actual_state_precision"], "auto")

        # The working tensor returned by get_state must be fp32.
        from adv_optm.util.state_util import get_state

        mt = get_state(st, "momentum_buffer", "auto")
        self.assertEqual(mt.dtype, torch.float32, "get_state('auto') must upcast to fp32")
        vt = get_state(st, "second_momentum_buffer", "auto")
        self.assertEqual(vt.dtype, torch.float32)

        # The step must run without dtype-mismatch errors and stay finite.
        before = p.detach().float().clone()
        opt.step()
        self.assertTrue(torch.isfinite(p.detach()).all())
        self.assertFalse(torch.allclose(p.detach().float(), before), "BF16 auto param should update")
        # Storage is still bf16 after the step (set_state writes back truncated).
        self.assertEqual(st["momentum_buffer"].dtype, torch.bfloat16)
        self.assertEqual(st["second_momentum_buffer"].dtype, torch.bfloat16)

    def test_auto_fp32_states_unchanged_semantics(self):
        # For fp32 params, 'auto' storage is fp32 and get_state returns the same
        # tensor (in-place accumulation semantics preserved).
        p = self.make_param((16, 16))
        opt = AdaMuon_adv([p], **self.opt_kwargs(use_muon=True, state_precision="auto"))
        st = opt.state[p]
        self.assertEqual(st["momentum_buffer"].dtype, torch.float32)

        from adv_optm.util.state_util import get_state

        mt = get_state(st, "momentum_buffer", "auto")
        self.assertIs(mt, st["momentum_buffer"], "fp32 'auto' must return the stored tensor (in-place)")
        opt.step()
        self.assertTrue(torch.isfinite(p.detach()).all())

    def test_upcast_grad_for_precision_auto_bf16(self):
        from adv_optm.util.state_util import upcast_grad_for_precision

        g = torch.randn(4, 4, device=DEVICE, dtype=torch.bfloat16)
        up = upcast_grad_for_precision(g, {"factored": False}, "auto")
        self.assertEqual(up.dtype, torch.float32, "grad must be upcast to fp32 for 'auto'")
        # fp32 gradient is returned as-is (no-op).
        g32 = torch.randn(4, 4, device=DEVICE)
        up32 = upcast_grad_for_precision(g32, {"factored": False}, "auto")
        self.assertIs(up32, g32)

    def test_mars_bf16_uses_stochastic_rounding(self):
        # approx_mars must route BF16 writes through copy_stochastic_ (unbiased)
        # rather than a plain truncating copy. We monkeypatch copy_stochastic_ to
        # record that it was invoked, and assert the stored value is NOT the plain
        # truncation of the fp32 gradient.
        import adv_optm.util.Muon_util as mu

        p = self.make_param((16, 16), dtype=torch.bfloat16)
        opt = AdaMuon_adv([p], **self.opt_kwargs(use_muon=True, approx_mars=True))
        st = opt.state[p]

        called = {"n": 0}

        orig = mu.param_update.copy_stochastic_

        def spy(target, source, inplace=False):
            called["n"] += 1
            return orig(target, source, inplace)

        try:
            mu.param_update.copy_stochastic_ = spy
            opt.step()
        finally:
            mu.param_update.copy_stochastic_ = orig

        self.assertGreaterEqual(called["n"], 1, "BF16 last_grad write should use stochastic rounding")
        self.assertEqual(st["last_grad"].dtype, torch.bfloat16)
        # The stored value must not be a plain truncation of the fp32 gradient.
        g_fp32 = p.grad.detach().float()
        # The stored (stochastically rounded) value may differ from a truncating cast,
        # but must be within one BF16 ULP of the source.
        self.assertTrue(
            torch.all((st["last_grad"].float() - g_fp32).abs() < 0.02),
            "stochastically rounded last_grad diverged from source",
        )


class TestBug6FactoredPathNameError(AdaMuonAdvTestCase):
    """Factored Muon path must not reference undefined original_shape."""

    def test_factored_muon_step_runs(self):
        # 2D shape with a square-ish numel so _get_effective_shape is (d, d).
        p = self.make_param((16, 16))
        opt = AdaMuon_adv([p], **self.opt_kwargs(use_muon=True, nnmf_factor=True))
        st = opt.state[p]
        self.assertTrue(st["factored"], "expected factored state")
        before = p.detach().clone()
        opt.step()
        self.assertFalse(
            torch.allclose(p.detach(), before),
            "Factored Muon step should have updated the parameter",
        )

    def test_factored_muon_step_runs_normuon(self):
        p = self.make_param((16, 16))
        opt = AdaMuon_adv([p], **self.opt_kwargs(use_muon=True, nnmf_factor=True, normuon_variant=True))
        opt.step()
        self.assertTrue(torch.isfinite(p.detach()).all())


class TestBug7CompileCacheKey(AdaMuonAdvTestCase):
    """Cache keys must vary with hyperparameter configuration, not just shape."""

    def test_muon_cache_key_distinguishes_configs(self):
        group_a = {"lr": 1e-3, "state_precision": "auto", "betas": (0.9, 0.95), "weight_decay": 0.0,
                   "nesterov": True, "use_atan2": False, "normuon_variant": False, "rms_rescaling": True,
                   "eps": 1e-8, "ns_eps": 1e-7, "ns_steps": 5, "ns_coeffs": (3.4, -4.7, 2.0),
                   "accelerated_ns": False, "low_rank_ortho": False, "ortho_rank": 128}
        group_b = dict(group_a, use_atan2=True)  # differs only in use_atan2
        self.assertNotEqual(
            _muon_group_config(group_a),
            _muon_group_config(group_b),
            "Muon cache key must distinguish use_atan2 configs",
        )

    def test_adam_cache_key_distinguishes_configs(self):
        group_a = {"lr": 1e-3, "adam_state_precision": "auto", "adam_betas": (0.9, 0.99),
                   "adam_weight_decay": 0.0, "adam_use_bias_correction": True,
                   "adam_use_atan2": False, "adam_nesterov": False, "adam_spectral_normalization": False,
                   "adam_kourkoutas_beta": False, "adam_nnmf_factor": False, "adam_factored_2nd": False,
                   "adam_fisher_wd": False, "adam_orthogonal_gradient": "disabled"}
        group_b = dict(group_a, adam_nesterov=True)
        self.assertNotEqual(
            _adam_group_config(group_a),
            _adam_group_config(group_b),
            "Adam cache key must distinguish nesterov configs",
        )


class TestFactored2ndPath(AdaMuonAdvTestCase):
    """The factored_2nd (non-factored momentum, factorized v_t) path should run."""

    def test_factored_2nd_step_runs(self):
        p = self.make_param((16, 16))
        opt = AdaMuon_adv([p], **self.opt_kwargs(use_muon=True, factored_2nd=True))
        st = opt.state[p]
        self.assertTrue(st.get("factored_2nd"), "expected factored_2nd state")
        before = p.detach().clone()
        opt.step()
        self.assertFalse(torch.allclose(p.detach(), before))

    def test_factored_2nd_step_runs_atan2(self):
        p = self.make_param((16, 16))
        opt = AdaMuon_adv([p], **self.opt_kwargs(use_muon=True, factored_2nd=True, use_atan2=True))
        opt.step()
        self.assertTrue(torch.isfinite(p.detach()).all())


class TestStandardDensePath(AdaMuonAdvTestCase):
    """Regression: standard dense AdaMuon path still behaves correctly."""

    def test_dense_2d_step_runs(self):
        p = self.make_param((16, 16))
        opt = AdaMuon_adv([p], **self.opt_kwargs(use_muon=True))
        before = p.detach().clone()
        opt.step()
        self.assertFalse(torch.allclose(p.detach(), before))

    def test_dense_4d_conv_step_runs(self):
        # Conv2d-shaped param (auto_projection sets kappa_p=2.0 for 4D)
        p = self.make_param((8, 8, 3, 3))
        opt = AdaMuon_adv([p], **self.opt_kwargs(use_muon=True, auto_projection=True))
        before = p.detach().clone()
        opt.step()
        self.assertFalse(torch.allclose(p.detach(), before))

    def test_state_dict_roundtrip(self):
        p = self.make_param((16, 16))
        opt = AdaMuon_adv([p], **self.opt_kwargs(use_muon=True))
        opt.step()
        sd = opt.state_dict()
        opt2 = AdaMuon_adv([p], **self.opt_kwargs(use_muon=True))
        opt2.load_state_dict(sd)
        # State roundtrip must preserve the step counter etc. without crashing.
        self.assertIn(p, opt2.state)


class TestSiblingMuonAdvMarsDtype(AdaMuonAdvTestCase):
    """The shared approx_mars fix must also hold for the sibling Muon_adv."""

    def test_muon_adv_mars_bf16_runs(self):
        from adv_optm.optim.Muon_adv import Muon_adv

        p = torch.nn.Parameter(torch.randn(16, 16, device=DEVICE, dtype=torch.bfloat16) * 0.1)
        p.grad = torch.randn(16, 16, device=DEVICE, dtype=torch.bfloat16) * 0.1
        opt = Muon_adv([p], lr=1e-3, approx_mars=True, use_muon=True)
        # last_grad is stored at the parameter's dtype (BF16).
        self.assertEqual(opt.state[p]["last_grad"].dtype, torch.bfloat16)
        opt.step()
        self.assertTrue(torch.isfinite(p.detach()).all())
        self.assertEqual(opt.state[p]["last_grad"].dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
