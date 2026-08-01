"""Unit and functional tests for the SinkSGD_adv bug fixes.

Covers the defects found during code review of `adv_optm/optim/SinkSGD_adv.py`:

- Defect 1: snr_cond=True with momentum == 0 crashed on `None.rsqrt()` /
  `atan2_(None)` because vt_row/vt_col/denom are only populated when momentum is
  non-zero. Fixed by gating snr_cond at runtime on `momentum > 0` and rejecting
  the invalid config at construction time.
- Defect 2: state_precision='factored' with momentum == 0 raised
  `KeyError: 'effective_shape'` because the factored state was allocated but the
  shape factors were only stored inside the momentum branch.
- Defect 3: the snr_cond validation used `and` instead of `or`, letting invalid
  combinations through.
- Defect 4: the compiled path drew extra stochastic-rounding RNG numbers when
  momentum == 0, diverging from the uncompiled path's deterministic stream.
- Defects 5/8: missing validation for momentum upper bound, nesterov_coef,
  orthogonal_gradient, and sinkhorn_iterations.
- Defect 6: the snr_cond 4/pi compensation was skipped under
  spectral_normalization.
- Defect 7: in-place sign/sinkhorn ops mutated the user's fp32 p.grad buffer.
- Defect 9: is_vector was defined inconsistently between state initialization
  and the step function.
- Defect 10: the compiled path drew ONE stochastic-rounding noise tensor and
  reused it for BOTH the momentum-state SR and the parameter SR (aliased via
  `random_int_state_tensor = random_int_tensor`), while the uncompiled path
  drew two independent tensors (state first, then parameter).  This broke the
  deterministic RNG stream parity between the two paths and correlated the two
  rounding operations. Fixed by drawing state SR before parameter SR and always
  using independent tensors.
- Defect 11: is_vector treated 0-dim (scalar) parameters inconsistently
  (`len(p.shape) == 1` in __init_state vs `grad.ndim < 2` in _step_parameter),
  so a scalar parameter with vector_reshape=True crashed with IndexError inside
  apply_sr_sinkhorn. Scalars are now always treated as vectors.

All tests run on CUDA as mandated by the project conventions.
"""

import os
import sys
import unittest

import torch

# Make the package importable when running from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adv_optm.optim.SinkSGD_adv import SinkSGD_adv  # noqa: E402
from adv_optm.util import param_update  # noqa: E402

DEVICE = torch.device("cuda:0")
torch.manual_seed(0)


def make_param(shape, dtype=torch.float32, **attrs):
    """Builds a Parameter with a random gradient; applies optional attributes."""
    p = torch.nn.Parameter(torch.randn(*shape, device=DEVICE, dtype=dtype) * 0.1)
    p.grad = torch.randn(*shape, device=DEVICE, dtype=dtype) * 0.1
    for key, val in attrs.items():
        setattr(p, key, val)
    return p


def base_kwargs(**overrides):
    kwargs = dict(lr=1e-3, momentum=0.9, weight_decay=0.0)
    kwargs.update(overrides)
    return kwargs


class TestValidation(unittest.TestCase):
    """Constructor validation (defects #3, #5, #8)."""

    def test_momentum_above_one_rejected(self):
        with self.assertRaises(ValueError):
            SinkSGD_adv([make_param((8, 8))], momentum=1.5)

    def test_momentum_negative_rejected(self):
        with self.assertRaises(ValueError):
            SinkSGD_adv([make_param((8, 8))], momentum=-0.1)

    def test_snr_cond_without_normed_momentum_rejected(self):
        # Old logic allowed this (bug: `and not momentum > 0`).
        with self.assertRaises(NotImplementedError):
            SinkSGD_adv([make_param((8, 8))], snr_cond=True, normed_momentum=False, momentum=0.9)

    def test_snr_cond_with_zero_momentum_rejected(self):
        with self.assertRaises(NotImplementedError):
            SinkSGD_adv([make_param((8, 8))], snr_cond=True, normed_momentum=True, momentum=0.0)

    def test_nesterov_coef_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            SinkSGD_adv([make_param((8, 8))], nesterov=True, nesterov_coef=1.5)

    def test_invalid_orthogonal_gradient_rejected(self):
        with self.assertRaises(ValueError):
            SinkSGD_adv([make_param((8, 8))], orthogonal_gradient="flatened")

    def test_negative_sinkhorn_iterations_rejected(self):
        with self.assertRaises(ValueError):
            SinkSGD_adv([make_param((8, 8))], sinkhorn_iterations=-1)

    def test_valid_default_config_constructs(self):
        opt = SinkSGD_adv([make_param((8, 8))], momentum=0.0)
        self.assertEqual(opt.param_groups[0]["momentum"], 0.0)


class TestGradNotMutated(unittest.TestCase):
    """Defect 7: fp32 p.grad must not be modified in place by the step."""

    def test_2d_grad_unchanged(self):
        p = make_param((16, 16))
        grad_before = p.grad.clone()
        opt = SinkSGD_adv([p], **base_kwargs())
        opt.step()
        torch.testing.assert_close(p.grad, grad_before)

    def test_vector_grad_unchanged(self):
        # The 1D branch applies grad.sign_() in place.
        p = make_param((64,))
        grad_before = p.grad.clone()
        opt = SinkSGD_adv([p], **base_kwargs())
        opt.step()
        torch.testing.assert_close(p.grad, grad_before)

    def test_normed_momentum_grad_unchanged(self):
        p = make_param((16, 16))
        grad_before = p.grad.clone()
        opt = SinkSGD_adv([p], **base_kwargs(normed_momentum=True, snr_cond=True))
        opt.step()
        torch.testing.assert_close(p.grad, grad_before)


class TestSnrCondRuntimeGuard(unittest.TestCase):
    """Defect 1: snr_cond must never dereference unpopulated vt/denom."""

    def test_snr_cond_zero_momentum_via_group_mutation_does_not_crash(self):
        # Construct a valid group, then mutate momentum to 0 at runtime, which
        # bypasses the constructor validation. The runtime guard must disable snr.
        p = make_param((16, 16))
        opt = SinkSGD_adv([p], **base_kwargs(normed_momentum=True, snr_cond=True))
        opt.param_groups[0]["momentum"] = 0.0
        opt.step()  # must not raise AttributeError on None
        self.assertEqual(opt.state[p]["step"], 1)

    def test_snr_cond_matrix_path_runs(self):
        p = make_param((16, 16))
        opt = SinkSGD_adv([p], **base_kwargs(normed_momentum=True, snr_cond=True))
        opt.step()
        self.assertEqual(opt.state[p]["step"], 1)

    def test_snr_cond_vector_path_runs(self):
        p = make_param((64,))
        opt = SinkSGD_adv([p], **base_kwargs(normed_momentum=True, snr_cond=True))
        opt.step()
        self.assertEqual(opt.state[p]["step"], 1)


class TestFactoredNoMomentum(unittest.TestCase):
    """Defect 2: factored state + momentum == 0 must not KeyError."""

    def test_factored_zero_momentum_runs(self):
        p = make_param((16, 16))
        opt = SinkSGD_adv([p], state_precision="factored", momentum=0.0)
        opt.step()
        self.assertTrue(opt.state[p]["factored"])
        self.assertIn("effective_shape", opt.state[p])
        self.assertEqual(opt.state[p]["step"], 1)

    def test_nnmf_factor_legacy_zero_momentum_runs(self):
        p = make_param((24, 16))
        opt = SinkSGD_adv([p], nnmf_factor=True, momentum=0.0)
        opt.step()
        self.assertTrue(opt.state[p]["factored"])

    def test_factored_with_momentum_runs(self):
        p = make_param((16, 16))
        opt = SinkSGD_adv([p], state_precision="factored", momentum=0.9)
        opt.step()
        self.assertIn("mu_b_nmf", opt.state[p])
        self.assertIn("mv_b_nmf", opt.state[p])

    def test_factored_vector_reshape_1d_runs(self):
        # Defect 9: init and step must agree on is_vector so a 1D param with
        # vector_reshape=True gets consistent factored/2D handling.
        p = make_param((256,))
        opt = SinkSGD_adv([p], state_precision="factored", vector_reshape=True, momentum=0.9)
        opt.step()
        opt.step()
        self.assertTrue(opt.state[p]["factored"])

    def test_factored_snr_cond_runs(self):
        p = make_param((16, 16))
        opt = SinkSGD_adv([p], state_precision="factored", momentum=0.9,
                          normed_momentum=True, snr_cond=True, nesterov=True)
        opt.step()
        self.assertEqual(opt.state[p]["step"], 1)


class TestBasicBehaviour(unittest.TestCase):
    """Functional sanity checks."""

    def test_update_moves_parameter(self):
        p = make_param((16, 16))
        before = p.detach().clone()
        opt = SinkSGD_adv([p], **base_kwargs())
        opt.step()
        self.assertFalse(torch.equal(p.detach(), before))

    def test_vector_param_runs(self):
        p = make_param((64,))
        opt = SinkSGD_adv([p], **base_kwargs())
        opt.step()
        self.assertEqual(opt.state[p]["step"], 1)

    def test_nesterov_runs(self):
        p = make_param((16, 16))
        opt = SinkSGD_adv([p], **base_kwargs(nesterov=True))
        opt.step()
        self.assertEqual(opt.state[p]["step"], 1)

    def test_nesterov_normed_snr_runs(self):
        p = make_param((16, 16))
        opt = SinkSGD_adv([p], **base_kwargs(nesterov=True, normed_momentum=True, snr_cond=True))
        opt.step()
        self.assertEqual(opt.state[p]["step"], 1)

    def test_normed_momentum_buffer_created(self):
        p = make_param((16, 16))
        opt = SinkSGD_adv([p], **base_kwargs(normed_momentum=True))
        opt.step()
        self.assertIn("momentum_buffer", opt.state[p])

    def test_momentum_zero_still_updates(self):
        p = make_param((16, 16))
        before = p.detach().clone()
        opt = SinkSGD_adv([p], lr=1e-3, momentum=0.0)
        opt.step()
        self.assertFalse(torch.equal(p.detach(), before))
        self.assertNotIn("momentum_buffer", opt.state[p])

    def test_weight_decay_decoupled_factor(self):
        # With a zero gradient the normalized update is exactly 0, so the step
        # reduces to the decoupled decay p <- p * (1 - wd * lr).
        p = make_param((16, 16))
        p.grad.zero_()
        wd, lr = 0.5, 0.1
        expected = p.detach() * (1.0 - wd * lr)
        opt = SinkSGD_adv([p], lr=lr, momentum=0.0, weight_decay=wd)
        opt.step()
        torch.testing.assert_close(p.detach(), expected)

    def test_geometric_wd_matrix_and_vector(self):
        p2d = make_param((16, 16))
        opt = SinkSGD_adv([p2d], **base_kwargs(geometric_wd=True, weight_decay=1e-2))
        opt.step()
        p1d = make_param((64,))
        opt1 = SinkSGD_adv([p1d], **base_kwargs(geometric_wd=True, weight_decay=1e-2))
        opt1.step()
        self.assertEqual(opt.state[p2d]["step"], 1)
        self.assertEqual(opt1.state[p1d]["step"], 1)

    def test_geometric_wd_with_centered_wd_vector(self):
        p1d = make_param((64,))
        opt = SinkSGD_adv([p1d], **base_kwargs(geometric_wd=True, weight_decay=1e-2,
                                               centered_wd=1e-2, centered_wd_mode="full"))
        opt.step()
        self.assertIn("anchor_data", opt.state[p1d])

    def test_centered_wd_anchor_initialized(self):
        p = make_param((16, 16))
        opt = SinkSGD_adv([p], **base_kwargs(centered_wd=1e-2, centered_wd_mode="float8"))
        self.assertIn("anchor_data", opt.state[p])
        opt.step()

    def test_cautious_wd_runs(self):
        p = make_param((16, 16))
        opt = SinkSGD_adv([p], **base_kwargs(weight_decay=1e-2, cautious_wd=True))
        opt.step()

    def test_orthogonal_gradient_modes(self):
        for mode in ("flattened", "iterative"):
            p = make_param((16, 16))
            opt = SinkSGD_adv([p], **base_kwargs(orthogonal_gradient=mode))
            opt.step()
            self.assertEqual(opt.state[p]["step"], 1)

    def test_orthogonal_sinkhorn_runs(self):
        p = make_param((16, 16))
        opt = SinkSGD_adv([p], **base_kwargs(orthogonal_sinkhorn=True))
        opt.step()

    def test_spectral_normalization_runs(self):
        p = make_param((16, 16))
        opt = SinkSGD_adv([p], **base_kwargs(spectral_normalization=True))
        opt.step()
        self.assertIn("spectral_u", opt.state[p])

    def test_spectral_with_snr_cond_runs(self):
        # Defect 6: the 4/pi compensation must apply in the spectral path too.
        p = make_param((16, 16))
        opt = SinkSGD_adv([p], **base_kwargs(spectral_normalization=True,
                                             normed_momentum=True, snr_cond=True))
        opt.step()

    def test_sinkhorn_iterations_zero_runs(self):
        p = make_param((16, 16))
        opt = SinkSGD_adv([p], **base_kwargs(sinkhorn_iterations=0))
        opt.step()


class TestStatePrecisions(unittest.TestCase):
    """Momentum state storage in every supported precision."""

    PRECISIONS = ["auto", "fp32", "bf16_sr", "fp16", "int8_sr", "factored"]

    def test_all_precisions_step(self):
        for precision in self.PRECISIONS:
            with self.subTest(precision=precision):
                p = make_param((16, 16))
                opt = SinkSGD_adv([p], **base_kwargs(state_precision=precision))
                opt.step()
                opt.step()  # a second step exercises the set_state round trip
                self.assertEqual(opt.state[p]["step"], 2)

    def test_int8_sr_scale_state_created(self):
        p = make_param((16, 16))
        opt = SinkSGD_adv([p], **base_kwargs(state_precision="int8_sr"))
        opt.step()
        self.assertIn("momentum_buffer_scale", opt.state[p])

    def test_bf16_param_stochastic_rounding(self):
        p = make_param((16, 16), dtype=torch.bfloat16)
        grad_before = p.grad.clone()
        opt = SinkSGD_adv([p], **base_kwargs())
        opt.step()
        # bf16 grads are upcast (copy) so p.grad must be untouched as well.
        torch.testing.assert_close(p.grad, grad_before)


class TestCompiledPath(unittest.TestCase):
    """torch.compile path including the RNG gating (defect #4)."""

    def test_compiled_basic_step(self):
        p = make_param((16, 16))
        opt = SinkSGD_adv([p], **base_kwargs(compiled_optimizer=True))
        opt.step()
        self.assertEqual(opt.state[p]["step"], 1)

    def test_compiled_bf16_stochastic_rounding(self):
        p = make_param((16, 16), dtype=torch.bfloat16)
        opt = SinkSGD_adv([p], **base_kwargs(compiled_optimizer=True))
        opt.step()
        self.assertEqual(opt.state[p]["step"], 1)

    def test_compiled_zero_momentum_matches_uncompiled(self):
        # Defect 4: compiled and uncompiled paths must consume the same RNG stream
        # when momentum == 0, producing identical parameter updates.
        torch.manual_seed(1234)
        p_comp = make_param((16, 16))
        p_uncomp = p_comp.detach().clone()
        p_uncomp.grad = p_comp.grad.clone()

        torch.manual_seed(1234)
        opt_comp = SinkSGD_adv([p_comp], lr=1e-3, momentum=0.0, compiled_optimizer=True)
        opt_comp.step()

        opt_uncomp = SinkSGD_adv([p_uncomp], lr=1e-3, momentum=0.0)
        opt_uncomp.step()

        torch.testing.assert_close(p_comp.detach(), p_uncomp.detach())

    def test_compiled_bf16_sr_state_precision(self):
        p = make_param((16, 16))
        opt = SinkSGD_adv([p], **base_kwargs(compiled_optimizer=True, state_precision="bf16_sr"))
        opt.step()
        self.assertEqual(opt.state[p]["step"], 1)


class TestCompiledRngStreamParity(unittest.TestCase):
    """Defect 10: compiled and uncompiled paths must consume the deterministic
    SR generator in the same order (state first, parameter second) and with the
    same number of independent draws, producing identical parameter updates."""

    def _run_and_log_draws(self, compiled: bool, state_precision: str, dtype):
        draw_log = []
        orig_draw = param_update._get_random_int_for_sr
        orig_draw8 = param_update._get_random_int_for_8bit_sr

        def spy_sr(source):
            t = orig_draw(source)
            draw_log.append(("sr", tuple(source.shape), int(t.flatten()[0].item())))
            return t

        def spy_8bit(source, numel=None):
            t = orig_draw8(source, numel)
            draw_log.append(("8bit", tuple(t.shape), int(t.flatten()[0].item())))
            return t

        param_update._get_random_int_for_sr = spy_sr
        param_update._get_random_int_for_8bit_sr = spy_8bit
        try:
            torch.manual_seed(1234)
            param_update.set_seed(DEVICE)
            p = make_param((16, 16), dtype=dtype)
            opt = SinkSGD_adv(
                [p],
                lr=1e-3,
                momentum=0.9,
                state_precision=state_precision,
                compiled_optimizer=compiled,
            )
            opt.step()
        finally:
            param_update._get_random_int_for_sr = orig_draw
            param_update._get_random_int_for_8bit_sr = orig_draw8
        return p.detach().clone(), draw_log

    def test_bf16_param_bf16_sr_state(self):
        # Two independent draws (state first, then parameter), identical in both
        # paths, and bitwise-identical parameter updates.
        p_comp, draws_comp = self._run_and_log_draws(True, "bf16_sr", torch.bfloat16)
        p_uncomp, draws_uncomp = self._run_and_log_draws(False, "bf16_sr", torch.bfloat16)
        self.assertEqual(len(draws_comp), 2)
        self.assertEqual(draws_comp, draws_uncomp)
        self.assertTrue(torch.equal(p_comp, p_uncomp))

    def test_bf16_param_int8_sr_state(self):
        p_comp, draws_comp = self._run_and_log_draws(True, "int8_sr", torch.bfloat16)
        p_uncomp, draws_uncomp = self._run_and_log_draws(False, "int8_sr", torch.bfloat16)
        self.assertEqual(len(draws_comp), 2)
        self.assertEqual(draws_comp, draws_uncomp)
        self.assertTrue(torch.equal(p_comp, p_uncomp))

    def test_fp32_param_bf16_sr_state(self):
        # Only the state needs SR noise here; both paths draw exactly once.
        p_comp, draws_comp = self._run_and_log_draws(True, "bf16_sr", torch.float32)
        p_uncomp, draws_uncomp = self._run_and_log_draws(False, "bf16_sr", torch.float32)
        self.assertEqual(len(draws_comp), 1)
        self.assertEqual(draws_comp, draws_uncomp)


class TestScalarParams(unittest.TestCase):
    """Defect 11: 0-dim (scalar) parameters must be handled consistently."""

    def test_scalar_vector_reshape_runs(self):
        p = torch.nn.Parameter(torch.tensor(0.1, device=DEVICE))
        p.grad = torch.tensor(0.05, device=DEVICE)
        opt = SinkSGD_adv([p], lr=1e-3, momentum=0.9, vector_reshape=True,
                          state_precision="factored")
        opt.step()
        self.assertEqual(opt.state[p]["step"], 1)

    def test_scalar_default_runs(self):
        p = torch.nn.Parameter(torch.tensor(0.1, device=DEVICE))
        p.grad = torch.tensor(0.05, device=DEVICE)
        opt = SinkSGD_adv([p], lr=1e-3, momentum=0.9)
        opt.step()
        opt.step()
        self.assertEqual(opt.state[p]["step"], 2)


class TestStateDict(unittest.TestCase):
    """Save/load round trip."""

    def test_state_dict_round_trip(self):
        p = make_param((16, 16))
        opt = SinkSGD_adv([p], **base_kwargs(state_precision="factored"))
        opt.step()
        sd = opt.state_dict()

        p2 = torch.nn.Parameter(p.detach().clone())
        p2.grad = p.grad.clone()
        opt2 = SinkSGD_adv([p2], **base_kwargs(state_precision="factored"))
        opt2.load_state_dict(sd)
        opt2.step()
        self.assertEqual(opt2.state[p2]["step"], 2)


if __name__ == "__main__":
    unittest.main()
