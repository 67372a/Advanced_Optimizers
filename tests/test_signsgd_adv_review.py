"""Review tests for SignSGD_adv.

Covers the defects found during the SignSGD_adv code review:

1. ``snr_cond`` validation used AND logic instead of OR, silently disabling SNR
   preconditioning for invalid combinations instead of raising.
2. The ``torch.compile`` path reused the SAME random tensor for both the
   parameter and the bf16_sr state stochastic rounding (correlated rounding +
   divergent RNG stream vs the uncompiled path).
3. The spectral-normalization path dropped the 4/pi SNR compensation factor
   that the non-spectral path applies.
4. ``is_vector`` was defined inconsistently between state allocation
   (``__init_state``) and the runtime step (``step_parameter`` /
   ``_step_parameter``), allocating factored state for 0-dim scalars and
   treating vector_reshape'd 1D params as plain vectors at step time.
5. Invalid ``orthogonal_gradient`` / out-of-range ``nesterov_coef`` were not
   validated, leading to confusing failures inside the step.

All tests run on CUDA as mandated by the project conventions.
"""

import math
import os
import sys
import unittest

import torch

# Make the package importable when running from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adv_optm.optim import SignSGD_adv  # noqa: E402
from adv_optm.util import param_update  # noqa: E402

DEVICE = torch.device("cuda:0")
torch.manual_seed(0)


def make_param(shape=(16, 16), dtype=torch.float32, scale=0.1):
    return torch.nn.Parameter(torch.randn(*shape, device=DEVICE, dtype=dtype) * scale)


def run_steps(opt, p, grads):
    """Runs a sequence of pre-built gradient tensors through the optimizer."""
    for g in grads:
        p.grad = g.clone()
        opt.step()


class TestValidation(unittest.TestCase):
    def test_snr_cond_requires_normed_momentum(self):
        with self.assertRaises(NotImplementedError):
            SignSGD_adv([make_param()], lr=1e-3, momentum=0.9,
                        normed_momentum=False, snr_cond=True)

    def test_snr_cond_requires_momentum_gt_zero(self):
        with self.assertRaises(NotImplementedError):
            SignSGD_adv([make_param()], lr=1e-3, momentum=0.0,
                        normed_momentum=True, snr_cond=True)

    def test_snr_cond_valid_combination(self):
        # Must construct without raising.
        SignSGD_adv([make_param()], lr=1e-3, momentum=0.9,
                    normed_momentum=True, snr_cond=True)

    def test_invalid_orthogonal_gradient_raises(self):
        with self.assertRaises(ValueError):
            SignSGD_adv([make_param()], lr=1e-3, orthogonal_gradient="bogus")

    def test_out_of_range_nesterov_coef_raises(self):
        with self.assertRaises(ValueError):
            SignSGD_adv([make_param()], lr=1e-3, nesterov_coef=1.5)

    def test_invalid_state_precision_raises(self):
        with self.assertRaises(ValueError):
            SignSGD_adv([make_param()], lr=1e-3, state_precision="octal")


class TestCompiledUncompiledParity(unittest.TestCase):
    """The compiled and uncompiled paths must be bit-identical when each runs
    its own deterministic RNG stream (sequential, re-seeded)."""

    def _assert_parity(self, **kwargs):
        torch.manual_seed(0)
        p_init = torch.randn(8, 8, device=DEVICE, dtype=torch.bfloat16) * 0.1
        g = torch.randn(8, 8, device=DEVICE, dtype=torch.bfloat16) * 0.01

        param_update.set_seed(DEVICE)
        p1 = torch.nn.Parameter(p_init.clone())
        opt1 = SignSGD_adv([p1], lr=1e-2, compiled_optimizer=True, **kwargs)
        run_steps(opt1, p1, [g] * 3)

        param_update.set_seed(DEVICE)
        p2 = torch.nn.Parameter(p_init.clone())
        opt2 = SignSGD_adv([p2], lr=1e-2, compiled_optimizer=False, **kwargs)
        run_steps(opt2, p2, [g] * 3)

        torch.testing.assert_close(p1, p2, msg="compiled and uncompiled params diverged")
        if "exp_avg" in opt1.state[p1]:
            torch.testing.assert_close(
                opt1.state[p1]["exp_avg"], opt2.state[p2]["exp_avg"],
                msg="compiled and uncompiled momentum states diverged",
            )

    def test_bf16_sr_state_parity(self):
        # bf16 param + bf16_sr state exercises BOTH the state and parameter SR.
        self._assert_parity(momentum=0.9, state_precision="bf16_sr")

    def test_auto_state_parity(self):
        # bf16 param + auto state exercises only the parameter SR.
        self._assert_parity(momentum=0.9, state_precision="auto")

    def test_stochastic_sign_parity(self):
        # Exercises the SSO noise draw ordering (non-normed: state SR -> SSO -> param SR).
        self._assert_parity(momentum=0.9, stochastic_sign=True)

    def test_normed_momentum_parity(self):
        # Exercises the reversed draw order (SSO noise -> state SR -> param SR).
        self._assert_parity(momentum=0.9, normed_momentum=True, snr_cond=True,
                            state_precision="bf16_sr")


class TestSpectralSnrScaling(unittest.TestCase):
    def test_spectral_path_uses_snr_compensated_scaling(self):
        # scale_update must receive lr * (4/pi) when snr_cond is active, matching
        # the non-spectral path.
        import sys as _sys
        mod = _sys.modules["adv_optm.optim.SignSGD_adv"]  # module (package __init__ shadows it)
        captured = {}
        orig = mod.scale_update

        def spy(p, update, lr, state=None):
            captured["lr"] = float(lr) if isinstance(lr, torch.Tensor) else float(lr)
            return orig(p, update, lr, state=state)

        mod.scale_update = spy
        try:
            p = make_param(shape=(6, 6))
            opt = SignSGD_adv([p], lr=0.01, momentum=0.9, normed_momentum=True,
                              snr_cond=True, spectral_normalization=True)
            p.grad = torch.randn(6, 6, device=DEVICE)
            opt.step()
        finally:
            mod.scale_update = orig
        expected = 0.01 * (4 / math.pi)
        self.assertIsNotNone(captured.get("lr"), "scale_update was never called")
        self.assertAlmostEqual(captured["lr"], expected, places=7)


class TestIsVectorConsistency(unittest.TestCase):
    def test_zero_dim_scalar_not_factored(self):
        # 0-dim scalars must be treated as vectors (no (1,1) factored state).
        p = torch.nn.Parameter(torch.tensor(0.5, device=DEVICE))
        opt = SignSGD_adv([p], lr=1e-2, momentum=0.9, state_precision="factored")
        p.grad = torch.tensor(0.1, device=DEVICE)
        opt.step()
        self.assertIs(opt.state[p].get("factored"), False)

    def test_vector_reshape_factored_runs(self):
        # vector_reshape + factored on a 1D param must allocate and step cleanly.
        p = torch.nn.Parameter(torch.randn(12, device=DEVICE))
        opt = SignSGD_adv([p], lr=1e-2, momentum=0.9,
                          state_precision="factored", vector_reshape=True)
        p.grad = torch.randn(12, device=DEVICE)
        opt.step()
        self.assertIs(opt.state[p].get("factored"), True)


class TestGradIntegrity(unittest.TestCase):
    @staticmethod
    def assert_grad_unchanged(p, grad_before):
        torch.testing.assert_close(p.grad, grad_before,
                                   msg="p.grad was mutated in place by the optimizer step")

    def test_stochastic_sign_grad_unchanged(self):
        p = make_param()
        p.grad = torch.randn(16, 16, device=DEVICE)
        grad_before = p.grad.clone()
        opt = SignSGD_adv([p], lr=1e-3, momentum=0.9, stochastic_sign=True)
        opt.step()
        self.assert_grad_unchanged(p, grad_before)

    def test_normed_snr_grad_unchanged(self):
        p = make_param()
        p.grad = torch.randn(16, 16, device=DEVICE)
        grad_before = p.grad.clone()
        opt = SignSGD_adv([p], lr=1e-3, momentum=0.9, normed_momentum=True, snr_cond=True)
        opt.step()
        self.assert_grad_unchanged(p, grad_before)

    def test_iterative_ortho_grad_unchanged(self):
        p = make_param()
        p.grad = torch.randn(16, 16, device=DEVICE)
        grad_before = p.grad.clone()
        opt = SignSGD_adv([p], lr=1e-3, orthogonal_gradient="iterative")
        opt.step()
        self.assert_grad_unchanged(p, grad_before)


class TestSemantics(unittest.TestCase):
    def test_plain_signsgd(self):
        # No momentum: p -= lr * sign(grad).
        p = torch.nn.Parameter(torch.tensor([0.5, -0.5, 1.0], device=DEVICE))
        opt = SignSGD_adv([p], lr=0.1, momentum=0.0)
        p.grad = torch.tensor([0.3, -0.2, 0.0], device=DEVICE)
        opt.step()
        expected = torch.tensor([0.4, -0.4, 1.0], device=DEVICE)
        torch.testing.assert_close(p, expected)

    def test_nesterov_both_branches_run(self):
        for normed in (True, False):
            with self.subTest(normed_momentum=normed):
                p = make_param(shape=(8, 8))
                opt = SignSGD_adv([p], lr=1e-2, momentum=0.9, nesterov=True,
                                  normed_momentum=normed)
                p.grad = torch.randn(8, 8, device=DEVICE)
                opt.step()

    def test_factored_snr_nesterov_sso_full_feature(self):
        p = make_param(shape=(16, 8))
        opt = SignSGD_adv(
            [p], lr=1e-2, momentum=0.9, normed_momentum=True, snr_cond=True,
            nesterov=True, stochastic_sign=True, geometric_wd=True, weight_decay=1e-2,
            cautious_wd=True, centered_wd=1e-2, centered_wd_mode="int8",
            state_precision="factored",
        )
        p.grad = torch.randn(16, 8, device=DEVICE)
        opt.step()
        self.assertIs(opt.state[p].get("factored"), True)

    def test_int8_sr_state_runs(self):
        p = make_param(shape=(8, 8))
        opt = SignSGD_adv([p], lr=1e-2, momentum=0.9, state_precision="int8_sr")
        p.grad = torch.randn(8, 8, device=DEVICE)
        opt.step()
        self.assertIn("exp_avg", opt.state[p])

    def test_fp16_state_runs(self):
        p = make_param(shape=(8, 8), dtype=torch.float16)
        opt = SignSGD_adv([p], lr=1e-2, momentum=0.9, state_precision="fp16")
        p.grad = torch.randn(8, 8, device=DEVICE, dtype=torch.float16)
        opt.step()
        self.assertIn("exp_avg", opt.state[p])

    def test_load_state_dict_roundtrip(self):
        p = make_param(shape=(8, 8))
        g = torch.randn(8, 8, device=DEVICE)
        opt = SignSGD_adv([p], lr=1e-2, momentum=0.9, state_precision="bf16_sr")
        p.grad = g.clone()
        opt.step()

        sd = opt.state_dict()
        q = make_param(shape=(8, 8))
        opt2 = SignSGD_adv([q], lr=1e-2, momentum=0.9, state_precision="bf16_sr")
        opt2.load_state_dict(sd)
        torch.testing.assert_close(opt.state[p]["exp_avg"], opt2.state[q]["exp_avg"])

        q.grad = g.clone()
        opt2.step()


if __name__ == "__main__":
    unittest.main()
