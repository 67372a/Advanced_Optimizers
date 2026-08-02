"""Unit and functional tests for the code review of `adv_optm/optim/Lion_adv.py`.

Positive tests verify the core Lion update, factored (SMMF) path, BF16
stochastic rounding, Lion-K variants, weight decays, spectral-normalization
initialization semantics, checkpointing, and the torch.compile path.

The tests below encode the CORRECT behaviour that was previously violated by
the defects found during review (all now fixed):

- Defect 1 (HIGH): `compiled_optimizer` was accepted in the constructor but
  never stored in `defaults` nor on `self`, so the torch.compile path was dead
  code. Fixed by adding it to `defaults`.
- Defect 2 (HIGH): `load_state_dict` crashed with
  KeyError: 'actual_state_precision' because Lion_adv never populated that
  group key while `fix_loaded_state_dtype` requires it. Fixed in `__init_state`.
- Defect 3 (HIGH): only fp32 gradients were defensively cloned, so with a
  BF16/FP16 parameter and `orthogonal_gradient='iterative'` the user's `p.grad`
  was mutated in place through tensor views. Fixed by cloning for every dtype.
- Defect 4 (MEDIUM): `orthogonal_gradient`/`kappa_p` were not validated.
  Fixed in the constructor.
- RNG parity: the compiled path drew stochastic-rounding ints before the
  stochastic-sign noise (reversed vs. the uncompiled path). Fixed so both
  paths draw SSO noise first, then SR ints, with matching dtypes.

All tests run on CUDA as mandated by the project conventions.
"""

import importlib
import os
import sys
import unittest
from unittest import mock

import torch

# Make the package importable when running from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NOTE: `from adv_optm.optim import Lion_adv` would resolve to the *class*
# because the package __init__ re-exports it, shadowing the submodule. Use
# importlib to reliably obtain the module for attribute patching.
lion_mod = importlib.import_module("adv_optm.optim.Lion_adv")  # noqa: E402
from adv_optm.optim.Lion_adv import Lion_adv  # noqa: E402
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


class TestFunctional(unittest.TestCase):
    """Positive sanity checks; these must pass on the current code."""

    def test_fp32_step_updates_params_and_state(self):
        p = make_param((16, 16))
        p_before = p.detach().clone()
        opt = Lion_adv([p], lr=1e-3)
        opt.step()
        self.assertFalse(torch.equal(p, p_before))
        st = opt.state[p]
        self.assertEqual(st["step"], 1)
        self.assertIn("exp_avg", st)

    def test_factored_step(self):
        p = make_param((16, 16))
        opt = Lion_adv([p], lr=1e-3, nnmf_factor=True)
        opt.step()
        st = opt.state[p]
        self.assertTrue(st["factored"])
        self.assertIn("mu_m_nmf", st)
        self.assertIn("mv_m_nmf", st)
        self.assertIn("sign", st)
        self.assertEqual(st["step"], 1)

    def test_bf16_step_with_stochastic_rounding(self):
        p = make_param((16, 16), dtype=torch.bfloat16)
        p_before = p.detach().clone()
        opt = Lion_adv([p], lr=1e-3)
        opt.step()
        self.assertFalse(torch.equal(p, p_before))

    def test_kappa_p_variants(self):
        for kp in (1.0, 2.0, 1.5):
            p = make_param((16, 16))
            opt = Lion_adv([p], lr=1e-3, kappa_p=kp)
            opt.step()  # must not raise

    def test_auto_kappa_p(self):
        p = make_param((4, 4, 3, 3))  # 4D -> spherical (p=2)
        opt = Lion_adv([p], lr=1e-3, auto_kappa_p=True)
        opt.step()  # must not raise

    def test_stochastic_sign(self):
        p = make_param((16, 16))
        opt = Lion_adv([p], lr=1e-3, stochastic_sign=True)
        opt.step()  # must not raise

    def test_weight_decay(self):
        p = make_param((16, 16))
        opt = Lion_adv([p], lr=1e-3, weight_decay=0.01)
        opt.step()  # must not raise

    def test_cautious_wd(self):
        p = make_param((16, 16))
        opt = Lion_adv([p], lr=1e-3, weight_decay=0.01, cautious_wd=True)
        opt.step()  # must not raise

    def test_centered_wd_float8(self):
        p = make_param((16, 16))
        opt = Lion_adv([p], lr=1e-3, centered_wd=0.01, centered_wd_mode="float8")
        opt.step()  # must not raise
        self.assertIn("anchor_data", opt.state[p])

    def test_centered_wd_int4(self):
        p = make_param((16, 16))
        opt = Lion_adv([p], lr=1e-3, centered_wd=0.01, centered_wd_mode="int4")
        opt.step()  # must not raise
        self.assertIn("anchor_data", opt.state[p])

    def test_orthogonal_gradient_flattened_and_iterative(self):
        for mode in ("flattened", "iterative"):
            p = make_param((16, 16))
            opt = Lion_adv([p], lr=1e-3, orthogonal_gradient=mode)
            opt.step()  # must not raise

    def test_vector_reshape_1d(self):
        p = make_param((64,))
        opt = Lion_adv([p], lr=1e-3, nnmf_factor=True, vector_reshape=True)
        opt.step()  # must not raise


class TestCompiledOptimizerFlag(unittest.TestCase):
    """Defect 1: compiled_optimizer must reach the param group (dead-code flag)."""

    def test_compiled_flag_reaches_param_group(self):
        p = make_param((8, 8))
        opt = Lion_adv([p], lr=1e-3, compiled_optimizer=True)
        # The constructor arg must be visible to the group so the step path
        # can actually take the torch.compile branch.
        self.assertTrue(opt.param_groups[0].get("compiled_optimizer", False))


class TestSpectralNormInitOnce(unittest.TestCase):
    """Spectral vectors are initialized once and persist across steps."""

    def test_spectral_vectors_not_reinitialized_every_step(self):
        p = make_param((8, 8))
        real_init = lion_mod.init_spectral_norm
        call_count = {"n": 0}

        def counting_init(state, p):
            call_count["n"] += 1
            return real_init(state, p)

        with mock.patch.object(lion_mod, "init_spectral_norm", counting_init):
            opt = Lion_adv([p], lr=1e-3, spectral_normalization=True)
            # Called exactly once during construction (init_step).
            self.assertEqual(call_count["n"], 1)
            opt.step()
            opt.step()
            # Correct behaviour: power-iteration vectors persist across steps,
            # so init must not run again after construction.
            self.assertEqual(call_count["n"], 1)


class TestGradNotMutated(unittest.TestCase):
    """Defect 3: the step must never mutate the user's p.grad buffer."""

    def test_bf16_2d_grad_unchanged_iterative_ortho(self):
        p = make_param((16, 16), dtype=torch.bfloat16)
        grad_before = p.grad.clone()
        opt = Lion_adv([p], lr=1e-3, orthogonal_gradient="iterative", nnmf_factor=False)
        opt.step()
        torch.testing.assert_close(p.grad, grad_before)

    def test_fp16_2d_grad_unchanged_iterative_ortho(self):
        p = make_param((16, 16), dtype=torch.float16)
        grad_before = p.grad.clone()
        opt = Lion_adv([p], lr=1e-3, orthogonal_gradient="iterative", nnmf_factor=False)
        opt.step()
        torch.testing.assert_close(p.grad, grad_before)


class TestValidation(unittest.TestCase):
    """Defect 4: invalid option values should be rejected at construction."""

    def test_invalid_orthogonal_gradient_rejected(self):
        with self.assertRaises(ValueError):
            Lion_adv([make_param((8, 8))], orthogonal_gradient="bogus")

    def test_kappa_p_out_of_domain_rejected(self):
        with self.assertRaises(ValueError):
            Lion_adv([make_param((8, 8))], kappa_p=3.0)


class TestLoadStateDict(unittest.TestCase):
    """Checkpoint round-trip must not crash (Defect 2 fix)."""

    def test_state_dict_roundtrip(self):
        p = make_param((16, 16))
        opt = Lion_adv([p], lr=1e-3)
        opt.step()
        sd = opt.state_dict()

        p2 = make_param((16, 16))
        opt2 = Lion_adv([p2], lr=1e-3)
        opt2.load_state_dict(sd)  # must not raise
        self.assertEqual(opt2.state[p2]["step"], 1)

    def test_state_dict_roundtrip_factored(self):
        p = make_param((16, 16))
        opt = Lion_adv([p], lr=1e-3, nnmf_factor=True)
        opt.step()
        sd = opt.state_dict()

        p2 = make_param((16, 16))
        opt2 = Lion_adv([p2], lr=1e-3, nnmf_factor=True)
        opt2.load_state_dict(sd)  # must not raise
        self.assertEqual(opt2.state[p2]["step"], 1)
        self.assertTrue(opt2.state[p2]["factored"])

    def test_state_dict_roundtrip_bf16_preserves_state(self):
        p = make_param((16, 16), dtype=torch.bfloat16)
        opt = Lion_adv([p], lr=1e-3)
        opt.step()
        sd = opt.state_dict()

        p2 = make_param((16, 16), dtype=torch.bfloat16)
        opt2 = Lion_adv([p2], lr=1e-3)
        opt2.load_state_dict(sd)  # must not raise
        self.assertEqual(opt2.state[p2]["step"], 1)


class TestCompiledPath(unittest.TestCase):
    """torch.compile path (Defect 1 fix + RNG parity fix)."""

    def test_compiled_step_runs(self):
        p = make_param((16, 16))
        p_before = p.detach().clone()
        opt = Lion_adv([p], lr=1e-3, compiled_optimizer=True)
        opt.step()
        self.assertFalse(torch.equal(p, p_before))
        self.assertEqual(opt.state[p]["step"], 1)
        # The compiled function cache must have been populated.
        self.assertTrue(len(opt._compiled_step_fns) > 0)

    def test_compiled_vs_uncompiled_deterministic_parity(self):
        # Same seed + same inputs must produce bit-identical parameters on
        # the compiled and uncompiled paths (RNG draw order/dtype parity).
        torch.manual_seed(0)
        p_c = torch.nn.Parameter(torch.randn(16, 16, device=DEVICE, dtype=torch.bfloat16) * 0.1)
        p_c.grad = torch.randn(16, 16, device=DEVICE, dtype=torch.bfloat16) * 0.1
        p_u = torch.nn.Parameter(p_c.detach().clone())
        p_u.grad = p_c.grad.detach().clone()

        opt_c = Lion_adv([p_c], lr=1e-3, compiled_optimizer=True, stochastic_sign=True)
        opt_u = Lion_adv([p_u], lr=1e-3, compiled_optimizer=False, stochastic_sign=True)

        # Re-seed the shared per-device generator so each path draws the same
        # deterministic stream in isolation.
        param_update.set_seed(DEVICE)
        for _ in range(3):
            opt_c.step()
        param_update.set_seed(DEVICE)
        for _ in range(3):
            opt_u.step()

        torch.testing.assert_close(p_c, p_u)

    def test_compiled_factored_step_runs(self):
        p = make_param((16, 16))
        opt = Lion_adv([p], lr=1e-3, nnmf_factor=True, compiled_optimizer=True)
        opt.step()  # must not raise
        self.assertEqual(opt.state[p]["step"], 1)
        self.assertTrue(opt.state[p]["factored"])


if __name__ == "__main__":
    unittest.main()
