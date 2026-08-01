"""Functional tests for the NorMuon variant (normuon_variant=True) in AdaMuon_adv.

Verifies the mechanics of the row-wise second-moment normalization:
- State allocation: per-row `normuon_v` of shape (rows,) replaces the full-size
  element-wise `second_momentum_buffer`.
- Shared-LR calibration is preserved: with rms_rescaling=True the applied
  update RMS is still pinned to 0.2 * lr.
- atan2 interaction: the 4/pi step-scale correction is intentionally NOT
  applied when normuon_variant=True (RMS target stays 0.2 * lr, vs
  0.2 * 4/pi * lr for atan2 without normuon).
- 1D graceful degradation: 1D params on the Muon path use global-RMS
  normalization (normuon_v is not allocated for 1D) and must not crash.

All tests run on CUDA as mandated by the project conventions.
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adv_optm.optim.AdaMuon_adv import AdaMuon_adv  # noqa: E402

DEVICE = torch.device("cuda:0")
A = 4.0 / torch.pi  # atan2 step-scale correction factor


def _rms(t: torch.Tensor) -> float:
    return t.square().mean().sqrt().item()


class TestNorMuon(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for these tests")
        torch.manual_seed(42)

    @staticmethod
    def make_param(shape):
        return torch.nn.Parameter(torch.randn(*shape, device=DEVICE, dtype=torch.float32) * 0.1)

    @staticmethod
    def run_steps(opt, p, steps=10):
        deltas = []
        for _ in range(steps):
            p.grad = torch.randn_like(p) * 0.1
            before = p.detach().clone()
            opt.step()
            deltas.append(p.detach() - before)
        return deltas

    def test_state_allocation_row_wise(self):
        """normuon_v must be a per-row vector; no full-size second moment."""
        rows, cols = 16, 64
        p = self.make_param((rows, cols))
        opt = AdaMuon_adv(
            [p], lr=1e-3, use_muon=True, normuon_variant=True,
            weight_decay=0.0, stochastic_rounding=False,
        )
        state = opt.state[p]
        self.assertIn('normuon_v', state)
        self.assertEqual(state['normuon_v'].shape, (rows,))
        self.assertNotIn('second_momentum_buffer', state)

    def test_shared_lr_calibration_preserved(self):
        """rms_rescaling still pins the applied update RMS to 0.2 * lr."""
        lr = 2e-3
        p = self.make_param((32, 64))
        opt = AdaMuon_adv(
            [p], lr=lr, use_muon=True, normuon_variant=True,
            weight_decay=0.0, rms_rescaling=True, stochastic_rounding=False,
        )
        for delta in self.run_steps(opt, p):
            self.assertAlmostEqual(_rms(delta) / lr, 0.2, delta=2e-3)

    def test_atan2_correction_skipped_with_normuon(self):
        """With use_atan2, the 4/pi step-scale applies only WITHOUT normuon."""
        lr = 2e-3
        p_nm = self.make_param((32, 64))
        p_at = self.make_param((32, 64))
        p_at.data.copy_(p_nm.data)

        opt_nm = AdaMuon_adv(
            [p_nm], lr=lr, use_muon=True, normuon_variant=True, use_atan2=True,
            weight_decay=0.0, rms_rescaling=True, stochastic_rounding=False,
        )
        opt_at = AdaMuon_adv(
            [p_at], lr=lr, use_muon=True, normuon_variant=False, use_atan2=True,
            weight_decay=0.0, rms_rescaling=True, stochastic_rounding=False,
        )
        r_nm = _rms(self.run_steps(opt_nm, p_nm, steps=5)[-1]) / lr
        r_at = _rms(self.run_steps(opt_at, p_at, steps=5)[-1]) / lr
        self.assertAlmostEqual(r_nm, 0.2, delta=2e-3)          # no A factor
        self.assertAlmostEqual(r_at, 0.2 * A, delta=2e-3)      # A factor applied

    def test_1d_param_graceful_degradation(self):
        """1D params on the Muon path fall back to global-RMS normalization."""
        lr = 1e-3
        p = self.make_param((64,))
        opt = AdaMuon_adv(
            [p], lr=lr, use_muon=True, normuon_variant=True,
            weight_decay=0.0, rms_rescaling=True, stochastic_rounding=False,
        )
        self.assertNotIn('normuon_v', opt.state[p])
        for delta in self.run_steps(opt, p):
            self.assertTrue(torch.isfinite(delta).all())
            self.assertAlmostEqual(_rms(delta) / lr, 0.2, delta=2e-3)

    def test_runs_with_mars_and_nesterov(self):
        """NorMuon must compose with approx_mars and nesterov (LoRA ablation config)."""
        lr = 1e-3
        p = self.make_param((16, 256))
        opt = AdaMuon_adv(
            [p], lr=lr, use_muon=True, normuon_variant=True,
            approx_mars=True, nesterov=True,
            weight_decay=0.0, rms_rescaling=True, stochastic_rounding=False,
        )
        for delta in self.run_steps(opt, p):
            self.assertTrue(torch.isfinite(delta).all())
            self.assertAlmostEqual(_rms(delta) / lr, 0.2, delta=2e-3)


if __name__ == "__main__":
    unittest.main()
