"""Functional tests: one shared learning rate for the Muon and AdamW paths of
AdaMuon_adv.

Verifies the claim that `rms_rescaling=True` (the default) RMS-aligns the Muon
update with Adam's update magnitude (~0.2 RMS), so a single learning rate and
schedule can drive both parameter types in the same optimizer:

- The Muon path rescales its final update to RMS == 0.2 * lr exactly
  (`rms_adjustment` in adv_optm/util/Muon_util.py).
- The AdamW path (Muon_AuxAdam) applies the standard Adam update m_hat/sqrt(v_hat)
  scaled by lr, whose empirical RMS for decorrelated gradients is ~0.2 * lr.
- A single optimizer instance holding mixed 'muon'/'adam' param groups with one
  shared lr produces per-step parameter deltas of the same order of magnitude
  on both paths.
- With rms_rescaling=False the Muon path falls back to aspect-ratio scaling and
  the shared-lr calibration no longer holds (documenting the caveat).

All tests run on CUDA as mandated by the project conventions.
"""

import os
import sys
import unittest

import torch

# Make the package importable when running from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adv_optm.optim.AdaMuon_adv import AdaMuon_adv  # noqa: E402

DEVICE = torch.device("cuda:0")
RMS_TARGET = 0.2  # Adam-matched RMS target used by rms_adjustment


def _rms(t: torch.Tensor) -> float:
    return t.square().mean().sqrt().item()


class SharedLRTestCase(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for these tests")
        torch.manual_seed(1234)

    @staticmethod
    def make_param(shape, dtype=torch.float32):
        return torch.nn.Parameter(torch.randn(*shape, device=DEVICE, dtype=dtype) * 0.1)

    @staticmethod
    def step_with_fresh_grad(opt, params, scale=0.1):
        """Assign fresh white-noise grads (decorrelated, like real training) and step."""
        for p in params:
            p.grad = torch.randn_like(p) * scale
        before = [p.detach().clone() for p in params]
        opt.step()
        deltas = [p.detach() - b for p, b in zip(params, before)]
        return deltas

    def test_muon_update_rms_matches_adam_target(self):
        """With rms_rescaling=True (default) and wd=0, the Muon path must apply
        an update whose RMS is exactly 0.2 * lr (rms_adjustment normalizes the
        final update), which is precisely Adam's empirical update RMS."""
        lr = 2e-3
        p = self.make_param((64, 64))
        opt = AdaMuon_adv(
            [p], lr=lr, use_muon=True, weight_decay=0.0,
            rms_rescaling=True, stochastic_rounding=False,
        )
        for _ in range(10):
            (delta,) = self.step_with_fresh_grad(opt, [p])
            rms_over_lr = _rms(delta) / lr
            self.assertAlmostEqual(
                rms_over_lr, RMS_TARGET, delta=2e-3,
                msg=f"Muon update RMS/lr = {rms_over_lr:.6f}, expected ~{RMS_TARGET}",
            )

    def test_adam_update_rms_same_order_of_magnitude(self):
        """The AuxAdam path scales m_hat/sqrt(v_hat) by lr directly. For
        white-noise gradients with adam_betas=(0.9, 0.99) the steady-state RMS
        is ~sqrt((1-b1)/(1+b1)) ~ 0.23, i.e. the same magnitude as the Muon
        path's 0.2 target under the same lr."""
        lr = 2e-3
        p = self.make_param((64, 64))
        opt = AdaMuon_adv(
            [p], lr=lr, use_muon=False, weight_decay=0.0, adam_weight_decay=0.0,
            stochastic_rounding=False,
        )
        ratios = []
        for i in range(40):
            (delta,) = self.step_with_fresh_grad(opt, [p])
            if i >= 20:  # skip bias-correction / v_t warmup
                ratios.append(_rms(delta) / lr)
        mean_ratio = sum(ratios) / len(ratios)
        self.assertGreater(mean_ratio, 0.10, f"Adam RMS/lr too small: {mean_ratio:.4f}")
        self.assertLess(mean_ratio, 0.40, f"Adam RMS/lr too large: {mean_ratio:.4f}")

    def test_single_optimizer_shared_lr_mixed_groups(self):
        """The user scenario: ONE AdaMuon_adv instance, ONE lr, mixed param
        groups (2D weight on Muon, embedding + bias on AdamW). Per-step delta
        RMS/lr of the two paths must agree within a factor of 2."""
        lr = 1e-3
        w = self.make_param((128, 64))          # hidden weight -> Muon
        emb = self.make_param((256, 32))        # embedding -> AdamW (explicit)
        bias = self.make_param((64,))           # 1D -> AdamW
        opt = AdaMuon_adv(
            [
                {"params": [w], "use_muon": True},
                {"params": [emb, bias], "use_muon": False},
            ],
            lr=lr, weight_decay=0.0, adam_weight_decay=0.0,
            rms_rescaling=True, stochastic_rounding=False,
        )
        muon_ratio = adam_ratio = None
        for _ in range(30):
            d_w, d_emb, d_bias = self.step_with_fresh_grad(opt, [w, emb, bias])
            muon_ratio = _rms(d_w) / lr
            adam_ratio = 0.5 * (_rms(d_emb) + _rms(d_bias)) / lr

        # Muon side is pinned to the 0.2 target by rms_adjustment.
        self.assertAlmostEqual(muon_ratio, RMS_TARGET, delta=2e-3)
        # Adam side lands in the same band.
        self.assertGreater(adam_ratio, 0.10)
        self.assertLess(adam_ratio, 0.40)
        # And the two paths agree within a factor of two.
        ratio = muon_ratio / adam_ratio
        self.assertGreater(ratio, 0.5, f"Muon/Adam update ratio {ratio:.3f} out of band")
        self.assertLess(ratio, 2.0, f"Muon/Adam update ratio {ratio:.3f} out of band")

    def test_rms_rescaling_disabled_breaks_shared_lr_calibration(self):
        """Caveat check: with rms_rescaling=False the Muon path uses the
        original aspect-ratio scaling lr * sqrt(max(1, r/c)), which is not
        matched to Adam's 0.2 RMS - so a single shared lr is no longer
        calibrated between the two paths."""
        lr = 2e-3
        p = self.make_param((16, 512))
        opt = AdaMuon_adv(
            [p], lr=lr, use_muon=True, weight_decay=0.0,
            rms_rescaling=False, stochastic_rounding=False,
        )
        rms_over_lr = None
        for _ in range(10):
            (delta,) = self.step_with_fresh_grad(opt, [p])
            rms_over_lr = _rms(delta) / lr
        self.assertFalse(
            0.16 <= rms_over_lr <= 0.24,
            f"With rms_rescaling=False the Muon RMS/lr should NOT match the "
            f"Adam target, got {rms_over_lr:.4f}",
        )


if __name__ == "__main__":
    unittest.main()
