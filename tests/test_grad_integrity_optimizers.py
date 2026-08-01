"""Gradient-integrity tests for all *_adv optimizers.

Verifies the defensive fp32-gradient clone that was added to every optimizer's
step_parameter wrapper (mirroring the SinkSGD_adv fix). Before the fix, in-place
normalization/sign operations applied to the upcast gradient corrupted the
user's p.grad buffer for fp32 parameters (upcast_grad_for_precision returns the
same tensor for fp32). Affected in-place operations include:

- grad.sign_() / apply_stochastic_sign_ on grad (SignSGD)
- grad.atan2_/div_ under normed_momentum (AdamW)
- apply_sr_sinkhorn / grad.sign_ (SinkSGD)
- iterative_ortho_project (shared OrthoGrad util, all optimizers)

All tests run on CUDA as mandated by the project conventions.
"""

import os
import sys
import unittest

import torch

# Make the package importable when running from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adv_optm.optim import (  # noqa: E402
    AdaMuon_adv,
    AdamW_adv,
    Adopt_adv,
    Lion_adv,
    Muon_adv,
    Prodigy_adv,
    SignSGD_adv,
    SinkSGD_adv,
)

DEVICE = torch.device("cuda:0")
torch.manual_seed(0)


def make_param(shape=(16, 16), dtype=torch.float32, is_muon=False):
    p = torch.nn.Parameter(torch.randn(*shape, device=DEVICE, dtype=dtype) * 0.1)
    p.grad = torch.randn(*shape, device=DEVICE, dtype=dtype) * 0.1
    if is_muon:
        p._is_muon = True
    return p


# Every optimizer accepts lr and orthogonal_gradient kwargs.
OPTIMIZERS = {
    "AdamW_adv": AdamW_adv,
    "Prodigy_adv": Prodigy_adv,
    "Adopt_adv": Adopt_adv,
    "Lion_adv": Lion_adv,
    "Muon_adv": Muon_adv,
    "AdaMuon_adv": AdaMuon_adv,
    "SignSGD_adv": SignSGD_adv,
    "SinkSGD_adv": SinkSGD_adv,
}


class GradIntegrityTestCase(unittest.TestCase):
    @staticmethod
    def assert_grad_unchanged(p, grad_before):
        torch.testing.assert_close(
            p.grad, grad_before,
            msg="p.grad was mutated in place by the optimizer step",
        )


class TestDefaultConfig(GradIntegrityTestCase):
    def test_fp32_grad_unchanged_for_all_optimizers(self):
        for name, opt_cls in OPTIMIZERS.items():
            with self.subTest(optimizer=name):
                p = make_param()
                grad_before = p.grad.clone()
                opt = opt_cls([p], lr=1e-3)
                opt.step()
                self.assert_grad_unchanged(p, grad_before)

    def test_bf16_param_grad_unchanged_for_all_optimizers(self):
        for name, opt_cls in OPTIMIZERS.items():
            with self.subTest(optimizer=name):
                p = make_param(dtype=torch.bfloat16)
                grad_before = p.grad.clone()
                opt = opt_cls([p], lr=1e-3)
                opt.step()
                self.assert_grad_unchanged(p, grad_before)


class TestIterativeOrthoGrad(GradIntegrityTestCase):
    """iterative_ortho_project mutates its input in place (shared OrthoGrad util)."""

    def test_fp32_grad_unchanged_with_iterative_ortho(self):
        for name, opt_cls in OPTIMIZERS.items():
            with self.subTest(optimizer=name):
                p = make_param()
                grad_before = p.grad.clone()
                opt = opt_cls([p], lr=1e-3, orthogonal_gradient="iterative")
                opt.step()
                self.assert_grad_unchanged(p, grad_before)


class TestInPlaceFeaturePaths(GradIntegrityTestCase):
    """Optimizers with in-place operations directly on the gradient."""

    def test_adamw_normed_momentum(self):
        # grad.atan2_(denom) / grad.div_(denom) run in place on the upcast grad.
        p = make_param()
        grad_before = p.grad.clone()
        opt = AdamW_adv([p], lr=1e-3, normed_momentum=True)
        opt.step()
        self.assert_grad_unchanged(p, grad_before)

    def test_signsgd_normed_snr(self):
        # grad.sign_() and apply_stochastic_sign_ on grad under normed_momentum.
        p = make_param()
        grad_before = p.grad.clone()
        opt = SignSGD_adv([p], lr=1e-3, normed_momentum=True, snr_cond=True)
        opt.step()
        self.assert_grad_unchanged(p, grad_before)

    def test_signsgd_stochastic_sign(self):
        p = make_param()
        grad_before = p.grad.clone()
        opt = SignSGD_adv([p], lr=1e-3, stochastic_sign=True)
        opt.step()
        self.assert_grad_unchanged(p, grad_before)

    def test_sinksgd_normed_snr(self):
        p = make_param()
        grad_before = p.grad.clone()
        opt = SinkSGD_adv([p], lr=1e-3, momentum=0.9, normed_momentum=True, snr_cond=True)
        opt.step()
        self.assert_grad_unchanged(p, grad_before)

    def test_lion_stochastic_sign(self):
        p = make_param()
        grad_before = p.grad.clone()
        opt = Lion_adv([p], lr=1e-3, stochastic_sign=True)
        opt.step()
        self.assert_grad_unchanged(p, grad_before)


class TestMuonPath(GradIntegrityTestCase):
    def test_muon_path_grad_unchanged(self):
        for name, opt_cls in (("Muon_adv", Muon_adv), ("AdaMuon_adv", AdaMuon_adv)):
            with self.subTest(optimizer=name):
                p = make_param(is_muon=True)
                grad_before = p.grad.clone()
                opt = opt_cls([p], lr=1e-3)
                opt.step()
                self.assert_grad_unchanged(p, grad_before)


if __name__ == "__main__":
    unittest.main()
