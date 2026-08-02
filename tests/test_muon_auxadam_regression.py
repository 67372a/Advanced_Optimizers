"""
Regression tests for defects found in adv_optm/util/Muon_AuxAdam.py.

Covered fixes:
1. Factored AuxAdam crashed with KeyError('shifter') when adam_betas[0] == 0
   (shifter was only initialized inside the `beta1 > 0` branch).
2. `adam_fisher_wd` was silently ignored: the Fisher-WD helpers read
   `group['fisher_wd']`, but Muon_adv / AdaMuon_adv store `adam_fisher_wd`.
3. Muon_adv clobbered the dynamic Kourkoutas-β beta2 with the static
   `group['adam_betas']` inside the bias-correction branch.

All tests run on CUDA.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adv_optm.optim import Muon_adv, AdaMuon_adv
from adv_optm.util import Muon_AuxAdam
from adv_optm.util.update_util import _init_fisher_wd_scaler, _get_fisher_wd_scaler

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class FakeOpt:
    """Minimal stand-in for the optimizer object consumed by the AuxAdam helpers."""

    def __init__(self, lr=1e-3):
        self.state = {}
        self._init_lr = lr if lr > 0 else 1


def _base_group(**overrides):
    group = {
        "lr": 1e-3,
        "adam_state_precision": "auto",
        "adam_nnmf_factor": False,
        "adam_factored_2nd": False,
        "adam_betas": (0.9, 0.99),
        "adam_eps": 1e-8,
        "adam_use_atan2": False,
        "adam_orthogonal_gradient": "disabled",
        "adam_nesterov": False,
        "adam_nesterov_coef": None,
        "adam_spectral_normalization": False,
        "adam_weight_decay": 0.0,
        "adam_use_bias_correction": True,
        "fisher_wd": False,
        "centered_wd": 0.0,
        "weight_decay": 0.0,
    }
    group.update(overrides)
    return group


def _run_helper_step(opt, p, group, beta1, beta2, step=1):
    state = opt.state[p]
    current_step = step + 1
    bias_correction1 = 1.0 - beta1 ** current_step
    sqrt_bias_correction2 = (1.0 - beta2 ** current_step) ** 0.5
    step_size = group["lr"] / bias_correction1
    grad = torch.randn_like(p)
    Muon_AuxAdam._adam_step_parameter(
        opt, p, grad, state, group,
        beta1, beta2, sqrt_bias_correction2, step_size,
        random_int_tensor=None, random_int_state_tensor=None,
    )
    return state


# ---------------------------------------------------------------------------
# Fix 1: factored state with beta1 == 0 must not crash on a missing shifter
# ---------------------------------------------------------------------------

def test_factored_step_beta1_zero_no_shifter_crash():
    p = torch.nn.Parameter(torch.randn(6, 4, device="cuda"))
    opt = FakeOpt()
    opt.state[p] = {}
    group = _base_group(adam_nnmf_factor=True, adam_betas=(0.0, 0.99))

    Muon_AuxAdam._init_auxadam_state(opt, p, group)
    state = opt.state[p]
    assert state["factored"] is True
    assert "shifter" in state, "shifter must be initialized even when beta1 == 0"

    before = p.detach().clone()
    _run_helper_step(opt, p, group, beta1=0.0, beta2=0.99)
    assert not torch.equal(p.detach(), before), "parameter should have been updated"
    assert "mu_v_nmf" in state and "mv_v_nmf" in state


def test_factored_step_beta1_zero_second_step():
    """Two consecutive steps (exercises factor/reconstruct round-trip)."""
    p = torch.nn.Parameter(torch.randn(8, 8, device="cuda"))
    opt = FakeOpt()
    opt.state[p] = {}
    group = _base_group(adam_nnmf_factor=True, adam_betas=(0.0, 0.99))

    Muon_AuxAdam._init_auxadam_state(opt, p, group)
    for step in (1, 2):
        _run_helper_step(opt, p, group, beta1=0.0, beta2=0.99, step=step)


# ---------------------------------------------------------------------------
# Fix 2: adam_fisher_wd must activate the Fisher-WD scaler
# ---------------------------------------------------------------------------

def test_adam_fisher_wd_creates_scaler():
    p = torch.nn.Parameter(torch.randn(6, 4, device="cuda"))
    opt = FakeOpt()
    opt.state[p] = {}
    group = _base_group(adam_fisher_wd=True)  # the key the optimizers actually set

    Muon_AuxAdam._init_auxadam_state(opt, p, group)
    wd_scaler = opt.state[p].get("wd_scaler")
    assert wd_scaler is not None, "adam_fisher_wd=True must create wd_scaler"
    assert wd_scaler.item() == 1.0

    # The step path must also compute a (non-None) Fisher scaler.
    state = _run_helper_step(opt, p, group, beta1=0.9, beta2=0.99)
    assert state.get("wd_scaler") is not None


def test_fisher_wd_legacy_key_still_works():
    """Plain optimizers (AdamW_adv, Adopt_adv, Prodigy_adv) use 'fisher_wd'."""
    p = torch.nn.Parameter(torch.randn(6, 4, device="cuda"))
    state = {}
    group = _base_group(fisher_wd=True)
    _init_fisher_wd_scaler(group, state, p)
    assert state["wd_scaler"].item() == 1.0

    denom = torch.ones(6, 4, device="cuda")
    scaler = _get_fisher_wd_scaler(group, state["wd_scaler"], p, denom, atan2=False, eps=1e-8)
    assert scaler is not None


def test_fisher_wd_disabled_when_both_flags_false():
    p = torch.nn.Parameter(torch.randn(6, 4, device="cuda"))
    state = {}
    group = _base_group()
    _init_fisher_wd_scaler(group, state, p)
    assert "wd_scaler" not in state


# ---------------------------------------------------------------------------
# Fix 3: Muon_adv must forward the dynamic Kourkoutas-β beta2 to the Adam step
# ---------------------------------------------------------------------------

def test_muon_adv_dynamic_kourkoutas_beta2_not_clobbered():
    p = torch.nn.Parameter(torch.randn(8, 4, device="cuda"))
    p.is_hidden = False  # route to the AuxAdam path

    opt = Muon_adv(
        [p],
        lr=1e-3,
        adam_betas=(0.9, 0.99),
        adam_kourkoutas_beta=True,
        adam_ema_alpha=0.95,
        adam_beta2_min=0.9,
    )
    assert opt.kourkoutas_helper is not None

    # Force a distinctive dynamic beta2 for this step.
    opt.kourkoutas_helper.get_beta2 = lambda _p, _g: 0.321

    captured = {}

    original = Muon_AuxAdam._adam_step_parameter

    def _wrapped(self, step_p, grad, state, group, beta1, beta2, sqrt_bc2, step_size, rit, rist):
        captured["beta2"] = beta2
        captured["sqrt_bias_correction2"] = sqrt_bc2
        return original(self, step_p, grad, state, group, beta1, beta2, sqrt_bc2, step_size, rit, rist)

    Muon_AuxAdam._adam_step_parameter = _wrapped
    try:
        p.grad = torch.randn_like(p)
        opt.step()
    finally:
        Muon_AuxAdam._adam_step_parameter = original

    assert captured["beta2"] == pytest.approx(0.321), \
        "dynamic Kourkoutas beta2 must reach _adam_step_parameter"
    # sqrt_bias_correction2 must be computed from the *dynamic* beta2 (0.321),
    # not the static 0.99 (which would give ~0.1).
    assert captured["sqrt_bias_correction2"] == pytest.approx((1.0 - 0.321) ** 0.5, rel=1e-4)


# ---------------------------------------------------------------------------
# End-to-end smoke tests for the Adam path (factored / factored_2nd)
# ---------------------------------------------------------------------------

def test_muon_adv_adam_path_factored_smoke():
    p = torch.nn.Parameter(torch.randn(10, 6, device="cuda"))
    p.is_hidden = False

    opt = Muon_adv(
        [p],
        lr=1e-3,
        adam_betas=(0.0, 0.99),  # beta1 == 0 exercises the shifter fix end-to-end
        adam_nnmf_factor=True,
    )
    before = p.detach().clone()
    for _ in range(2):
        p.grad = torch.randn_like(p)
        opt.step()
    assert not torch.equal(p.detach(), before)
    state = opt.state[p]
    assert state["factored"] is True


def test_adamuon_adv_adam_path_factored_2nd_smoke():
    p = torch.nn.Parameter(torch.randn(10, 6, device="cuda"))
    p.is_hidden = False

    opt = AdaMuon_adv(
        [p],
        lr=1e-3,
        adam_betas=(0.9, 0.99),
        adam_factored_2nd=True,
        adam_nnmf_factor=False,
    )
    before = p.detach().clone()
    for _ in range(2):
        p.grad = torch.randn_like(p)
        opt.step()
    assert not torch.equal(p.detach(), before)
    state = opt.state[p]
    assert state.get("factored_2nd") is True
    assert "mu_v_nmf" in state and "mv_v_nmf" in state
