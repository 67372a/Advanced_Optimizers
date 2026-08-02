"""
Review-driven verification tests for adv_optm/util/Muon_AuxAdam.py.

These tests exercise the AuxAdam step/init paths directly (factored,
factored_2nd, beta1 == 0, nesterov, atan2, precision modes, spectral
normalization, Fisher WD) and specifically probe the tensor-`beta2`
(Kourkoutas per-row) handling, where a broadcast mismatch is suspected
between the factored branch (denom shaped (d1, d2)) and the non-factored
branch (denom shaped p.shape).

All tests run on CUDA.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adv_optm.util import Muon_AuxAdam

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class FakeOpt:
    """Minimal stand-in for the optimizer object consumed by the AuxAdam helpers."""

    def __init__(self, lr=1e-3):
        self.state = {}
        self._init_lr = lr if lr > 0 else 1
        self.kourkoutas_helper = None
        self.stochastic_rounding = False


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
        "adam_fisher_wd": False,
        "fisher_wd": False,
        "centered_wd": 0.0,
        "weight_decay": 0.0,
    }
    group.update(overrides)
    return group


def _run_helper_step(opt, p, group, beta1, beta2, step=1, sqrt_bias_correction2=None):
    state = opt.state[p]
    current_step = step + 1
    bias_correction1 = 1.0 - beta1 ** current_step
    if sqrt_bias_correction2 is None:
        sqrt_bias_correction2 = (1.0 - beta2 ** current_step) ** 0.5
    step_size = group["lr"] / bias_correction1
    grad = torch.randn_like(p)
    Muon_AuxAdam._adam_step_parameter(
        opt, p, grad, state, group,
        beta1, beta2, sqrt_bias_correction2, step_size,
        random_int_tensor=None, random_int_state_tensor=None,
    )
    return state


def _init(opt, p, group):
    opt.state[p] = {}
    Muon_AuxAdam._init_auxadam_state(opt, p, group)
    return opt.state[p]


# ---------------------------------------------------------------------------
# Smoke: factored paths
# ---------------------------------------------------------------------------

def test_factored_beta1_zero_atan2_smoke():
    """Factored + beta1 == 0 + atan2 (shifter exists, no momentum, atan2 path)."""
    p = torch.nn.Parameter(torch.randn(6, 4, device="cuda"))
    opt = FakeOpt()
    group = _base_group(adam_nnmf_factor=True, adam_betas=(0.0, 0.99), adam_use_atan2=True)
    state = _init(opt, p, group)
    assert state["factored"] is True
    assert "shifter" in state

    before = p.detach().clone()
    for step in (1, 2, 3):
        _run_helper_step(opt, p, group, beta1=0.0, beta2=0.99, step=step)
        assert torch.isfinite(p.detach()).all()
    assert not torch.equal(p.detach(), before)


def test_factored_nesterov_smoke():
    """Factored + nesterov momentum (exercises the lookahead branch)."""
    p = torch.nn.Parameter(torch.randn(10, 6, device="cuda"))
    opt = FakeOpt()
    group = _base_group(
        adam_nnmf_factor=True,
        adam_betas=(0.9, 0.99),
        adam_nesterov=True,
        adam_nesterov_coef=0.9,
    )
    _init(opt, p, group)
    for step in (1, 2, 3):
        _run_helper_step(opt, p, group, beta1=0.9, beta2=0.99, step=step)
        assert torch.isfinite(p.detach()).all()


def test_factored_2nd_bf16_sr_smoke():
    """Non-factored first moment + factored second moment + bf16_sr states."""
    p = torch.nn.Parameter(torch.randn(8, 8, device="cuda", dtype=torch.bfloat16))
    opt = FakeOpt()
    group = _base_group(
        adam_state_precision="bf16_sr",
        adam_factored_2nd=True,
    )
    state = _init(opt, p, group)
    assert state.get("factored_2nd") is True
    assert "mu_v_nmf" in state and "mv_v_nmf" in state and "shifter" in state
    for step in (1, 2, 3):
        _run_helper_step(opt, p, group, beta1=0.9, beta2=0.99, step=step)
        assert torch.isfinite(p.float().detach()).all()


def test_nonfactored_int8_sr_smoke():
    """Non-factored exp_avg/exp_avg_sq stored as int8 with stochastic rounding."""
    p = torch.nn.Parameter(torch.randn(6, 4, device="cuda"))
    opt = FakeOpt()
    group = _base_group(adam_state_precision="int8_sr")
    state = _init(opt, p, group)
    assert "exp_avg" in state and state["exp_avg"].dtype == torch.int8
    assert "exp_avg_sq" in state and state["exp_avg_sq"].dtype == torch.uint8
    for step in (1, 2, 3):
        _run_helper_step(opt, p, group, beta1=0.9, beta2=0.99, step=step)
        assert torch.isfinite(p.detach()).all()


def test_factored_spectral_normalization_smoke():
    """Factored + explicit spectral normalization (scale_update path)."""
    p = torch.nn.Parameter(torch.randn(6, 4, device="cuda"))
    opt = FakeOpt()
    group = _base_group(adam_nnmf_factor=True, adam_spectral_normalization=True)
    state = _init(opt, p, group)
    assert "spectral_u" in state and "spectral_v" in state
    for step in (1, 2, 3):
        _run_helper_step(opt, p, group, beta1=0.9, beta2=0.99, step=step)
        assert torch.isfinite(p.detach()).all()


def test_factored_fisher_wd_smoke():
    """Factored + Fisher weight decay (wd_scaler path)."""
    p = torch.nn.Parameter(torch.randn(6, 4, device="cuda"))
    opt = FakeOpt()
    group = _base_group(adam_nnmf_factor=True, adam_fisher_wd=True, adam_weight_decay=0.1)
    state = _init(opt, p, group)
    assert state.get("wd_scaler") is not None
    for step in (1, 2, 3):
        _run_helper_step(opt, p, group, beta1=0.9, beta2=0.99, step=step)
        assert torch.isfinite(p.detach()).all()


# ---------------------------------------------------------------------------
# Tensor beta2 (per-row Kourkoutas-style) handling
# ---------------------------------------------------------------------------

def test_nonfactored_tensor_beta2_works():
    """
    Non-factored path with a per-row beta2 tensor must work: denom has p.shape,
    so (64, 16) / (64, 1) broadcasts correctly.
    """
    p = torch.nn.Parameter(torch.randn(64, 16, device="cuda"))
    opt = FakeOpt()
    group = _base_group(adam_state_precision="fp32")
    _init(opt, p, group)

    beta2 = torch.full((64, 1), 0.99, device="cuda")
    sqrt_bias_correction2 = (1.0 - beta2 ** 2) ** 0.5  # current_step = 2
    _run_helper_step(opt, p, group, beta1=0.9, beta2=beta2, step=1,
                     sqrt_bias_correction2=sqrt_bias_correction2)
    assert torch.isfinite(p.detach()).all()


def test_factored_tensor_beta2_per_row_bc2_works():
    """
    Factored path with a per-row beta2 tensor must apply the per-row
    sqrt_bias_correction2 in parameter space. For p (64, 16) the effective shape
    is (32, 32), which is NOT broadcast-compatible with the (64, 1) bc2 factor;
    the step must still run without error (regression for the previous
    RuntimeError / silent mis-broadcast).
    """
    p = torch.nn.Parameter(torch.randn(64, 16, device="cuda"))
    opt = FakeOpt()
    group = _base_group(adam_nnmf_factor=True, adam_state_precision="fp32")
    state = _init(opt, p, group)
    d1, d2 = state["effective_shape"]
    assert (d1, d2) != p.shape, "test shape chosen so effective shape != p.shape"

    beta2 = torch.full((64, 1), 0.99, device="cuda")
    sqrt_bias_correction2 = (1.0 - beta2 ** 2) ** 0.5

    before = p.detach().clone()
    for step in (1, 2, 3):
        _run_helper_step(opt, p, group, beta1=0.9, beta2=beta2, step=step,
                         sqrt_bias_correction2=sqrt_bias_correction2)
        assert torch.isfinite(p.detach()).all()
    assert not torch.equal(p.detach(), before)


def test_factored_tensor_beta2_atan2_per_row_works():
    """Factored + per-row beta2 + atan2 update must also broadcast bc2 correctly."""
    p = torch.nn.Parameter(torch.randn(64, 16, device="cuda"))
    opt = FakeOpt()
    group = _base_group(adam_nnmf_factor=True, adam_state_precision="fp32",
                        adam_use_atan2=True)
    _init(opt, p, group)

    beta2 = torch.full((64, 1), 0.95, device="cuda")
    sqrt_bias_correction2 = (1.0 - beta2 ** 2) ** 0.5

    for step in (1, 2, 3):
        _run_helper_step(opt, p, group, beta1=0.9, beta2=beta2, step=step,
                         sqrt_bias_correction2=sqrt_bias_correction2)
        assert torch.isfinite(p.detach()).all()


# ---------------------------------------------------------------------------
# Params with prime numel -> effective shape (numel, 1); factored must still run
# ---------------------------------------------------------------------------

def test_factored_prime_numel_smoke():
    """numel=17 -> effective shape (17, 1); exercises degenerate factorization."""
    p = torch.nn.Parameter(torch.randn(17, device="cuda"))
    opt = FakeOpt()
    group = _base_group(adam_nnmf_factor=True, vector_reshape=True)
    state = _init(opt, p, group)
    assert state["factored"] is True
    for step in (1, 2):
        _run_helper_step(opt, p, group, beta1=0.9, beta2=0.99, step=step)
        assert torch.isfinite(p.detach()).all()
