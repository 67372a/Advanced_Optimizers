"""Review tests for Prodigy_adv.

These tests empirically confirm suspected defects found during a code review of
``adv_optm/optim/Prodigy_adv.py``:

1. ``KeyError: 'shifter'`` when ``state_precision='factored'`` (or the legacy
   ``nnmf_factor=True``) is combined with ``betas[0] == 0``.
2. ``calculate_d`` never assigns ``d = d_hat`` for ``d != d0``, so with a finite
   ``growth_rate`` the adaptive step size ``d`` monotonically ratchets up from
   the running max of ``d_hat`` instead of tracking ``d_hat`` (deviates from the
   reference Prodigy update ``d = max(d, d_hat) if d == d0 else d_hat``).
3. Per-group ``betas`` are ignored: ``init_step`` reads ``beta1``/``beta2``/
   ``beta3`` only from ``param_groups[0]`` and ``_step_parameter`` uses the
   instance-level ``self.beta1``/``self.beta2_default``/``self.beta3`` for every
   parameter, so parameters in other groups get group-0 momentum decay.

Plus smoke tests for the main feature combinations (CUDA only, per project
conventions).
"""

import math

import pytest
import torch

from adv_optm.optim import Prodigy_adv

DEVICE = torch.device("cuda:0")


def make_param(shape=(16, 16), dtype=torch.float32, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    p = torch.nn.Parameter(torch.randn(*shape, device=DEVICE, dtype=dtype, generator=g) * 0.1)
    p.grad = torch.randn(*shape, device=DEVICE, dtype=dtype, generator=g) * 0.1
    return p


# ---------------------------------------------------------------------------
# Smoke tests: every major feature combination must complete a step cleanly.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"nnmf_factor": True},
        {"state_precision": "factored"},
        {"state_precision": "bf16_sr"},
        {"state_precision": "fp32"},
        {"state_precision": "int8_sr"},
        {"use_atan2": True},
        {"kourkoutas_beta": True},
        {"orthogonal_gradient": "flattened"},
        {"orthogonal_gradient": "iterative"},
        {"nesterov": True},
        {"fisher_wd": True},
        {"cautious_wd": True, "weight_decay": 0.01},
        {"centered_wd": 0.01},
        {"spectral_normalization": True},
        {"compiled_optimizer": True},
        {"safeguard_warmup": True},
        {"d_limiter": True, "growth_rate": 1.02},
        {"prodigy_steps": 3},
        {"factored_2nd": True},
        {"factored_2nd": True, "use_atan2": True},
        {"nnmf_factor": True, "use_atan2": True},
        {"kourkoutas_beta": True, "compiled_optimizer": True},
    ],
    ids=lambda kw: ",".join(f"{k}={v}" for k, v in kw.items()) or "default",
)
def test_smoke_step(kwargs):
    p = make_param()
    opt = Prodigy_adv([p], lr=1e-2, **kwargs)
    opt.step()
    assert p.isfinite().all(), "parameter became non-finite"
    assert opt.state[p]["step"] == 1


def test_smoke_bf16_param():
    p = make_param(dtype=torch.bfloat16)
    opt = Prodigy_adv([p], lr=1e-2, stochastic_rounding=True)
    opt.step()
    assert p.isfinite().all()
    assert p.dtype == torch.bfloat16


def test_smoke_zero_beta1_standard_states():
    # Standard (non-factored) states should work with betas[0] == 0.
    p = make_param()
    opt = Prodigy_adv([p], betas=(0.0, 0.999), lr=1e-2)
    opt.step()
    assert p.isfinite().all()


def test_smoke_multiple_steps_d_grows():
    """With an aligned (constant-sign) gradient, Prodigy's d must grow past d0."""
    p = make_param(seed=1)
    g = torch.ones_like(p) * 0.1
    opt = Prodigy_adv([p], lr=1.0)
    for _ in range(8):
        p.grad = g
        opt.step()
    assert opt.param_groups[0]["d"] > opt.param_groups[0]["d0"]
    assert opt.param_groups[0]["k"] == 8


# ---------------------------------------------------------------------------
# Fix 1: factored states + betas[0] == 0 (previously KeyError('shifter'))
# ---------------------------------------------------------------------------

def test_factored_with_zero_beta1_runs():
    """'shifter' must be initialized unconditionally in the factored branch so
    that the second-moment reconstruction/factorization works even when
    betas[0] == 0 (no first moment)."""
    p = make_param()
    opt = Prodigy_adv([p], betas=(0.0, 0.999), nnmf_factor=True)
    opt.step()
    assert p.isfinite().all()
    assert "shifter" in opt.state[p]
    assert opt.state[p]["step"] == 1


# ---------------------------------------------------------------------------
# Fix 2: calculate_d now tracks d_hat for d != d0
# ---------------------------------------------------------------------------

def test_calculate_d_tracks_d_hat_downward():
    """Reference Prodigy update::

        d_hat = d_coef * d_numerator / d_denom
        d = max(d, d_hat) if d == d0 else d_hat
        d_max = max(d_max, d_hat)
        d = min(d_max, d * growth_rate)

    With growth_rate=1.02:
      step 1, d_hat=10  -> d = min(10, 10*1.02)   = 10.0
      step 2, d_hat=0.1 -> d = min(10, 0.1*1.02)  = 0.102

    d must shrink when d_hat collapses (previously it ratcheted at the running
    max of d_hat and never went down).
    """
    p = make_param()
    opt = Prodigy_adv([p], lr=1.0, growth_rate=1.02)

    g = opt.param_groups[0]
    assert g["d"] == g["d0"] == 1e-6

    # Step 1: strong gradient signal -> d_hat = 1.0 * 100 / 10 = 10
    opt.d_numerator = torch.tensor(100.0, device=DEVICE)
    opt.d_denom = torch.tensor(10.0, device=DEVICE)
    opt.calculate_d()
    assert g["d"] == pytest.approx(10.0)
    assert g["k"] == 1

    # Step 2: weak gradient signal -> d_hat = 1.0 * 1 / 10 = 0.1
    opt.d_numerator = torch.tensor(1.0, device=DEVICE)
    opt.d_denom = torch.tensor(10.0, device=DEVICE)
    opt.calculate_d()

    assert g["d"] == pytest.approx(0.1 * 1.02)
    assert g["d_max"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Fix 3: per-group betas are honored
# ---------------------------------------------------------------------------

def test_per_group_betas_honored():
    """Each parameter group must use its own beta1 (previously every group was
    decayed with param_groups[0]'s beta1)."""
    p1 = make_param(seed=1)
    p2 = make_param(seed=2)
    opt = Prodigy_adv(
        [
            {"params": [p1], "betas": (0.1, 0.9), "lr": 1.0},
            {"params": [p2], "betas": (0.9, 0.9), "lr": 1.0},
        ],
        d0=1e-6,
    )
    opt.step()

    d0 = 1e-6
    # exp_avg = d0 * (1 - beta1) * grad  (first step, d == d0)
    expected_p1 = d0 * (1.0 - 0.1) * p1.grad
    expected_p2 = d0 * (1.0 - 0.9) * p2.grad
    torch.testing.assert_close(opt.state[p1]["exp_avg"], expected_p1, atol=1e-9, rtol=1e-6)
    torch.testing.assert_close(opt.state[p2]["exp_avg"], expected_p2, atol=1e-9, rtol=1e-6)
    assert not torch.allclose(opt.state[p2]["exp_avg"], expected_p1, atol=1e-9)


# ---------------------------------------------------------------------------
# Additional observations
# ---------------------------------------------------------------------------

def test_zero_d0_division_by_zero():
    """d0 == 0 is not validated; the accumulation divides by d0."""
    p = make_param()
    with pytest.raises(ZeroDivisionError):
        opt = Prodigy_adv([p], d0=0.0)
        opt.step()


def test_slice_p_zero_raises():
    """slice_p == 0 is not validated; p.flatten()[::0] raises ValueError."""
    p = make_param()
    with pytest.raises(ValueError):
        opt = Prodigy_adv([p], slice_p=0)
        opt.step()


def test_load_state_dict_into_fresh_optimizer():
    """Checkpoint produced by a stepped optimizer must load into a fresh
    Prodigy_adv and keep stepping (group runtime keys such as
    'actual_state_precision' are persisted via param_groups)."""
    p1 = make_param(seed=1)
    opt1 = Prodigy_adv([p1], lr=1e-2)
    opt1.step()
    state_dict = opt1.state_dict()

    p2 = make_param(seed=2)
    opt2 = Prodigy_adv([p2], lr=1e-2)
    opt2.load_state_dict(state_dict)
    p2.grad = torch.randn_like(p2) * 0.1
    opt2.step()
    assert p2.isfinite().all()
    assert opt2.state[p2]["step"] == 2


def test_beta3_default_is_sqrt_beta2():
    p = make_param()
    opt = Prodigy_adv([p], betas=(0.9, 0.999))
    assert opt.beta3 == pytest.approx(math.sqrt(0.999))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
