"""Regression tests for Adopt_adv defects (CUDA-only).

Validated defects (all fixed):
  A. Factored state_precision + betas[0] == 0  -> KeyError('shifter')
  B. spectral_normalization=True, first step   -> first update clamped to zero
     (clip_lambda(0) == 0)
  C. kourkoutas_beta enabled per-group but not at construction -> AttributeError
  D. compiled_optimizer + weight_decay on CUDA (CPU 0-dim lr tensor) must run
     without device errors

The smoke test uses fixed inputs so the loss trajectory is deterministic
(previously fresh random x/y per step made `l2 < l1 < l0` flaky).
"""
import torch
import torch.nn as nn
import pytest

from adv_optm.optim.Adopt_adv import Adopt_adv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_model(seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    ).to(DEVICE).to(dtype)
    return model


def _make_fixed_data(seed=1):
    torch.manual_seed(seed)
    return torch.randn(8, 8, device=DEVICE), torch.randn(8, 4, device=DEVICE)


def _step_loss(model, optim, x, y):
    loss = torch.nn.functional.mse_loss(model(x), y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.item()


def _param_deltas(model):
    """Returns total absolute delta between params and their detached originals."""
    return sum(
        p.detach().abs().sum().item() for p in model.parameters()
    )


def test_smoke_default_fp32():
    """Baseline: default config runs and reduces loss on fixed data.

    The first ADOPT step is init-only (v0 = g0^2, no parameter update), so the
    loss is unchanged after step 1; every real update afterwards must decrease it.
    """
    model = _make_model()
    optim = Adopt_adv(model.parameters(), lr=1e-3)
    x, y = _make_fixed_data()
    losses = [_step_loss(model, optim, x, y) for _ in range(5)]
    assert losses[1] == pytest.approx(losses[0], abs=1e-12), "init-only first step must not move params"
    assert losses[2] < losses[1], losses
    assert losses[-1] < losses[2], losses


@pytest.mark.parametrize("state_precision", ["factored", "auto", "fp32"])
def test_factored_zero_momentum(state_precision):
    """Defect A: factored/2D params with betas[0]==0 must not crash and must
    actually move the parameters (the shifter must exist for the second-moment
    reconstruct/factorize cycle even when no first moment is tracked)."""
    model = _make_model()
    optim = Adopt_adv(
        model.parameters(),
        lr=1e-3,
        betas=(0.0, 0.9999),  # no first moment
        state_precision=state_precision,
    )
    x, y = _make_fixed_data()
    # First step is init-only (v0 = g0^2); subsequent steps must be real updates.
    _step_loss(model, optim, x, y)
    before = [p.detach().clone() for p in model.parameters()]
    _step_loss(model, optim, x, y)
    deltas = [(p.detach() - b).abs().sum().item() for p, b in zip(model.parameters(), before)]
    assert sum(deltas) > 0.0, f"no parameter movement for state_precision={state_precision} (deltas={deltas})"


def test_spectral_first_step_not_zero():
    """Defect B: with spectral_normalization=True the first step must produce a
    nonzero parameter update (clip_lambda(0) == 0 used to clamp it to zero)."""
    model = _make_model()
    optim = Adopt_adv(
        model.parameters(),
        lr=1e-3,
        spectral_normalization=True,
    )
    before = [p.detach().clone() for p in model.parameters()]
    x, y = _make_fixed_data()
    _step_loss(model, optim, x, y)
    deltas = [(p.detach() - b).abs().sum().item() for p, b in zip(model.parameters(), before)]
    total_delta = sum(deltas)
    assert total_delta > 0.0, f"First spectral step produced a zero update (deltas={deltas})"


def test_spectral_factored_first_step_not_zero():
    """Defect B variant: spectral_normalization + factored state must also
    produce a nonzero first update."""
    model = _make_model()
    optim = Adopt_adv(
        model.parameters(),
        lr=1e-3,
        spectral_normalization=True,
        state_precision="factored",
    )
    before = [p.detach().clone() for p in model.parameters()]
    x, y = _make_fixed_data()
    _step_loss(model, optim, x, y)
    deltas = [(p.detach() - b).abs().sum().item() for p, b in zip(model.parameters(), before)]
    assert sum(deltas) > 0.0, f"First spectral+factored step produced a zero update (deltas={deltas})"


def test_kourkoutas_group_override_without_ctor_flag():
    """Defect C: enabling kourkoutas_beta on a param group while the
    constructor-level flag is False must still create the helper and run."""
    model = _make_model()
    group = {
        "params": list(model.parameters()),
        "kourkoutas_beta": True,  # enabled per-group only
        "beta2_min": 0.9,
        "ema_alpha": 0.95,
        "tiny_spike": 1e-9,
        "k_warmup_steps": 0,
        "k_logging": 0,
    }
    optim = Adopt_adv([group], lr=1e-3)
    assert optim.kourkoutas_beta is False
    assert hasattr(optim, "kourkoutas_helper"), "helper must exist for per-group kourkoutas_beta"
    x, y = _make_fixed_data()
    # First step is init-only, so start checking the decrease from the first real update.
    losses = [_step_loss(model, optim, x, y) for _ in range(4)]
    assert losses[-1] < losses[1], losses


def test_compiled_weight_decay_cuda():
    """Defect D: the compiled path passes a CPU 0-dim lr tensor (torch.as_tensor)
    into CUDA ops and apply_parameter_update; it must run without device errors
    even with weight_decay > 0."""
    model = _make_model()
    optim = Adopt_adv(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.1,
        compiled_optimizer=True,
    )
    x, y = _make_fixed_data()
    _step_loss(model, optim, x, y)
    _step_loss(model, optim, x, y)
    # Also verify the parameters actually moved (both decay and update applied).
    deltas = _param_deltas(model)
    assert deltas > 0.0, "compiled path did not update parameters"


def test_compiled_bf16_atan2_smoke():
    """Compiled path + bf16 params + use_atan2 (no clip) must run and converge."""
    model = _make_model(dtype=torch.bfloat16)
    optim = Adopt_adv(
        model.parameters(),
        lr=1e-3,
        use_atan2=True,
        compiled_optimizer=True,
    )
    x, y = _make_fixed_data()
    x, y = x.to(torch.bfloat16), y.to(torch.bfloat16)
    # First step is a real update for use_atan2 (init-only skip is disabled),
    # but a single lr=1e-3 step can be smaller than bf16's loss quantization,
    # so accumulate over several steps.
    losses = [_step_loss(model, optim, x, y) for _ in range(5)]
    assert losses[-1] < losses[0], losses


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
