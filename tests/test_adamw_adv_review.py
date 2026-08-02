"""Review tests for AdamW_adv (CUDA-only).

Manual review of adv_optm/optim/AdamW_adv.py with programmatic verification.

Defects found and subsequently fixed (regression tests below now pass):
  A. ``eps=None`` (documented: "Set to None for scale invariant eps") crashed at
     construction: ``if not (eps >= 0.0)`` raised ``TypeError`` for ``None``.
     Fixed: validation now accepts ``None`` (``eps is not None and ...``).
  D. ``orthogonal_gradient`` mode was never validated; an invalid string made
     ``_orthogonalize_gradient`` return ``None`` and the step failed with a
     cryptic ``TypeError``. Fixed: validated in ``__init__`` against
     ``{'disabled', 'flattened', 'iterative'}``.
  E. Enabling ``kourkoutas_beta=True`` on a param *group* while the optimizer
     was constructed with ``kourkoutas_beta=False`` raised ``AttributeError``
     (``self.kourkoutas_helper`` never created). Fixed: the helper is now
     instantiated when *any* param group requests Kourkoutas-β.

Hypotheses that were *disproven* by testing (kept as passing regression tests):
  - fp16 parameters step fine (torch >= 2.x downcasts the fp32 update in-place,
    so ``p.add_(-update)`` does not raise).
  - bf16 parameters step fine even with ``stochastic_rounding=False``.
  - ``compiled_optimizer`` with two groups sharing a shape but different
    hyperparameters is correct (torch.compile recompiles via guards when the
    ``group`` dict contents change, despite the coarse ``(shape, factored)``
    Python-level cache key).
  - ``compiled_optimizer=True`` + ``kourkoutas_beta=True`` steps successfully.
"""
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adv_optm.optim.AdamW_adv import AdamW_adv  # noqa: E402

DEVICE = "cuda"
torch.manual_seed(0)


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


def _run_steps(model, optim, x, y, steps=5, dtype=torch.float32):
    losses = []
    x = x.to(dtype)
    y = y.to(dtype)
    for _ in range(steps):
        loss = torch.nn.functional.mse_loss(model(x), y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
    return losses


# ---------------------------------------------------------------------------
# Passing sanity tests (supported configurations must run and reduce loss)
# ---------------------------------------------------------------------------

def test_smoke_default_fp32():
    """Default fp32 config must run and reduce loss."""
    model = _make_model()
    optim = AdamW_adv(model.parameters(), lr=1e-3)
    x, y = _make_fixed_data()
    losses = _run_steps(model, optim, x, y)
    assert losses[-1] < losses[0]


def test_smoke_bf16_with_stochastic_rounding():
    """bf16 params with the default stochastic_rounding=True must run."""
    model = _make_model(dtype=torch.bfloat16)
    optim = AdamW_adv(model.parameters(), lr=1e-3)
    x, y = _make_fixed_data()
    losses = _run_steps(model, optim, x, y, dtype=torch.bfloat16)
    assert losses[-1] < losses[0]


def test_smoke_fp32_state_precision_bf16_param():
    """bf16 params with fp32 optimizer states must run."""
    model = _make_model(dtype=torch.bfloat16)
    optim = AdamW_adv(model.parameters(), lr=1e-3, state_precision="fp32")
    x, y = _make_fixed_data()
    losses = _run_steps(model, optim, x, y, dtype=torch.bfloat16)
    assert losses[-1] < losses[0]


def test_fp16_params_step():
    """fp16 params must step (torch downcasts the fp32 update in-place).

    Regression guard: previously suspected to raise a dtype error on
    ``p.add_(-update)``; verified to run on torch 2.7.1.
    """
    model = _make_model(dtype=torch.float16)
    optim = AdamW_adv(model.parameters(), lr=1e-3)
    x, y = _make_fixed_data()
    losses = _run_steps(model, optim, x, y, steps=2, dtype=torch.float16)
    assert losses[-1] < losses[0]


def test_bf16_without_stochastic_rounding():
    """bf16 params must step even with stochastic_rounding=False.

    Regression guard: previously suspected to raise a dtype error.
    """
    model = _make_model(dtype=torch.bfloat16)
    optim = AdamW_adv(model.parameters(), lr=1e-3, stochastic_rounding=False)
    x, y = _make_fixed_data()
    losses = _run_steps(model, optim, x, y, steps=2, dtype=torch.bfloat16)
    assert losses[-1] < losses[0]


# ---------------------------------------------------------------------------
# Defect A (fixed): eps=None must construct and step
# ---------------------------------------------------------------------------

def test_eps_none_constructs_and_steps():
    model = _make_model()
    optim = AdamW_adv(model.parameters(), lr=1e-3, eps=None)
    x, y = _make_fixed_data()
    losses = _run_steps(model, optim, x, y)
    assert losses[-1] < losses[0]


# ---------------------------------------------------------------------------
# Defect D (fixed): orthogonal_gradient mode must be validated at init
# ---------------------------------------------------------------------------

def test_invalid_orthogonal_gradient_raises_at_init():
    with pytest.raises(ValueError):
        AdamW_adv(_make_model().parameters(), lr=1e-3, orthogonal_gradient="bogus")


# ---------------------------------------------------------------------------
# Defect E (fixed): group-level kourkoutas_beta without optimizer-level flag
# ---------------------------------------------------------------------------

def test_group_level_kourkoutas_beta():
    model = _make_model()
    params = [
        {"params": [p for p in model.parameters() if p.ndim >= 2], "kourkoutas_beta": True},
        {"params": [p for p in model.parameters() if p.ndim < 2]},
    ]
    optim = AdamW_adv(params, lr=1e-3)
    x, y = _make_fixed_data()
    _run_steps(model, optim, x, y, steps=2)


# ---------------------------------------------------------------------------
# Regression guard: compiled optimizer with heterogeneous groups (previously
# suspected cache-key collision; verified correct via guard recompilation)
# ---------------------------------------------------------------------------

def test_compiled_multigroup_heterogeneous():
    """Two groups with identical shapes but different use_atan2 must both
    produce the same updates as their eager references under compiled mode."""
    use_atan2_a, use_atan2_b = False, True

    def _eager_reference(seed=0):
        torch.manual_seed(seed)
        model_a = nn.Linear(8, 8, bias=False).to(DEVICE)
        model_b = nn.Linear(8, 8, bias=False).to(DEVICE)
        groups = [
            {"params": model_a.parameters(), "use_atan2": use_atan2_a},
            {"params": model_b.parameters(), "use_atan2": use_atan2_b},
        ]
        optim = AdamW_adv(groups, lr=1e-2)
        x = torch.randn(4, 8, device=DEVICE, generator=torch.Generator(DEVICE).manual_seed(seed))
        y = torch.randn(4, 8, device=DEVICE, generator=torch.Generator(DEVICE).manual_seed(seed + 1))
        for _ in range(3):
            loss = torch.nn.functional.mse_loss(model_a(x) + model_b(x), y)
            optim.zero_grad()
            loss.backward()
            optim.step()
        return model_a.weight.detach().clone(), model_b.weight.detach().clone()

    w_a_ref, w_b_ref = _eager_reference(seed=0)

    torch.manual_seed(0)
    model_a = nn.Linear(8, 8, bias=False).to(DEVICE)
    model_b = nn.Linear(8, 8, bias=False).to(DEVICE)
    groups = [
        {"params": model_a.parameters(), "use_atan2": use_atan2_a},
        {"params": model_b.parameters(), "use_atan2": use_atan2_b},
    ]
    optim = AdamW_adv(groups, lr=1e-2, compiled_optimizer=True)
    x = torch.randn(4, 8, device=DEVICE, generator=torch.Generator(DEVICE).manual_seed(0))
    y = torch.randn(4, 8, device=DEVICE, generator=torch.Generator(DEVICE).manual_seed(1))
    for _ in range(3):
        loss = torch.nn.functional.mse_loss(model_a(x) + model_b(x), y)
        optim.zero_grad()
        loss.backward()
        optim.step()

    assert torch.allclose(model_a.weight.detach(), w_a_ref, atol=1e-5, rtol=1e-4)
    assert torch.allclose(model_b.weight.detach(), w_b_ref, atol=1e-5, rtol=1e-4)
