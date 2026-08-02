"""Tests for layer-marker-aware Muon suitability auto-detection.

Verifies that `_is_suitable_for_muon` honors the layer markers set by network
libraries (PEFT / diffusers / timm-style) when routing parameters between the
Muon and AuxAdam paths of `Muon_adv` / `AdaMuon_adv`:

  - `_is_dora_scale`, `_is_oft`, `is_vector` -> always AdamW (opt-out).
  - `is_hidden=False` -> AdamW (opt-out).
  - `is_hidden=True` -> always Muon for 2D+ tensors without unit leading dims
    (opt-in), bypassing only the `min_dim_size` / `max_aspect_ratio` threshold
    checks; the structural and unit-dim validity rules still apply.
  - `_is_lora_A` / `_is_lora_B` receive no special treatment here and follow
    the other flags plus the regular shape checks.
  - Explicit `use_muon` in the param group overrides all markers.

All tests run on CUDA as mandated by the project conventions.
"""

import os
import sys
import unittest

import torch

# Make the package importable when running from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adv_optm.optim.AdaMuon_adv import AdaMuon_adv  # noqa: E402
from adv_optm.optim.Muon_adv import Muon_adv  # noqa: E402
from adv_optm.util.Muon_util import _is_suitable_for_muon  # noqa: E402

DEVICE = torch.device("cuda:0")
torch.manual_seed(0)


def _make_param(shape, **markers):
    p = torch.nn.Parameter(torch.randn(*shape, device=DEVICE) * 0.1)
    p.grad = torch.randn(*shape, device=DEVICE) * 0.1
    for name, value in markers.items():
        setattr(p, name, value)
    return p


class TestMarkerAwareSuitability(unittest.TestCase):
    """Direct `_is_suitable_for_muon` unit tests for the layer markers."""

    def _check(self, shape, expected, min_dim_size=4, max_aspect_ratio=128.0, **markers):
        p = _make_param(shape, **markers)
        self.assertIs(
            _is_suitable_for_muon(p, min_dim_size, max_aspect_ratio),
            expected,
            f"shape={shape}, min_dim_size={min_dim_size}, "
            f"max_aspect_ratio={max_aspect_ratio}, markers={markers}",
        )

    # --- Opt-out markers: always routed to AdamW ---

    def test_dora_scale_opt_out(self):
        self._check((16, 4), False, _is_dora_scale=True)

    def test_oft_opt_out(self):
        self._check((16, 4), False, _is_oft=True)

    def test_is_vector_opt_out(self):
        self._check((16, 4), False, is_vector=True)

    def test_is_hidden_false_opt_out(self):
        self._check((16, 4), False, is_hidden=False)

    # --- Opt-in: is_hidden=True force-includes 2D+ tensors ---

    def test_is_hidden_true_muon_on_suitable_shape(self):
        self._check((16, 4), True, is_hidden=True)

    def test_is_hidden_true_bypasses_min_dim_size(self):
        # (2, 2) fails min_dim_size=4 but is_hidden=True force-includes it.
        self._check((2, 2), True, min_dim_size=4, is_hidden=True)

    def test_is_hidden_true_bypasses_max_aspect_ratio(self):
        # (4096, 8) has aspect ratio 512 > 2 but is_hidden=True force-includes it.
        self._check((4096, 8), True, max_aspect_ratio=2.0, is_hidden=True)

    def test_unit_dim_rules_still_apply_when_is_hidden_true(self):
        # Rank-degenerate (1, seq, dim): the s[0] == 1 exclusion is a validity
        # rule and stays active even with is_hidden=True.
        self._check((1, 16, 64), False, is_hidden=True)
        # Depthwise conv (out, 1, h, w): the s[1] == 1 exclusion also stays
        # active even with is_hidden=True.
        self._check((16, 1, 8, 8), False, is_hidden=True)

    def test_is_hidden_true_on_1d_rejected(self):
        # 1D tensors keep the structural ndim >= 2 guard (Muon step has no 1D branch).
        self._check((4,), False, is_hidden=True)

    # --- LoRA receives no special treatment ---

    def test_lora_b_falls_through_to_shape_checks(self):
        # (16, 4) passes the default shape checks -> Muon.
        self._check((16, 4), True, _is_lora_B=True)

    def test_lora_a_rejected_by_aspect_ratio(self):
        # (8, 4096): aspect ratio 512 > 128 -> AdamW.
        self._check((8, 4096), False, _is_lora_A=True)

    def test_lora_b_with_is_hidden_true(self):
        self._check((16, 4), True, _is_lora_B=True, is_hidden=True)

    def test_lora_b_with_is_hidden_false(self):
        self._check((16, 4), False, _is_lora_B=True, is_hidden=False)

    # --- Precedence: opt-out markers win over is_hidden=True ---

    def test_dora_scale_wins_over_is_hidden_true(self):
        self._check((16, 4), False, _is_dora_scale=True, is_hidden=True)

    def test_is_vector_wins_over_is_hidden_true(self):
        self._check((16, 4), False, is_vector=True, is_hidden=True)

    # --- No markers: behavior unchanged ---

    def test_no_markers_suitable(self):
        self._check((16, 4), True)

    def test_no_markers_not_suitable(self):
        self._check((1, 16), False)


class TestMarkerRoutingOptimizers(unittest.TestCase):
    """Both optimizers must honor the markers when auto-detecting (use_muon=None)."""

    def _assert_routing(self, opt_cls, shape, expected, **markers):
        p = _make_param(shape, **markers)
        opt = opt_cls([p], use_muon=None)
        self.assertIs(
            opt.state[p]["is_muon"],
            expected,
            f"{opt_cls.__name__} shape={shape} markers={markers}",
        )

    def test_opt_out_markers_route_to_adam(self):
        for opt_cls in (AdaMuon_adv, Muon_adv):
            for markers in (
                {"_is_dora_scale": True},
                {"_is_oft": True},
                {"is_vector": True},
                {"is_hidden": False},
            ):
                with self.subTest(opt_cls=opt_cls.__name__, markers=markers):
                    self._assert_routing(opt_cls, (16, 4), False, **markers)

    def test_is_hidden_true_routes_to_muon(self):
        for opt_cls in (AdaMuon_adv, Muon_adv):
            with self.subTest(opt_cls=opt_cls.__name__):
                # (2, 2) fails min_dim_size but is_hidden=True force-includes it.
                self._assert_routing(opt_cls, (2, 2), True, is_hidden=True)

    def test_unit_dim_with_is_hidden_true_stays_adam(self):
        for opt_cls in (AdaMuon_adv, Muon_adv):
            with self.subTest(opt_cls=opt_cls.__name__):
                # Unit-dim validity rules are not bypassed by is_hidden=True.
                self._assert_routing(opt_cls, (1, 16, 64), False, is_hidden=True)
                self._assert_routing(opt_cls, (16, 1, 8, 8), False, is_hidden=True)

    def test_lora_follows_other_flags(self):
        for opt_cls in (AdaMuon_adv, Muon_adv):
            with self.subTest(opt_cls=opt_cls.__name__):
                self._assert_routing(opt_cls, (16, 4), True, _is_lora_B=True)
                self._assert_routing(
                    opt_cls, (16, 4), False, _is_lora_B=True, is_hidden=False
                )
                self._assert_routing(
                    opt_cls, (16, 4), True, _is_lora_B=True, is_hidden=True
                )

    def test_explicit_use_muon_overrides_markers(self):
        for opt_cls in (AdaMuon_adv, Muon_adv):
            with self.subTest(opt_cls=opt_cls.__name__):
                # use_muon=True wins over the _is_dora_scale opt-out marker.
                p = _make_param((16, 4), _is_dora_scale=True)
                opt = opt_cls([p], use_muon=True)
                self.assertTrue(opt.state[p]["is_muon"])

                # use_muon=False wins over the is_hidden=True opt-in marker.
                p2 = _make_param((16, 4), is_hidden=True)
                opt2 = opt_cls([p2], use_muon=False)
                self.assertFalse(opt2.state[p2]["is_muon"])


class TestMarkerRoutingStep(unittest.TestCase):
    """A full step must run on both routing paths with markers set."""

    def test_step_on_muon_path_with_force_include(self):
        for opt_cls in (AdaMuon_adv, Muon_adv):
            with self.subTest(opt_cls=opt_cls.__name__):
                p = _make_param((2, 2), is_hidden=True)
                opt = opt_cls([p], use_muon=None)
                self.assertTrue(opt.state[p]["is_muon"])
                before = p.detach().clone()
                opt.step()
                self.assertFalse(
                    torch.equal(p.detach(), before), "parameter did not change"
                )

    def test_step_on_adam_path_with_opt_out_marker(self):
        for opt_cls in (AdaMuon_adv, Muon_adv):
            with self.subTest(opt_cls=opt_cls.__name__):
                p = _make_param((16, 4), _is_dora_scale=True)
                opt = opt_cls([p], use_muon=None)
                self.assertFalse(opt.state[p]["is_muon"])
                before = p.detach().clone()
                opt.step()
                self.assertFalse(
                    torch.equal(p.detach(), before), "parameter did not change"
                )

    def test_step_on_muon_path_with_lora_marker(self):
        for opt_cls in (AdaMuon_adv, Muon_adv):
            with self.subTest(opt_cls=opt_cls.__name__):
                p = _make_param((16, 4), _is_lora_B=True)
                opt = opt_cls([p], use_muon=None)
                self.assertTrue(opt.state[p]["is_muon"])
                before = p.detach().clone()
                opt.step()
                self.assertFalse(
                    torch.equal(p.detach(), before), "parameter did not change"
                )


if __name__ == "__main__":
    unittest.main()
