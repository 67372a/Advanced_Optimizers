"""Tests for the `min_dim_size` / `max_aspect_ratio` auto-detection args.

Verifies that both `AdaMuon_adv` and `Muon_adv` expose `min_dim_size` and
`max_aspect_ratio` and plumb them through to `_is_suitable_for_muon` so the
auto-detect routing (when `use_muon=None`) honors the caller's thresholds.

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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


def _make_param(shape):
    p = torch.nn.Parameter(torch.randn(*shape, device=DEVICE) * 0.1)
    p.grad = torch.randn(*shape, device=DEVICE) * 0.1
    return p


class TestMuonSuitabilityArgs(unittest.TestCase):
    """The new args must be exposed by both optimizers and stored in groups."""

    def test_args_accepted_and_stored_in_param_group(self):
        for opt_cls in (AdaMuon_adv, Muon_adv):
            with self.subTest(opt_cls=opt_cls.__name__):
                p = _make_param((16, 4))
                opt = opt_cls([p], min_dim_size=8, max_aspect_ratio=64.0)
                self.assertEqual(opt.param_groups[0]["min_dim_size"], 8)
                self.assertEqual(opt.param_groups[0]["max_aspect_ratio"], 64.0)

    def test_defaults_match_muon_util(self):
        for opt_cls in (AdaMuon_adv, Muon_adv):
            with self.subTest(opt_cls=opt_cls.__name__):
                p = _make_param((16, 4))
                opt = opt_cls([p])
                self.assertEqual(opt.param_groups[0]["min_dim_size"], 4)
                self.assertEqual(opt.param_groups[0]["max_aspect_ratio"], 128.0)

    def test_param_group_level_override(self):
        p = _make_param((16, 4))
        opt = AdaMuon_adv([{"params": [p], "min_dim_size": 8}])
        self.assertEqual(opt.param_groups[0]["min_dim_size"], 8)
        # 16x4 has min dim 4 < 8 -> routed to Adam path.
        self.assertFalse(opt.state[p]["is_muon"])


class TestAutoDetectRouting(unittest.TestCase):
    """`_is_suitable_for_muon` must receive the exposed thresholds."""

    def _assert_routing(self, opt_cls, shape, min_dim_size, max_aspect_ratio, expected):
        p = _make_param(shape)
        opt = opt_cls(
            [p],
            use_muon=None,  # force auto-detect
            min_dim_size=min_dim_size,
            max_aspect_ratio=max_aspect_ratio,
        )
        self.assertIs(opt.state[p]["is_muon"], expected)

    def test_default_thresholds_route_16x4_to_muon(self):
        for opt_cls in (AdaMuon_adv, Muon_adv):
            with self.subTest(opt_cls=opt_cls.__name__):
                self._assert_routing(opt_cls, (16, 4), 4, 128.0, True)

    def test_min_dim_size_filters(self):
        for opt_cls in (AdaMuon_adv, Muon_adv):
            with self.subTest(opt_cls=opt_cls.__name__):
                # min(16, 4) == 4 < 8 -> not suitable.
                self._assert_routing(opt_cls, (16, 4), 8, 128.0, False)

    def test_max_aspect_ratio_filters(self):
        for opt_cls in (AdaMuon_adv, Muon_adv):
            with self.subTest(opt_cls=opt_cls.__name__):
                # aspect ratio 16 / 4 == 4 > 2 -> not suitable.
                self._assert_routing(opt_cls, (16, 4), 4, 2.0, False)
                # aspect ratio exactly at the bound -> suitable.
                self._assert_routing(opt_cls, (16, 4), 4, 4.0, True)

    def test_unit_dimension_always_not_suitable(self):
        for opt_cls in (AdaMuon_adv, Muon_adv):
            with self.subTest(opt_cls=opt_cls.__name__):
                # s[0] == 1 fails before any threshold check.
                self._assert_routing(opt_cls, (1, 16), 1, 1e6, False)


class TestFunctionalStep(unittest.TestCase):
    """A full optimizer step must run with the new args on both routing paths."""

    def test_step_runs_with_min_dim_size_filter(self):
        for opt_cls in (AdaMuon_adv, Muon_adv):
            with self.subTest(opt_cls=opt_cls.__name__):
                p = _make_param((16, 4))
                opt = opt_cls(
                    [p],
                    use_muon=None,
                    min_dim_size=8,  # routes (16, 4) to the Adam path
                    max_aspect_ratio=128.0,
                )
                self.assertFalse(opt.state[p]["is_muon"])
                before = p.detach().clone()
                opt.step()
                self.assertFalse(torch.equal(p.detach(), before), "parameter did not change")

    def test_step_runs_with_max_aspect_ratio_filter(self):
        for opt_cls in (AdaMuon_adv, Muon_adv):
            with self.subTest(opt_cls=opt_cls.__name__):
                p = _make_param((16, 4))
                opt = opt_cls(
                    [p],
                    use_muon=None,
                    min_dim_size=4,
                    max_aspect_ratio=2.0,  # routes (16, 4) to the Adam path
                )
                self.assertFalse(opt.state[p]["is_muon"])
                before = p.detach().clone()
                opt.step()
                self.assertFalse(torch.equal(p.detach(), before), "parameter did not change")

    def test_step_runs_on_muon_path_with_defaults(self):
        for opt_cls in (AdaMuon_adv, Muon_adv):
            with self.subTest(opt_cls=opt_cls.__name__):
                p = _make_param((16, 4))
                opt = opt_cls(
                    [p],
                    use_muon=None,
                    min_dim_size=4,
                    max_aspect_ratio=128.0,  # (16, 4) stays on the Muon path
                )
                self.assertTrue(opt.state[p]["is_muon"])
                before = p.detach().clone()
                opt.step()
                self.assertFalse(torch.equal(p.detach(), before), "parameter did not change")


if __name__ == "__main__":
    unittest.main()
