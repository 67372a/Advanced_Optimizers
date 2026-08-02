"""Unit and functional tests for the AdaMuon_adv optimization/correctness fixes.

Covers the issues identified during code review of `adv_optm/optim/AdaMuon_adv.py`:

1. `_adam_group_config()` omitted hyperparameters that are read inside the
   (optionally compiled) AuxAdam step (`adam_eps`, `centered_wd`, `cautious_wd`,
   `centered_wd_mode`), so groups sharing a shape but differing in those values
   could silently reuse a stale compiled graph.
2. Kourkoutas-β gradient-norm accumulation ran inside the torch.compiled
   `_adam_step_parameter` region, where Python dict side effects and `id(p)`
   layer keys do not survive tracing; it now runs in the (non-compiled)
   `step_parameter` and persists.
3. MARS-M reused the stochastic-rounding random tensor also consumed by
   `set_state` on the compiled path, correlating the `last_grad` rounding noise;
   a dedicated random tensor is now generated for the MARS write.
4. `lr` / `step_size` scalars were created as CPU tensors and fed into compiled
   CUDA graphs, forcing a host-device transfer per step; they are now created on
   the parameter's device.
5. `step_parameter()` re-invoked `__init_state()` on every step; state is now
   initialized lazily (once per parameter).
6. NorMuon received the raw `group['eps']` while every other branch uses the
   scale-invariant `adaptive_eps`; it now receives `adaptive_eps`.

All tests run on CUDA as mandated by the project conventions.
"""

import math
import os
import sys
import unittest

import torch

# Make the package importable when running from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adv_optm.optim.AdaMuon_adv import AdaMuon_adv, _adam_group_config  # noqa: E402

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


def make_param(shape, dtype=torch.float32):
    p = torch.nn.Parameter(torch.randn(*shape, device=DEVICE, dtype=dtype) * 0.1)
    p.grad = torch.randn(*shape, device=DEVICE, dtype=dtype) * 0.1
    return p


def make_adam_group(**overrides):
    g = dict(
        lr=1e-3,
        adam_state_precision="auto",
        adam_betas=(0.9, 0.99),
        adam_weight_decay=0.0,
        adam_use_bias_correction=True,
        adam_fisher_wd=False,
        adam_use_atan2=False,
        adam_orthogonal_gradient="disabled",
        adam_nesterov=False,
        adam_nesterov_coef=None,
        adam_spectral_normalization=False,
        adam_kourkoutas_beta=False,
        adam_nnmf_factor=False,
        adam_factored_2nd=False,
        adam_eps=1e-8,
        centered_wd=0.0,
        cautious_wd=False,
        centered_wd_mode="float8",
    )
    g.update(overrides)
    return g


class TestAdamGroupConfigCompleteness(unittest.TestCase):
    """_adam_group_config() must capture every hyperparameter read inside the
    compiled AuxAdam step so the torch.compile cache key is unique per config."""

    def test_config_changes_with_adam_eps(self):
        self.assertNotEqual(
            _adam_group_config(make_adam_group(adam_eps=1e-8)),
            _adam_group_config(make_adam_group(adam_eps=1e-4)),
        )

    def test_config_changes_with_centered_wd(self):
        self.assertNotEqual(
            _adam_group_config(make_adam_group(centered_wd=0.0)),
            _adam_group_config(make_adam_group(centered_wd=0.01)),
        )

    def test_config_changes_with_cautious_wd(self):
        self.assertNotEqual(
            _adam_group_config(make_adam_group(cautious_wd=False)),
            _adam_group_config(make_adam_group(cautious_wd=True)),
        )

    def test_config_changes_with_centered_wd_mode(self):
        self.assertNotEqual(
            _adam_group_config(make_adam_group(centered_wd_mode="float8")),
            _adam_group_config(make_adam_group(centered_wd_mode="full")),
        )

    def test_config_contains_adam_eps_and_mode_values(self):
        cfg = _adam_group_config(make_adam_group())
        self.assertIn(1e-8, cfg)
        self.assertIn("float8", cfg)

    def test_compiled_cache_keys_differ_on_adam_eps(self):
        """Two adam groups sharing a shape but differing in adam_eps must get
        separate compiled graphs (no stale-graph reuse)."""
        p1 = make_param((8, 8))
        p2 = make_param((8, 8))
        opt = AdaMuon_adv(
            [
                {"params": [p1], "use_muon": False, "adam_eps": 1e-8},
                {"params": [p2], "use_muon": False, "adam_eps": 1e-4},
            ],
            compiled_optimizer=True,
        )
        opt.step()
        self.assertEqual(len(opt._compiled_adam_step_fns), 2)


class TestKourkoutasAccumulation(unittest.TestCase):
    """Kourkoutas-β norm accumulation must run outside the compiled region and
    persist in the helper's Python state."""

    def _assert_accumulated(self, opt):
        # Plain params bucket by tuple(shape); see KourkoutasHelper default key fn.
        key = (8, 8)
        self.assertIn(key, opt.kourkoutas_helper.layer_state)
        acc = opt.kourkoutas_helper.layer_state[key]["sum_sq_accumulator"]
        self.assertGreater(acc.abs().sum().item(), 0.0)

    def test_accumulation_persists_uncompiled(self):
        p = make_param((8, 8))
        opt = AdaMuon_adv(
            [p],
            use_muon=False,
            adam_kourkoutas_beta=True,
            adam_betas=(0.9, 0.99),
        )
        opt.step()
        self._assert_accumulated(opt)

    def test_accumulation_persists_compiled(self):
        p = make_param((8, 8))
        opt = AdaMuon_adv(
            [p],
            use_muon=False,
            adam_kourkoutas_beta=True,
            adam_betas=(0.9, 0.99),
            compiled_optimizer=True,
        )
        for _ in range(2):
            opt.step()
        self._assert_accumulated(opt)

    def test_muon_adv_accumulation_persists(self):
        """Regression guard: Muon_AuxAdam._adam_step_parameter no longer
        accumulates, so Muon_adv's step_parameter must do it."""
        from adv_optm.optim.Muon_adv import Muon_adv

        p = make_param((8, 8))
        opt = Muon_adv(
            [p],
            use_muon=False,
            adam_kourkoutas_beta=True,
            adam_betas=(0.9, 0.99),
        )
        opt.step()
        self._assert_accumulated(opt)


class TestMarsDedicatedRandomTensor(unittest.TestCase):
    """On the compiled path the MARS last_grad write must get a dedicated
    stochastic-rounding tensor, not the one consumed by set_state."""

    def test_two_random_draws_when_compiled_bf16_mars(self):
        import adv_optm.util.param_update as pu

        real = pu._get_random_int_for_sr
        draws = []

        def counting(source, *args, **kwargs):
            draws.append(source)
            return real(source, *args, **kwargs)

        pu._get_random_int_for_sr = counting
        try:
            p = make_param((8, 16), dtype=torch.bfloat16)
            opt = AdaMuon_adv(
                [p],
                use_muon=True,
                approx_mars=True,
                mars_gamma=0.025,
                state_precision="bf16_sr",
                compiled_optimizer=True,
                stochastic_rounding=True,
            )
            opt.step()
            # Draw 1: random_int_tensor for the parameter update / state writes.
            # Draw 2: dedicated mars_random_tensor for the last_grad write.
            self.assertGreaterEqual(len(draws), 2)
        finally:
            pu._get_random_int_for_sr = real


class TestScalarsOnParamDevice(unittest.TestCase):
    """lr / step_size must be created on the parameter's device on the compiled
    path so they do not force a host-device transfer per step."""

    def test_lr_and_step_size_device_matches_param(self):
        pa = make_param((8, 8))   # adam path -> step_size
        pm = make_param((8, 16))  # muon path -> lr
        opt = AdaMuon_adv(
            [
                {"params": [pa], "use_muon": False},
                {"params": [pm], "use_muon": True},
            ],
            compiled_optimizer=True,
        )
        # Warm up so the compiled graphs are cached before we patch as_tensor.
        opt.step()
        opt.step()

        real_as_tensor = torch.as_tensor
        seen = []

        def patched(obj, *args, **kwargs):
            if isinstance(obj, (int, float)) and "device" in kwargs:
                seen.append(kwargs["device"])
            return real_as_tensor(obj, *args, **kwargs)

        torch.as_tensor = patched
        try:
            opt.step()
        finally:
            torch.as_tensor = real_as_tensor

        self.assertGreaterEqual(len(seen), 2)
        for device in seen:
            self.assertEqual(device, DEVICE)


class TestLazyStateInit(unittest.TestCase):
    """__init_state() must run once per parameter (during __init__/first step),
    not on every step_parameter() call."""

    def test_init_state_not_called_every_step(self):
        real_init = AdaMuon_adv._AdaMuon_adv__init_state
        count = [0]

        def counting(self, p, group):
            count[0] += 1
            return real_init(self, p, group)

        AdaMuon_adv._AdaMuon_adv__init_state = counting
        try:
            p1 = make_param((8, 16))
            opt = AdaMuon_adv([p1], use_muon=True)
            calls_after_init = count[0]  # init_step() ran once per param
            for _ in range(3):
                opt.step()
            self.assertEqual(count[0], calls_after_init)
        finally:
            AdaMuon_adv._AdaMuon_adv__init_state = real_init

    def test_add_param_group_initialized_lazily_once(self):
        real_init = AdaMuon_adv._AdaMuon_adv__init_state
        count = [0]

        def counting(self, p, group):
            count[0] += 1
            return real_init(self, p, group)

        AdaMuon_adv._AdaMuon_adv__init_state = counting
        try:
            p1 = make_param((8, 16))
            opt = AdaMuon_adv([p1], use_muon=True)
            calls_after_init = count[0]

            p2 = make_param((4, 8))
            opt.add_param_group({"params": [p2], "use_muon": True})

            # First step initializes the freshly added param exactly once;
            # the second step must not re-initialize it.
            opt.step()
            after_first_step = count[0]
            opt.step()
            self.assertEqual(after_first_step, calls_after_init + 1)
            self.assertEqual(count[0], after_first_step)
        finally:
            AdaMuon_adv._AdaMuon_adv__init_state = real_init


class TestNormuonAdaptiveEps(unittest.TestCase):
    """NorMuon must receive the scale-invariant adaptive_eps, not the raw
    group['eps'] (which is None when eps=None)."""

    def test_normuon_uses_adaptive_eps(self):
        # AdaMuon_adv imports normuon_update directly (`from ..util.Muon_util
        # import normuon_update`), so the spy must be installed on the
        # AdaMuon_adv module namespace, not on Muon_util.
        import importlib

        mod = importlib.import_module("adv_optm.optim.AdaMuon_adv")

        real_normuon = mod.normuon_update
        captured = {}

        def spy(update, v_t, beta2, eps):
            captured["eps"] = eps
            return real_normuon(update, v_t, beta2, eps)

        mod.normuon_update = spy
        try:
            p = make_param((8, 16))
            opt = AdaMuon_adv(
                [p],
                use_muon=True,
                normuon_variant=True,
                eps=None,  # scale_eps(None, p) -> 1/sqrt(numel)
            )
            opt.step()
            self.assertIsNotNone(captured.get("eps"))
            expected = 1.0 / math.sqrt(p.numel())
            self.assertAlmostEqual(captured["eps"], expected, places=6)
        finally:
            mod.normuon_update = real_normuon


if __name__ == "__main__":
    unittest.main()
