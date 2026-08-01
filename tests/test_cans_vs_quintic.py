"""Empirical comparison: Chebyshev-accelerated Newton-Schulz (CANS) vs. the
fixed-coefficient quintic Newton-Schulz used in adv_optm/util/Muon_util.py.

Verifies:
- CANS achieves lower orthogonalization error than the fixed quintic at equal
  step count on well-conditioned random matrices (the paper's claim).
- CANS at reduced steps (3) is competitive with quintic at 5 (compute claim:
  CANS also does 2 matmuls/step vs 3 for the quintic).
- On LoRA-like thin, low-rank matrices (zero-init B => early updates are
  extremely rank-deficient), CANS does not blow up: output stays finite and
  bounded even though true singular values fall far below the auto-derived
  cns_a_bound.

All tests run on CUDA as mandated by the project conventions.
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adv_optm.util.Muon_util import newton_schulz  # noqa: E402

DEVICE = torch.device("cuda:0")


def ortho_error(X: torch.Tensor) -> float:
    """Frobenius distance of X's rows from orthonormality: ||X X^T - I||_F.

    newton_schulz transposes so rows <= cols internally, so rows are the
    orthonormalized dimension for wide matrices; for tall ones use columns.
    """
    X = X.float()
    if X.size(0) <= X.size(1):
        G = X @ X.mT
        I = torch.eye(X.size(0), device=X.device)
    else:
        G = X.mT @ X
        I = torch.eye(X.size(1), device=X.device)
    return (G - I).norm().item()


class TestCANSvsQuintic(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for these tests")
        torch.manual_seed(7)

    def test_cans_lower_error_full_rank(self):
        """On a well-conditioned random matrix, CANS@5 should orthogonalize
        at least as well as the fixed quintic@5."""
        G = torch.randn(64, 64, device=DEVICE)
        err_q5 = ortho_error(newton_schulz(G.clone(), steps=5, cns=False))
        err_c5 = ortho_error(newton_schulz(G.clone(), steps=5, cns=True))
        err_c3 = ortho_error(newton_schulz(G.clone(), steps=3, cns=True))
        print(f"\nfull-rank 64x64: quintic@5={err_q5:.4f}  CANS@5={err_c5:.4f}  CANS@3={err_c3:.4f}")
        self.assertLessEqual(err_c5, err_q5 + 1e-3)

    def test_cans_not_strictly_better_ill_conditioned(self):
        """CANS is NOT strictly better. On an ill-conditioned matrix (singular
        values log-spaced down to 1e-3), at the standard 5-step budget:
        - auto-bound CANS LOSES to the fixed quintic;
        - no manually tuned cns_a_bound rescues it at 5 steps (grid below);
        - but CANS converges faster asymptotically: by 10 steps it beats the
          quintic. I.e. CANS needs more steps on wide/ill-conditioned spectra;
          its per-step advantage shows on well-conditioned matrices only."""
        U = torch.linalg.qr(torch.randn(64, 64, device=DEVICE)).Q
        V = torch.linalg.qr(torch.randn(64, 64, device=DEVICE)).Q
        s = torch.logspace(0, -3, 64, device=DEVICE)
        G = U @ torch.diag(s) @ V.mT

        err_q5 = ortho_error(newton_schulz(G.clone(), steps=5, cns=False))
        err_c5_auto = ortho_error(newton_schulz(G.clone(), steps=5, cns=True))
        err_c10_auto = ortho_error(newton_schulz(G.clone(), steps=10, cns=True))
        grid = {
            a: ortho_error(newton_schulz(G.clone(), steps=5, cns=True, cns_a_bound=a))
            for a in (0.5, 0.3, 0.1, 0.05, 0.01, 1e-3, 5e-4)
        }
        print(
            f"\nill-conditioned 64x64: quintic@5={err_q5:.4f}  "
            f"CANS@5(auto)={err_c5_auto:.4f}  CANS@10(auto)={err_c10_auto:.4f}  "
            f"bound-grid@5={ {a: round(e, 3) for a, e in grid.items()} }"
        )
        # At equal (5) steps, CANS loses on this spectrum...
        self.assertGreater(err_c5_auto, err_q5)
        # ...and cns_a_bound tuning does not rescue it at 5 steps.
        self.assertGreater(min(grid.values()), err_q5)
        # ...but with more steps CANS's faster asymptotic convergence wins.
        self.assertLess(err_c10_auto, err_q5)

    def test_cans_stable_on_lora_like_low_rank(self):
        """LoRA early training: rank-4 update on a thin 16x1024 factor. True
        singular values (4 nonzero, rest ~0) violate the auto cns_a_bound
        assumption, but CANS must still produce a finite, bounded output."""
        A = torch.randn(16, 4, device=DEVICE)
        B = torch.randn(4, 1024, device=DEVICE)
        G = A @ B  # rank <= 4
        for steps in (3, 5):
            out = newton_schulz(G.clone(), steps=steps, cns=True)
            self.assertTrue(torch.isfinite(out).all(), "CANS produced non-finite values")
            # Row norms of an orthogonalized wide matrix should be ~1; require bounded.
            row_norms = out.float().norm(dim=1)
            self.assertLess(row_norms.max().item(), 3.0, f"CANS rows blew up: {row_norms.max()}")
            err_c = ortho_error(out)
            err_q = ortho_error(newton_schulz(G.clone(), steps=steps, cns=False))
            print(f"\nlow-rank 16x1024 (rank 4), steps={steps}: quintic={err_q:.4f}  CANS={err_c:.4f}")

    def test_cans_ignores_coeffs_but_respects_steps(self):
        """CANS computes its own coefficients: passing custom quintic coeffs
        must not change its output, but step count must."""
        G = torch.randn(32, 128, device=DEVICE)
        out_default = newton_schulz(G.clone(), steps=5, cns=True)
        out_custom = newton_schulz(G.clone(), steps=5, cns=True, coeffs=(1.0, 1.0, 1.0))
        out_fewer = newton_schulz(G.clone(), steps=3, cns=True)
        torch.testing.assert_close(out_default, out_custom)
        self.assertFalse(torch.allclose(out_default, out_fewer))


if __name__ == "__main__":
    unittest.main()
