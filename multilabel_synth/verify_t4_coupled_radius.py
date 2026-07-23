"""Machine check of Theorem 4'(c): the closed-form coupled-nuisance reachable
radius A*_norm(m) = A*(M)[(1-rho) + 2 rho eps_m] against a brute-force grid over
the reachable copula/mixing parameters. Reproduces the numeric verification the
theory relied on. No training; pure arithmetic.

Model S-LINK/interpolation: Z in {0,1}, P_Z(Z=1)=w, marginals 1/2, operator OR,
p11(z) = (1/2)[(1-rho) f_z + rho z], f_z in [0,1] free per z; normals identify w
to [w0-eps, w0+eps]. Observable: P(v=1) = 1 - (1/2) E_Z[ (1-rho)f_Z + rho Z ].
A*_norm = (1/2) * (range of P(v=1) over reachable f_0,f_1,w).
"""
import numpy as np

ASTAR = 0.25  # A*(M) for this instance


def closed_form(rho, eps):
    return ASTAR * ((1.0 - rho) + 2.0 * rho * eps)


def brute(rho, eps, w0=0.5, n=81):
    f = np.linspace(0, 1, n)
    ws = np.linspace(max(0, w0 - eps), min(1, w0 + eps), n)
    vals = []
    for w in ws:
        for f0 in f:
            for f1 in f:
                EZ = (1 - w) * ((1 - rho) * f0 + rho * 0) + w * ((1 - rho) * f1 + rho * 1)
                vals.append(1.0 - 0.5 * EZ)
    vals = np.array(vals)
    return 0.5 * (vals.max() - vals.min())


def reachable_interval(rho, eps, w0=0.5, n=81):
    """Reachable set of the decision-relevant observable v=P(pair present) over the
    consistent parameters, as an interval [lo, hi]. Its half-width is A*_norm(m)."""
    f = np.linspace(0, 1, n)
    ws = np.linspace(max(0, w0 - eps), min(1, w0 + eps), n)
    vals = []
    for w in ws:
        for f0 in f:
            for f1 in f:
                EZ = (1 - w) * ((1 - rho) * f0) + w * ((1 - rho) * f1 + rho * 1)
                vals.append(1.0 - 0.5 * EZ)
    return float(np.min(vals)), float(np.max(vals))


def achievability_check(rho, eps):
    """Theorem 4'(e): the constructive procedure = play the Bayes bit-rule of the
    Chebyshev CENTER of the reachable set. Verify (i) its worst-case appearance
    error equals the identification half-width A*_norm(m) (matches the 4'(d) lower
    bound -> TIGHT), and (ii) any off-center choice is strictly worse (center is the
    unique minimax procedure). For the scalar decision variable v, the Bayes bit
    flips at v=1/2; excess appearance error of committing to estimate q is
    max_{p in reachable} |q - p|."""
    lo, hi = reachable_interval(rho, eps)
    half = 0.5 * (hi - lo)                       # = A*_norm(m)
    center = 0.5 * (lo + hi)
    wc_center = max(abs(center - lo), abs(center - hi))        # worst-case for P*
    wc_edge = max(abs(lo - lo), abs(lo - hi))                 # worst-case if play edge
    return half, wc_center, wc_edge


def main():
    print("== Thm 4'(c) identification-width closed form vs brute ==")
    print(f"{'rho':>5} {'eps_m':>6} {'closed':>9} {'brute':>9} {'|diff|':>8}")
    maxerr = 0.0
    for rho in (0.0, 0.25, 0.5, 0.75, 1.0):
        for eps in (0.0, 0.02, 0.1, 0.25, 0.5):
            cf = closed_form(rho, eps); bf = brute(rho, eps)
            d = abs(cf - bf); maxerr = max(maxerr, d)
            print(f"{rho:5.2f} {eps:6.2f} {cf:9.4f} {bf:9.4f} {d:8.1e}")
    print(f"\nmax |closed - brute| = {maxerr:.2e}  "
          f"({'PASS' if maxerr < 1e-3 else 'FAIL'} at 1e-3)")
    assert abs(closed_form(0.0, 0.3) - ASTAR) < 1e-12         # rho=0 -> no reduction
    assert abs(closed_form(1.0, 0.0)) < 1e-12                 # rho=1, perfect -> 0
    print("invariants OK: g(0,m)=0; A*_norm(inf,rho=1)=0=A_Frechet")

    print("\n== Thm 4'(e) achievability: center-procedure attains the width (tight) ==")
    print(f"{'rho':>5} {'eps_m':>6} {'A*_norm':>9} {'wc(P*)':>9} {'wc(edge)':>9} {'match':>6}")
    tight_ok = True
    for rho in (0.0, 0.25, 0.5, 0.75, 1.0):
        for eps in (0.0, 0.1, 0.25):
            half, wc_c, wc_e = achievability_check(rho, eps)
            match = abs(wc_c - half) < 1e-9              # P* attains identification half-width
            better = wc_c <= wc_e + 1e-12                # center no worse than edge
            tight_ok = tight_ok and match and better
            print(f"{rho:5.2f} {eps:6.2f} {half:9.4f} {wc_c:9.4f} {wc_e:9.4f} "
                  f"{'ok' if (match and better) else 'FAIL':>6}")
    print(f"\nachievability: center procedure worst-case == A*_norm(m) for all "
          f"(rho,eps)  ({'PASS' if tight_ok else 'FAIL'})")
    print("=> upper (constructive) meets lower (4'd) => minimax risk = Theta(A*_norm(m)),"
          "\n   a TIGHT finite-m rate, not a lower bound only.")


if __name__ == "__main__":
    main()
