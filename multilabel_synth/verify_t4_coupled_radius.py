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


def main():
    print(f"{'rho':>5} {'eps_m':>6} {'closed':>9} {'brute':>9} {'|diff|':>8}")
    maxerr = 0.0
    for rho in (0.0, 0.25, 0.5, 0.75, 1.0):
        for eps in (0.0, 0.02, 0.1, 0.25, 0.5):
            cf = closed_form(rho, eps); bf = brute(rho, eps)
            d = abs(cf - bf); maxerr = max(maxerr, d)
            print(f"{rho:5.2f} {eps:6.2f} {cf:9.4f} {bf:9.4f} {d:8.1e}")
    print(f"\nmax |closed - brute| = {maxerr:.2e}  "
          f"({'PASS' if maxerr < 1e-3 else 'FAIL'} at 1e-3)")
    # sanity: g(0,m)=0 for all m; A*_norm(inf)=A*(1-rho)=A_Frechet
    assert abs(closed_form(0.0, 0.3) - ASTAR) < 1e-12         # rho=0 -> no reduction
    assert abs(closed_form(1.0, 0.0)) < 1e-12                 # rho=1, perfect -> 0
    print("invariants OK: g(0,m)=0; A*_norm(inf,rho=1)=0=A_Frechet")


if __name__ == "__main__":
    main()
