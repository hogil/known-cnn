"""Machine checks for the T6 operator-selection-regret cluster (260724).

T6a: the 2-world selection game has value Delta/2 (randomized) vs Delta
     (deterministic).
T6b: the exact minimax regret V(I) = min_p sup_Q [max_j u_j(Q) - p.u(Q)] equals its
     convex-game dual max_pi [E_pi max_j u_j - max_j E_pi u_j] (LP strong duality),
     checked on random finite instances.
T6d: with m i.i.d. target multi-positives, empirical-argmax selection regret decays
     as O(sqrt(log K / m)) (rate slope check).

Pure numpy + scipy.optimize.linprog. No training.
"""
import numpy as np
from scipy.optimize import linprog


# ---- T6a: 2-world 2-operator regret game ----
def t6a(Delta=0.4, U=1.0):
    # worlds A,B; utilities u[world, op]
    u = np.array([[U, U - Delta],   # world A: op a best
                  [U - Delta, U]])  # world B: op b best
    # deterministic: pick op a -> regret = max(0, Delta) in world B; worst-case = Delta
    det = min(max(u[:, 0].max() - u[:, 0].min(), 0.0),  # not the game; do it directly:
              Delta)
    # regret of fixed op j = max_w (max_k u[w,k] - u[w,j])
    reg_det = [max(u[w].max() - u[w, j] for w in range(2)) for j in range(2)]
    det_val = min(reg_det)                       # best deterministic worst-case regret
    # randomized p=(0.5,0.5): regret_w = max_k u[w,k] - p.u[w]
    p = np.array([0.5, 0.5])
    rand_val = max(u[w].max() - p @ u[w] for w in range(2))
    return det_val, rand_val, Delta


# ---- T6b: minimax == maximin (LP strong duality) on a finite instance ----
def minimax_primal(u):
    """u: (M worlds) x (K ops). V = min_{p in simplex} max_w [best_w - p.u_w].
    LP: min t s.t. best_w - p.u_w <= t for all w; sum p =1; p>=0."""
    M, K = u.shape
    best = u.max(axis=1)
    # vars: [p_1..p_K, t]
    c = np.zeros(K + 1); c[-1] = 1.0
    # best_w - p.u_w <= t  ->  -p.u_w - t <= -best_w
    A_ub = np.hstack([-u, -np.ones((M, 1))]); b_ub = -best
    A_eq = np.zeros((1, K + 1)); A_eq[0, :K] = 1.0; b_eq = [1.0]
    bounds = [(0, None)] * K + [(None, None)]
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return r.fun, r.x[:K]


def maximin_dual(u):
    """V = max_{pi over worlds} [ E_pi best_w - max_j E_pi u_j ]. LP:
    max s  s.t.  s <= pi.best - pi.u_j  for all j (i.e. max_j pi.u_j term);
    equivalently max (pi.best - z), z >= pi.u_j all j, sum pi=1, pi>=0."""
    M, K = u.shape
    best = u.max(axis=1)
    # vars: [pi_1..pi_M, z]; maximize pi.best - z = min -(pi.best) + z
    c = np.concatenate([-best, [1.0]])
    # z >= pi.u_j  ->  pi.u_j - z <= 0 for each op j
    A_ub = np.hstack([u.T, -np.ones((K, 1))]); b_ub = np.zeros(K)
    A_eq = np.zeros((1, M + 1)); A_eq[0, :M] = 1.0; b_eq = [1.0]
    bounds = [(0, None)] * M + [(None, None)]
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return -r.fun  # undo the sign (we minimized the negative)


# ---- T6d: WORST-CASE (minimax over gap) empirical-argmax regret rate ----
def t6d_rate(seed=0):
    """The O(sqrt(log K/m)) rate is MINIMAX (worst gap), not fixed-gap. For each m,
    the adversary chooses the gap Delta maximizing E[regret] = Delta * P(mis-pick).
    Two Bernoulli arms at 0.5 +/- Delta/2; regret = Delta * P(argmax wrong). Sweep
    Delta, take the max -> that envelope decays as ~1/sqrt(m)."""
    rng = np.random.default_rng(seed)
    ms = [10, 20, 40, 80, 160, 320, 640, 1280]
    gaps = np.linspace(0.005, 0.5, 60)
    regs = []
    for m in ms:
        wc = 0.0
        for D in gaps:
            pa, pb = 0.5 + D / 2, 0.5 - D / 2
            mis = 0
            for _ in range(4000):
                ea = rng.binomial(m, pa) / m; eb = rng.binomial(m, pb) / m
                if eb > ea or (eb == ea and rng.random() < 0.5):
                    mis += 1
            wc = max(wc, D * mis / 4000.0)      # worst-case regret at this m
        regs.append(wc)
    ms = np.array(ms, float); regs = np.array(regs)
    slope = np.polyfit(np.log(ms), np.log(regs), 1)[0]
    return ms, regs, slope


def main():
    print("== T6a: 2-world selection game (Delta=0.4) ==")
    det, rand, D = t6a(0.4)
    print(f"  deterministic worst-case regret = {det:.3f}  (= Delta = {D})")
    print(f"  randomized (1/2,1/2) worst-case  = {rand:.3f}  (= Delta/2 = {D/2})")
    assert abs(det - D) < 1e-9 and abs(rand - D / 2) < 1e-9
    print("  PASS: randomized halves worst-case to Delta/2 (game value)")

    print("\n== T6b: minimax primal == maximin dual (strong duality) ==")
    maxerr = 0.0
    rng = np.random.default_rng(1)
    for t in range(200):
        M = rng.integers(2, 6); K = rng.integers(2, 6)
        u = rng.uniform(0, 1, size=(M, K))
        vp, _ = minimax_primal(u); vd = maximin_dual(u)
        maxerr = max(maxerr, abs(vp - vd))
    print(f"  max |V_primal - V_dual| over 200 random instances = {maxerr:.2e}  "
          f"({'PASS' if maxerr < 1e-6 else 'FAIL'})")

    print("\n== T6d: empirical-argmax regret rate (expect log-log slope ~ -0.5) ==")
    ms, regs, slope = t6d_rate()
    for m, r in zip(ms, regs):
        print(f"   m={int(m):5d}  regret={r:.5f}")
    print(f"  log-log slope = {slope:.3f}  "
          f"({'PASS ~ -0.5 (O(1/sqrt m))' if -0.75 < slope < -0.35 else 'CHECK'})")


if __name__ == "__main__":
    main()
