"""
Corrected two-person model for the folie a deux cusp paper (correction, Oct 2026).
Revised 23 Oct 2026 after an outside review: equilibria are now solved for directly
and checked by residual, and the exact threshold and fold are reported next to the
reduced (first-order in P) predictions. Revised 26 Oct 2026 after a second review:
equilibria are classified with a tolerance, and those with an eigenvalue within 1e-8 of
zero are reported as nonhyperbolic (linear stability is inconclusive there).

Parameter domain throughout: E > 0, P >= 0, couplings Kp, Ks >= 0.

Each person i holds a belief q_i about one proposition, measured from the consensus
baseline (q = 0 is what everyone else believes).
  E  : external precision, the pull of the outside world back toward baseline
  K  : coupling precision, how much each treats the partner's belief as evidence
  P,m: the primary's own fixed position m, held with prior precision P (Mode 1)
  cubic term: beliefs saturate (bounded confidence)

Joint potential (symmetric coupling):
  V = E/2 (q1^2+q2^2) - K q1 q2 + (q1^4+q2^4)/4 + P/2 (q1-m)^2
Dynamics: q_i' = -dV/dq_i  (each person descends their own free energy).

Shared view s = (qp+qs)/2, half the gap d = (qp-qs)/2. Exact equations:
  s' = (K - E - P/2) s - s^3 + P m/2 - (P/2) d - 3 s d^2
  d' = -(E + K + P/2 + 3 s^2) d - d^3 + (P/2)(m - s)
With d = O(P) the last two terms of s' are O(P^2), so to first order in P
  s' = r s - s^3 + h,   r = K - E - P/2,   h = P m / 2
Exact baseline threshold at m = 0:  Kps*Ksp = E(E+P)  (symmetric: K_c = sqrt(E(E+P))).
"""
import numpy as np
from numpy.polynomial import Polynomial as Poly

def rhs(q, E, Kp, Ks, P, m):
    # q[0] primary, q[1] secondary. Kp = primary's influence on secondary, Ks = secondary's on primary.
    qp, qs = q
    return np.array([-E*qp + Ks*qs - qp**3 + P*(m - qp),
                     -E*qs + Kp*qp - qs**3])

def jacobian(q, E, Kp, Ks, P):
    qp, qs = q
    return np.array([[-E - 3*qp**2 - P, Ks], [Kp, -E - 3*qs**2]])

def settle(q0, E, Kp, Ks, P=0.0, m=0.0, T=400.0, dt=0.01):
    q = np.array(q0, float)
    for _ in range(int(T/dt)):
        q = q + dt*rhs(q, E, Kp, Ks, P, m)
    return q

def newton(q0, E, Kp, Ks, P, m, steps=80):
    # refine a candidate equilibrium; returns (point, residual), keeping the best point seen
    q = np.array(q0, float)
    best, best_res = q.copy(), np.max(np.abs(rhs(q, E, Kp, Ks, P, m)))
    for _ in range(steps):
        f = rhs(q, E, Kp, Ks, P, m)
        if np.max(np.abs(f)) < 1e-14: break
        try: q = q - np.linalg.solve(jacobian(q, E, Kp, Ks, P), f)
        except np.linalg.LinAlgError: break
        res = np.max(np.abs(rhs(q, E, Kp, Ks, P, m)))
        if res < best_res: best, best_res = q.copy(), res
    return best, best_res

EIG_TOL = 1e-8

def classify(q, E, Kp, Ks, P):
    # 'stable' / 'unstable' by the largest real part of the Jacobian's eigenvalues;
    # 'nonhyperbolic' when it is within EIG_TOL of zero, where linear analysis is inconclusive
    lam = np.max(np.linalg.eigvals(jacobian(q, E, Kp, Ks, P)).real)
    if abs(lam) < EIG_TOL: return 'nonhyperbolic'
    return 'stable' if lam < 0 else 'unstable'

def is_stable(q, E, Kp, Ks, P):
    return classify(q, E, Kp, Ks, P) == 'stable'

def equilibria(E, Kp, Ks, P=0.0, m=0.0):
    """Equilibria computed numerically and checked by residual. With Kp != 0 the
    secondary's equation gives qp = (E y + y^3)/Kp with y = qs; substituting into the
    primary's equation leaves a single degree-9 polynomial in y. Each real or nearly real
    root is Newton-refined and kept only if the residual of both equations is below 1e-10.
    Returns [(q, cls)], cls one of 'stable', 'unstable', 'nonhyperbolic'."""
    if Kp != 0:
        y = Poly([0, 1]); qp = (E*y + y**3)/Kp
        poly = -(E + P)*qp - qp**3 + Ks*y + P*m
        cands = [((E*z + z**3)/Kp, z) for z in poly.roots() if abs(z.imag) < 1e-4 for z in [z.real]]
    else:  # secondary is not influenced: qs = 0, primary solves a cubic
        cands = [(z.real, 0.0) for z in Poly([P*m, -(E + P), 0, -1]).roots() if abs(z.imag) < 1e-4]
    out = []
    for c in cands:
        q, res = newton(c, E, Kp, Ks, P, m)
        if res < 1e-10 and not any(np.allclose(q, u, atol=1e-7) for u, _ in out):
            out.append((q, classify(q, E, Kp, Ks, P)))
    return out

def stable_states(E, Kp, Ks, P=0.0, m=0.0):
    return [q for q, c in equilibria(E, Kp, Ks, P, m) if c == 'stable']

def nonhyperbolic_states(E, Kp, Ks, P=0.0, m=0.0):
    return [q for q, c in equilibria(E, Kp, Ks, P, m) if c == 'nonhyperbolic']

def stable_by_integration(E, Kp, Ks, P=0.0, m=0.0):
    # cross-check: integrate from a grid of starts, Newton-refine each end point, keep it
    # only if it is a true equilibrium close to where the run ended and it is stable
    starts = [(a, b) for a in (-1.5, -0.4, 0.4, 1.5) for b in (-1.5, -0.4, 0.4, 1.5)]
    uniq = []
    for s0 in starts:
        e = settle(s0, E, Kp, Ks, P, m)
        q, res = newton(e, E, Kp, Ks, P, m)
        if res > 1e-10 or np.max(np.abs(q - e)) > 0.05 or not is_stable(q, E, Kp, Ks, P): continue
        if not any(np.allclose(q, u, atol=1e-6) for u in uniq): uniq.append(q)
    return uniq

def reduced_stable(r, h):
    # hyperbolic stable equilibria of x' = rx - x^3 + h only; at r = h = 0 the origin
    # is stable but nonhyperbolic and is not returned (not used at that point here)
    return [x.real for x in np.roots([1.0, 0.0, -r, -h]) if abs(x.imag) < 1e-9 and r - 3*x.real**2 < 0]

def exact_fold(E, K, P, guess):
    # solve both equilibrium equations together with det J = 0 for (qp, qs, m)
    v = np.array(guess, float)
    def F(v):
        qp, qs, m = v
        return np.array([*rhs((qp, qs), E, K, K, P, m), (E + P + 3*qp**2)*(E + 3*qs**2) - K**2])
    for _ in range(60):
        f = F(v); J = np.zeros((3, 3)); h = 1e-7
        for i in range(3):
            dv = np.zeros(3); dv[i] = h; J[:, i] = (F(v + dv) - f)/h
        v = v - np.linalg.solve(J, f)
    return v, np.max(np.abs(F(v)))

def rk4_settle(q, E, K, P, m, T, dt=0.05):
    q = np.array(q, float)
    for _ in range(int(T/dt)):
        k1 = rhs(q, E, K, K, P, m); k2 = rhs(q + dt/2*k1, E, K, K, P, m)
        k3 = rhs(q + dt/2*k2, E, K, K, P, m); k4 = rhs(q + dt*k3, E, K, K, P, m)
        q = q + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    return q

def first_jump(ms, path, thresh=0.3):
    # first parameter value at which the shared belief jumps by more than thresh; None if no jump
    d = np.abs(np.diff(path)) > thresh
    return ms[np.argmax(d) + 1] if d.any() else None

def sweep(E, K, P, m_from, m_to, step, T):
    # quasi-static sweep of m starting on the equilibrium branch at m_from
    ms = np.arange(m_from, m_to + np.sign(m_to - m_from)*step/2, np.sign(m_to - m_from)*step)
    st = stable_states(E, K, K, P, ms[0])
    q = min(st, key=lambda u: np.sign(m_to - m_from)*(u[0] + u[1]))  # start on the branch that will jump
    path = []
    for mm in ms:
        q = rk4_settle(q, E, K, P, mm, T); path.append(0.5*(q[0] + q[1]))
    return ms, np.array(path)

if __name__ == "__main__":
    fmt = lambda q: f"({q[0]:+.6f},{q[1]:+.6f})"

    print("A. Symmetric coupling, no anchor (P = 0): which variable tips?")
    E = 0.5
    for K in (0.3, 0.45, 0.6, 0.8, 1.0):
        st = stable_states(E, K, K); n_int = len(stable_by_integration(E, K, K))
        desc = [f"shared {0.5*(u[0]+u[1]):+.6f}, gap {0.5*(u[0]-u[1]):+.1e}" for u in st]
        print(f"   K={K}: {len(st)} stable (integration finds {n_int}): {desc} | predicted shared +/-{np.sqrt(max(K-E, 0)):.6f}")

    print("\nA2. Threshold with the anchor at m = 0 (E = 0.5, P = 0.2)")
    E, P = 0.5, 0.2
    Kc = np.sqrt(E*(E + P))
    print(f"   exact K_c = sqrt(E(E+P)) = {Kc:.6f}; reduced (r = 0) K = E + P/2 = {E + P/2:.6f}; "
          f"series E + P/2 - P^2/(8E) = {E + P/2 - P**2/(8*E):.6f}")
    for K in (Kc - 0.005, Kc + 0.005, E + P/2 - 0.001):
        lam = np.linalg.eigvals(jacobian((0, 0), E, K, K, P)).real.max()
        print(f"   K={K:.4f}: largest baseline eigenvalue {lam:+.5f}, {len(stable_states(E, K, K, P, 0.0))} stable state(s)")

    print("\nA3. Nonhyperbolic test case: E = K = 0.5, P = 0 (exactly at the threshold)")
    E = 0.5
    eq = equilibria(E, E, E)
    for q, c in eq:
        lam = np.linalg.eigvals(jacobian(q, E, E, E, 0.0)).real
        print(f"   equilibrium {fmt(q)}: eigenvalues {np.sort(lam)}, classified {c}")
    q = settle((0.1, 0.1), E, E, E, T=400)
    print(f"   linear analysis is inconclusive here; along the shared direction s' = -s^3, so the origin is "
          f"stable but slow: from (0.1, 0.1), shared belief after t = 400 is {0.5*(q[0]+q[1]):.4f} "
          f"(s' = -s^3 predicts {0.1/np.sqrt(1 + 2*0.01*400):.4f})")

    print("\nB. Anchored model (E = 0.5, K = 1.2, P = 0.2): full model vs reduced cusp")
    E, K, P = 0.5, 1.2, 0.2
    r = K - E - P/2
    for m in (0.0, 0.3, 0.6, 1.0, 1.4, 1.76, 1.78, 1.8, 2.0):
        h = P*m/2
        eq = equilibria(E, K, K, P, m); st = [q for q, c in eq if c == 'stable']
        nh = [q for q, c in eq if c == 'nonhyperbolic']
        n_int = len(stable_by_integration(E, K, K, P, m))
        print(f"   m={m:.2f}: full {len(st)} stable of {len(eq)} equilibria, {len(nh)} nonhyperbolic (integration finds {n_int}), "
              f"reduced predicts {len(reduced_stable(r, h))}; "
              + "; ".join(f"shared {0.5*(u[0]+u[1]):+.4f} gap {0.5*(u[0]-u[1]):+.4f}" for u in st))
    q0 = max(stable_states(E, K, K, P, 0.0), key=lambda u: u[0])
    s, d = 0.5*(q0[0] + q0[1]), 0.5*(q0[0] - q0[1])
    print(f"   m=0 positive state {fmt(q0)}: shared {s:.6f}, gap {d:+.6f}; reduced sqrt(r) = {np.sqrt(r):.6f}")
    print(f"   first-order gap estimate P(m - s)/(2(E+K+3s^2)) = {P*(0 - s)/(2*(E + K + 3*s**2)):+.6f}")
    mz = np.sqrt(K - E)
    qz = np.array([mz, mz])
    print(f"   the anchor generally opens a gap, but not always: at m = sqrt(K - E) = {mz:.6f} the point "
          f"q_p = q_s = m has residual {np.max(np.abs(rhs(qz, E, K, K, P, mz))):.1e}, gap 0, classified {classify(qz, E, K, K, P)}")

    print("\nB2. Exact fold of the full model vs the reduced prediction")
    v, res = exact_fold(E, K, P, (-0.35, -0.53, 1.77))
    m_red = 2*2*(r/3)**1.5/P
    print(f"   exact fold (qp, qs, m) = ({v[0]:+.8f}, {v[1]:+.8f}, {v[2]:.8f}), residual {res:.1e}")
    print(f"   reduced fold |h| = 2(r/3)^1.5 -> |m| = {m_red:.8f}; difference {100*(m_red - v[2])/v[2]:.2f}%")
    m_x = 1.78
    print(f"   m = {m_x}: full model {len(stable_states(E, K, K, P, m_x))} stable, reduced {len(reduced_stable(r, P*m_x/2))} stable")
    m_b = v[2] + 1e-5
    print(f"   m = fold + 1e-5 = {m_b:.8f}: full model {len(stable_states(E, K, K, P, m_b))} stable "
          f"(integration with residual check finds {len(stable_by_integration(E, K, K, P, m_b))})")

    print("\nC. Unequal coupling")
    for Kp, Ks in ((0.9, 0.3), (1.2, 0.0), (0.6, 0.6)):
        st = stable_states(0.5, Kp, Ks)
        print(f"   P=0: Kp={Kp}, Ks={Ks}: sqrt(Kp*Ks)={np.sqrt(Kp*Ks):.3f} vs E=0.5 -> {len(st)} stable: "
              + "; ".join(fmt(u) for u in st))
    E, P = 0.5, 0.2
    for Kp, Ks in ((0.9, 0.38), (0.9, 0.40)):
        lam = np.linalg.eigvals(jacobian((0, 0), E, Kp, Ks, P)).real.max()
        print(f"   P=0.2, m=0: Kp*Ks={Kp*Ks:.3f} vs E(E+P)={E*(E+P):.3f}: largest baseline eigenvalue {lam:+.5f}, "
              f"{len(stable_states(E, Kp, Ks, P, 0.0))} stable")
    print("   one-way influence (Kp = 1.2, Ks = 0) with an anchor: number of equilibria and tracking")
    for m in (0.5, 2.0, 5.0):
        eq = equilibria(E, 1.2, 0.0, P, m); q = eq[0][0]
        print(f"     m={m}: {len(eq)} equilibrium; qs = {q[1]:+.4f}, linear tracking (Kp/E) qp = {1.2/E*q[0]:+.4f}")

    print("\nD. Hysteresis sweeps of m (E = 0.5, K = 1.2, P = 0.2); the true fold is at |m| = %.4f" % v[2])
    for step, T in ((0.1, 60), (0.01, 60), (0.01, 400)):
        ms, path = sweep(E, K, P, 1.0, 2.2, step, T); up = first_jump(ms, path)
        ms, path = sweep(E, K, P, -1.0, -2.2, step, T); dn = first_jump(ms, path)
        f = lambda x: "no jump" if x is None else f"{x:+.2f}"
        print(f"   step {step}, time {T} per step: jump up at m = {f(up)}, down at m = {f(dn)}")
    ms, path = sweep(E, K, P, 1.0, 1.5, 0.1, 60)
    print(f"   detector check on a sweep with no jump (m 1.0 to 1.5): {first_jump(ms, path)}")

    print("\nE. Separation as treatment: cut coupling to K=0 from a locked state")
    q = max(stable_states(0.5, 1.2, 1.2, 0.2, 1.0), key=lambda u: u[0])
    after = settle(q, 0.5, 0.0, 0.0, 0.2, 1.0, T=200)
    exact_p = [z.real for z in Poly([0.2, -0.7, 0, -1]).roots() if abs(z.imag) < 1e-9][0]
    print(f"   locked (primary, secondary) = {fmt(q)}; after separation = {fmt(after)}; "
          f"primary's exact resting point {exact_p:.6f}")
    print("   the secondary returns to baseline at rate E; the primary settles where their own conviction balances external precision")
