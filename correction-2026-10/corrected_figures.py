"""Corrected Figures 1 and 3 (Figure 2 is unchanged). numpy + matplotlib only."""
import io, contextlib, os, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import corrected_model as cm
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"); os.makedirs(OUT, exist_ok=True)

# ---------- Figure 1: same normal form, corrected labels ----------
def count_stable(r, h):
    roots = np.roots([1.0, 0.0, -r, -h])
    return sum(1 for x in roots if abs(x.imag) < 1e-8 and r - 3*x.real**2 < 0)
rv = np.linspace(-1.2, 2.8, 250); hv = np.linspace(-2.2, 2.2, 250)
R, H = np.meshgrid(rv, hv); Z = np.vectorize(count_stable)(R, H)
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.contourf(R, H, Z, levels=[0.5, 1.5, 2.5], cmap=ListedColormap(['#ADD8E6', '#90EE90']), alpha=0.85)
rp = np.linspace(0, 2.8, 300); hb = np.sqrt(4/27*rp**3)
ax.plot(rp, hb, 'r-', lw=2.5, label='Cusp bifurcation set (analytical)'); ax.plot(rp, -hb, 'r-', lw=2.5)
ax.axhline(0, color='gray', ls='--', alpha=.5, lw=1); ax.axvline(0, color='gray', ls='--', alpha=.5, lw=1)
ax.plot(0, 0, 'ko', ms=8, label='Cusp point (r=0, h=0)')
ax.text(-0.55, 0.95, 'MONOSTABLE\n(one state; near the\nbaseline only when\n$h$ is small)', ha='center', va='center', fontsize=9, color='#00008B', fontweight='bold')
ax.text(1.45, 0.0, 'BISTABLE\n(two stable shared states, with an\nunstable state between them;\nhistory decides; hysteresis)', ha='center', va='center', fontsize=8.5, color='#006400', fontweight='bold')
ax.text(1.45, 1.7, "MONOSTABLE\n(one shared state, tilted\ntoward the primary's belief)", ha='center', va='center', fontsize=9, color='#8B0000', fontweight='bold')
ax.text(1.45, -1.7, "MONOSTABLE\n(one shared state, primary's\nbelief in the opposite direction)", ha='center', va='center', fontsize=9, color='#8B0000', fontweight='bold')
ax.set_xlabel('r = K - E - P/2  (mutual coupling precision relative to external precision; first order in P)', fontsize=10)
ax.set_ylabel("h = Pm/2  (the primary's fixed belief)", fontsize=10)
ax.set_title("Cusp phase diagram of the pair's shared belief", fontsize=13, pad=12)
ax.legend(loc='upper left', fontsize=9); ax.set_xlim(-1.2, 2.8); ax.set_ylim(-2.2, 2.2); ax.grid(True, alpha=.3)
plt.colorbar(im, ax=ax, ticks=[1, 2]).set_label('Number of stable equilibria')
fig.savefig(os.path.join(OUT, 'fig1_phase_diagram.png'), dpi=300, bbox_inches='tight'); plt.close(fig)

# ---------- Figure 3: full two-agent model checks ----------
E, K, P = 0.5, 1.2, 0.2
r = K - E - P/2; m_jump = 2*np.sqrt(4/27*r**3)/P
m_fold = cm.exact_fold(E, K, P, (-0.35, -0.53, 1.77))[0][2]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
ms_up = np.linspace(-2.5, 2.5, 251); ms_dn = ms_up[::-1]
q = np.array([-1.0, -1.0]); up = []
for mm in ms_up:
    q = cm.rk4_settle(q, E, K, P, mm, T=200); up.append(0.5*(q[0]+q[1]))
dn = []
for mm in ms_dn:
    q = cm.rk4_settle(q, E, K, P, mm, T=200); dn.append(0.5*(q[0]+q[1]))
a1.plot(ms_up, up, lw=2.2, label='m swept up'); a1.plot(ms_dn, dn, '--', lw=2.2, label='m swept down')
for s in (-1, 1):
    a1.axvline(s*m_jump, color='r', alpha=.6, lw=1.2)
    a1.axvline(s*m_fold, color='k', ls=':', lw=1.4)
a1.axhline(0, color='gray', ls=':', lw=1)
a1.set_xlabel("primary's position m"); a1.set_ylabel('shared belief  (q_p + q_s)/2')
a1.set_title(f'A. Hysteresis in the full model\n(dotted: full-model fold m = +/-{m_fold:.3f}; red: reduced prediction +/-{m_jump:.3f})', fontsize=10)
a1.legend(fontsize=8); a1.grid(True, alpha=.3)
dt, T1, T2 = 0.01, 20.0, 60.0
q = cm.settle([1.5, 1.5], E, K, K, P, 1.0); ts = [0.0]; qp = [q[0]]; qs = [q[1]]; t = 0.0
while t < T2:
    k = K if t < T1 else 0.0
    q = q + dt*cm.rhs(q, E, k, k, P, 1.0); t += dt
    ts.append(t); qp.append(q[0]); qs.append(q[1])
a2.plot(ts, qp, lw=2.2, label='primary'); a2.plot(ts, qs, lw=2.2, label='secondary')
a2.axvline(T1, color='k', ls='--', alpha=.6); a2.text(T1+0.8, 0.55, 'separation\n(K = 0)', fontsize=8)
a2.axhline(0, color='gray', ls=':', lw=1)
a2.set_xlabel('time'); a2.set_ylabel('belief (0 = consensus baseline)')
a2.set_title('B. Separation from a locked state (m = 1)', fontsize=10); a2.legend(fontsize=8); a2.grid(True, alpha=.3)
plt.tight_layout(); fig.savefig(os.path.join(OUT, 'fig3_full_model_checks.png'), dpi=300, bbox_inches='tight'); plt.close(fig)
print("saved figures to", OUT)
print(f"final secondary {qs[-1]:+.3f}, primary {qp[-1]:+.3f}")
