#!/usr/bin/env python3
"""fig_arc_scan.py -- schematic of the closed-loop arc-scan agent (Sec. 3.9).

True to scale in metres: route segment = circle of radius 100 m (the S-route
consists of two such semicircles), arc radius 10 m, seven candidates at
15 deg spacing (+-45 deg). The agent pose and the chosen candidate are
illustrative, not measured data.

    python fig_arc_scan.py            -> figures/fig_arc_scan.png / .pdf
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R_ROUTE, R_ARC = 100.0, 10.0
OFFS = [0, 15, -15, 30, -30, 45, -45]          # sampling order (right = +)
C_ROUTE, C_AGENT, C_ARC, C_SEL, C_CMP = "#c9a400", "#1a1a1a", "#1f5fa8", "#c0392b", "#2e7d32"


def route_xy(s):
    """Route point at arc length s: circle of radius 100 m, centre at
    (R, 0), travelling north at s = 0 and curving to the right."""
    a = s / R_ROUTE
    return R_ROUTE - R_ROUTE * math.cos(a), R_ROUTE * math.sin(a)


def route_bearing(s):
    return math.degrees(s / R_ROUTE)            # compass bearing of the tangent


def step(p, bearing_deg, d):
    b = math.radians(bearing_deg)
    return p[0] + d * math.sin(b), p[1] + d * math.cos(b)


def draw_route(ax, s0, s1):
    s = np.linspace(s0, s1, 200)
    xy = np.array([route_xy(v) for v in s])
    ax.plot(xy[:, 0], xy[:, 1], color=C_ROUTE, lw=1.8, zorder=1)
    for v in np.arange(math.ceil(s0 / 5) * 5, s1 + 1e-6, 5.0):
        ax.plot(*route_xy(v), "o", ms=2.6, color=C_ROUTE, mec="#7a6400", mew=0.5, zorder=2)


def arrow(ax, p, q, color, lw=1.4, ls="-", z=5):
    ax.annotate("", xy=q, xytext=p, zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                shrinkA=0, shrinkB=0, mutation_scale=9))


def draw_arc(ax, p, heading, color=C_ARC, label_order=False, chosen=None, alpha=1.0):
    ang = np.linspace(heading - 45, heading + 45, 60)
    pts = np.array([step(p, a, R_ARC) for a in ang])
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.2, alpha=alpha, zorder=3)
    for side in (-45, 45):
        q = step(p, heading + side, R_ARC)
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=0.6, ls=":", alpha=alpha, zorder=3)
    for k, off in enumerate(OFFS):
        q = step(p, heading + off, R_ARC)
        sel = chosen is not None and off == chosen
        ax.plot(*q, "o", ms=5.5 if sel else 4.2, zorder=6, alpha=alpha,
                color=C_SEL if sel else "white", mec=C_SEL if sel else color, mew=1.2)
        if label_order:
            t = step(p, heading + off, R_ARC + 2.7)
            ax.text(*t, str(k + 1), fontsize=7.5, ha="center", va="center", color=color)


fig, (a, b) = plt.subplots(1, 2, figsize=(5.9, 2.4))

# ---- (a) arc scan ----------------------------------------------------------
s_k, lat, head_err, chosen = 18.0, -7.0, -4.0, 30
rp = route_xy(s_k)
nb = route_bearing(s_k) + 90.0                       # right-hand normal
p = step(rp, nb, lat)
psi = route_bearing(s_k) + head_err
draw_route(a, 0, 60)
draw_arc(a, p, psi, label_order=True, chosen=chosen)
arrow(a, p, step(p, psi, 5.2), C_AGENT)
a.plot(*p, "s", ms=6, color=C_AGENT, zorder=7)
a.text(p[0] - 1.2, p[1] - 0.4, r"$\mathbf{p}_k$", fontsize=9, ha="right", va="top")
a.text(*step(p, psi + 2, 7.4), r"$\psi_k$", fontsize=9, ha="center", va="center")
q = step(p, psi + chosen, R_ARC)
a.annotate(r"selected: min. $g$", xy=q, xytext=(q[0] + 7.5, q[1] - 7), fontsize=8, color=C_SEL, va="center",
           arrowprops=dict(arrowstyle="-", color=C_SEL, lw=0.6, shrinkB=4))
m = step(p, psi - 45, R_ARC * 0.55)
a.text(m[0] - 1.6, m[1] - 2.2, r"$r=10\,$m", fontsize=8, color=C_ARC, ha="right")
w = route_xy(39)
a.annotate("taught route\n(5 m waypoints)", xy=w, xytext=(w[0] + 6, w[1] + 1.5), fontsize=7.5,
           color="#7a6400", va="center", arrowprops=dict(arrowstyle="-", color="#7a6400", lw=0.6))
a.set_title("(a) Arc scan at step $k$", fontsize=9, loc="left")

# ---- (b) heading update and evaluation -------------------------------------
draw_route(b, 0, 60)
draw_arc(b, p, psi, chosen=chosen, alpha=0.35)
b.plot(*p, "s", ms=5, color=C_AGENT, alpha=0.45, zorder=7)
arrow(b, p, q, C_SEL, lw=1.6)
psi_ext = psi + chosen
arrow(b, q, step(q, psi_ext, 13), C_SEL, lw=1.1, ls="--")
# compass: route bearing as recalled from memory at the chosen candidate
ts = np.linspace(0, 60, 600)
d = [math.hypot(q[0] - route_xy(v)[0], q[1] - route_xy(v)[1]) for v in ts]
s_near = float(ts[int(np.argmin(d))])
psi_cmp = route_bearing(s_near)
arrow(b, q, step(q, psi_cmp, 13), C_CMP, lw=1.1, ls="--")
e1, e2 = step(q, psi_ext, 14.2), step(q, psi_cmp, 14.2)
b.text(e1[0] + 0.8, e1[1], "extend:\n" + r"$\psi_k+\alpha_{i^*}$", fontsize=8, color=C_SEL, ha="left", va="center")
b.text(e2[0] - 2.0, e2[1] - 0.5, "compass:\n" + r"$\psi_k+\hat\theta_{k,i^*}$", fontsize=8, color=C_CMP,
       ha="right", va="center")
# ground-truth evaluation: nearest route point, cross-track distance
n = route_xy(s_near)
b.plot([q[0], n[0]], [q[1], n[1]], color=C_AGENT, lw=1.3, zorder=6)
b.plot(*n, "x", ms=6, color=C_AGENT, mew=1.4, zorder=7)
b.annotate(r"cross-track $x_{k+1}$", xy=((q[0] + n[0]) / 2, (q[1] + n[1]) / 2), xytext=(n[0] + 4.0, n[1] - 5.0),
           fontsize=7.5, va="center", arrowprops=dict(arrowstyle="-", color=C_AGENT, lw=0.6))
b.annotate("nearest route point,\n" + r"arc length $s_{k+1}$", xy=n, xytext=(n[0] + 4.0, n[1] - 12.0), fontsize=7.5,
           va="center", arrowprops=dict(arrowstyle="-", color=C_AGENT, lw=0.6))
b.text(q[0] - 1.2, q[1] + 3.0, r"$\mathbf{p}_{k+1}$", fontsize=9, ha="right", va="bottom")
b.set_title("(b) Heading update and evaluation", fontsize=9, loc="left")

for ax in (a, b):
    ax.set_aspect("equal")
    ax.set_xlim(-21, 31)
    ax.set_ylim(12, 46)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color("#888888")
    # scale bar
    ax.plot([-18.5, -8.5], [12.5, 12.5], color="k", lw=1.5)
    ax.text(-13.5, 13.5, "10 m", fontsize=7.5, ha="center")

fig.tight_layout(w_pad=0.8)
out = Path("figures")
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "fig_arc_scan.png", dpi=400, bbox_inches="tight", pad_inches=0.02)
fig.savefig(out / "fig_arc_scan.pdf", bbox_inches="tight", pad_inches=0.02)
print("ok", out)
