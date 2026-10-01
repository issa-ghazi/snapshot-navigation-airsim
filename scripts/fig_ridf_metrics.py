#!/usr/bin/env python3
"""Schematic figure defining the RIDF-based diagnostic measures used in
Section 3.6.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.linewidth": 0.8,
})

# Same angular sampling as used in the evaluation:
# 360 rotations with a step size of 1 degree.
th = np.arange(-180.0, 180.0, 1.0)

BLUE = "#1f4e79"
GRAY = "#8c8c8c"
RED = "#b5443f"
GREEN = "#4f6228"


def ridf(depth, centre, width, base, ripple=0.03, phase=0.0):
    """Construct a periodic schematic RIDF curve."""
    d = np.minimum(
        np.abs(th - centre),
        360.0 - np.abs(th - centre)
    )

    dip = depth * np.exp(-(d ** 2) / (2.0 * width ** 2))
    wob = ripple * np.cos(np.radians(2.0 * (th - phase)))

    return base * (1.0 - dip + wob)


# Schematic RIDF of the ground-truth scene and its strongest competitor.
true_c = ridf(
    depth=0.55,
    centre=-14.0,
    width=42.0,
    base=100.0,
    phase=30.0,
)

comp_c = ridf(
    depth=0.30,
    centre=118.0,
    width=55.0,
    base=100.0,
    ripple=0.025,
    phase=-40.0,
)

# Quantities defined in Section 3.6.
d_star = true_c.min()
th_star = th[true_c.argmin()]
mean_true = true_c.mean()

d_comp = comp_c.min()
th_comp = th[comp_c.argmin()]

delta = (mean_true - d_star) / mean_true
alias_ratio = d_star / d_comp


fig, ax = plt.subplots(figsize=(6.3, 3.1))

# RIDF curves.
ax.plot(th, true_c, color=BLUE, lw=1.7)
ax.plot(th, comp_c, color=GRAY, lw=1.3)

ax.text(
    -176,
    true_c[0] + 4,
    r"true scene $j^{\star}$",
    color=BLUE,
    fontsize=8.5,
    ha="left",
    va="bottom",
    bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
)

ax.text(
    52,
    104,
    r"best competing scene $j_c$",
    color=GRAY,
    fontsize=8.5,
    ha="left",
    va="bottom",
    bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
)

# Mean difference of the ground-truth scene over all rotations.
ax.axhline(
    mean_true,
    color=BLUE,
    lw=0.8,
    ls=(0, (4, 3)),
)

ax.text(
    178,
    mean_true + 2,
    r"$\overline{D}_{j^{\star}}$",
    color=BLUE,
    ha="right",
    va="bottom",
    fontsize=8.5,
)

# Minimum of the ground-truth scene.
ax.hlines(
    d_star,
    th_star,
    178,
    color=BLUE,
    lw=0.7,
    ls=(0, (1.5, 2)),
)

ax.plot(
    [th_star],
    [d_star],
    "o",
    ms=4.5,
    color=BLUE,
)

ax.text(
    178,
    d_star - 2,
    r"$d^{\star}$",
    color=BLUE,
    ha="right",
    va="top",
    fontsize=9,
)

# Minimum of the best competing scene.
ax.hlines(
    d_comp,
    th_comp,
    178,
    color=GRAY,
    lw=0.7,
    ls=(0, (1.5, 2)),
)

ax.plot(
    [th_comp],
    [d_comp],
    "o",
    ms=4.5,
    color=GRAY,
)

ax.text(
    178,
    d_comp + 2,
    r"$d_c$",
    color=GRAY,
    ha="right",
    va="bottom",
    fontsize=9,
)

# RIDF depth.
ax.annotate(
    "",
    xy=(th_star, d_star),
    xytext=(th_star, mean_true),
    arrowprops=dict(
        arrowstyle="<->",
        color=RED,
        lw=1.1,
    ),
)

ax.annotate(
    r"$\delta="
    r"\dfrac{\overline{D}_{j^{\star}}-d^{\star}}"
    r"{\overline{D}_{j^{\star}}}$",
    xy=(th_star - 1, (d_star + mean_true) / 2),
    xytext=(-115, 22),
    color=RED,
    ha="center",
    va="center",
    fontsize=8.5,
    arrowprops=dict(
        arrowstyle="->",
        color=RED,
        lw=0.7,
        shrinkB=4,
    ),
)

# Alias ratio. The corresponding quantities d* and d_c are marked by
# the horizontal guides above; no distance between them is implied.
ax.text(
    112,
    57,
    r"$a=\dfrac{d^{\star}}{d_c}$",
    color=GREEN,
    ha="center",
    va="center",
    fontsize=8.5,
)

# Rotation of the minimum of the ground-truth scene.
ax.vlines(
    th_star,
    0,
    d_star,
    color=BLUE,
    lw=0.7,
    ls=(0, (1.5, 2)),
)

ax.text(
    th_star - 4,
    3,
    r"$\theta_{\mathrm{true}}$",
    color=BLUE,
    ha="right",
    va="bottom",
    fontsize=8.5,
)

# Axes.
ax.set_xlim(-180, 180)
ax.set_xticks([-180, -90, 0, 90, 180])
ax.set_xlabel(r"image rotation $\theta$ (deg)")
ax.set_ylabel(r"image difference $D(\theta,j)$")

ax.set_ylim(0, 122)
ax.set_yticks([0])
ax.set_yticklabels(["0"])

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout(pad=0.4)

out = Path("figures/fig_ridf_metrics.png")
out.parent.mkdir(parents=True, exist_ok=True)

import os
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig_ridf_metrics.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(
    f"saved: {out}\n"
    f"theta_true = {th_star:.1f} deg\n"
    f"depth = {delta:.3f}\n"
    f"alias ratio = {alias_ratio:.3f}"
)