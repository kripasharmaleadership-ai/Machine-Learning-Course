"""Draws assets/banner.png (1280 x 640, the size GitHub uses for a repository's social preview).

    python assets/make_banner.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

MODULES = ["ML\nFoundations", "Data\nPreparation", "Regression", "Classification", "Trees\n& XAI",
           "Unsupervised\n& Embeddings", "Neural Nets\n& TinyGPT", "GenAI, RAG\n& Agents"]
COLORS = ["#4C78A8"] * 5 + ["#E45756"] * 2 + ["#F58518"]           # classical ML / representations / GenAI

fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1280)
ax.set_ylim(0, 640)
ax.axis("off")
gradient = np.linspace(0, 1, 256).reshape(1, -1)
ax.imshow(gradient, extent=(0, 1280, 0, 640), cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
    "bg", ["#0d1b2a", "#1b263b", "#243b55"]), aspect="auto", zorder=0)

rng = np.random.default_rng(42)                                       # a faint scatter "dataset" in the background
pts = rng.normal(size=(160, 2)) * [170, 60] + [980, 520]
ax.scatter(pts[:, 0], pts[:, 1], s=8, color="#9fb3c8", alpha=0.25, zorder=1)
xs = np.linspace(780, 1200, 50)
ax.plot(xs, 520 + 0.35 * (xs - 990), color="#9fb3c8", alpha=0.35, lw=2, zorder=1)

ax.text(60, 520, "Machine Learning Course", color="white", fontsize=44, weight="bold", va="center", zorder=2)
ax.text(62, 455, "From first principles to RAG and AI agents", color="#d9e2ec", fontsize=22, va="center", zorder=2)
ax.text(62, 405, "105 runnable labs  ·  8 modules  ·  offline  ·  CPU only  ·  known ground truth",
        color="#9fb3c8", fontsize=15, va="center", zorder=2)

w, gap, y = 132, 18, 170
x0 = (1280 - (8 * w + 7 * gap)) / 2
for i, (name, col) in enumerate(zip(MODULES, COLORS)):
    x = x0 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, y), w, 120, boxstyle="round,pad=0,rounding_size=14", fc=col, ec="none",
                                alpha=0.92, zorder=2))
    ax.text(x + w / 2, y + 92, f"{i + 1}", color="white", fontsize=20, weight="bold", ha="center", va="center", zorder=3)
    ax.text(x + w / 2, y + 45, name, color="white", fontsize=11.5, ha="center", va="center", zorder=3, linespacing=1.2)
    if i < 7:
        ax.annotate("", xy=(x + w + gap - 2, y + 60), xytext=(x + w + 2, y + 60),
                    arrowprops=dict(arrowstyle="-|>", color="#d9e2ec", lw=1.5), zorder=3)
for label, i0, i1, col in [("classical ML", 0, 4, "#9ecae9"), ("representations", 5, 6, "#ff9d9a"), ("GenAI", 7, 7, "#ffbf79")]:
    xa, xb = x0 + i0 * (w + gap), x0 + i1 * (w + gap) + w
    ax.plot([xa, xb], [y - 22, y - 22], color=col, lw=3, solid_capstyle="round", zorder=2)
    ax.text((xa + xb) / 2, y - 50, label, color=col, fontsize=13, ha="center", va="center", zorder=2)

ax.text(1220, 40, "scikit-learn · pandas · NumPy · PyTorch", color="#7b8fa6", fontsize=12, ha="right", zorder=2)
out = Path(__file__).resolve().parent / "banner.png"
fig.savefig(out, dpi=100)
print(f"saved {out}")
