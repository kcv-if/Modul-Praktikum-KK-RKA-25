"""Particle Swarm Optimization helpers for the Evolutionary Computation notebook.

Holds the optimizer, the benchmark functions, and the explainer-style plotting
and animation helpers, so the notebook cells stay short.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch
from IPython.display import HTML

FIG_DIR = Path(__file__).resolve().parent / "figures"
HAS_FFMPEG = animation.writers.is_available("ffmpeg")

# One palette for plots and for the \color{} terms in the notebook equations.
C_INERTIA = "#3A7DCE"
C_COG = "#E8871E"
C_SOC = "#C2185B"
C_RES = "#2E7D32"
C_PARTICLE = "#1B1B1B"
C_LABEL = "#9A9A9A"
C_TEXT = "#4A4A4A"

# Viridis blended with white so dark particles stay visible on every level.
_base = plt.get_cmap("viridis")(np.linspace(0, 1, 256))
CMAP = ListedColormap(0.5 * _base + 0.5 * np.array([1, 1, 1, 1]), name="viridis_soft")

RASTRIGIN_BOUNDS = (-5.12, 5.12)

# Parameter settings compared in section 4 (and marked on the stability plot).
CONFIGS = [
    dict(name="balanced", label="seimbang", w=0.7, c1=1.5, c2=1.5, w_end=None, color="#333333"),
    dict(name="small_w", label="w kecil", w=0.2, c1=1.5, c2=1.5, w_end=None, color="#8AB6E8"),
    dict(name="large_w", label="w besar", w=1.0, c1=1.5, c2=1.5, w_end=None, color="#1F4E8C"),
    dict(name="cognitive", label="kognitif dominan", w=0.7, c1=2.5, c2=0.5, w_end=None, color=C_COG),
    dict(name="social", label="sosial dominan", w=0.7, c1=0.5, c2=2.5, w_end=None, color=C_SOC),
    dict(name="ldw", label="w turun 0.9→0.4", w=0.9, c1=1.5, c2=1.5, w_end=0.4, color=C_RES),
]


def config_title(cfg):
    w = f"{cfg['w']}→{cfg['w_end']}" if cfg["w_end"] is not None else f"{cfg['w']}"
    return f"{cfg['label']}\nw={w}, c₁={cfg['c1']}, c₂={cfg['c2']}"


# --------------------------------------------------------------------------
# Benchmark functions: x has shape (..., dim), result has shape (...).
# --------------------------------------------------------------------------

def rastrigin(x):
    x = np.asarray(x, dtype=float)
    return 10 * x.shape[-1] + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=-1)


def sphere(x):
    return np.sum(np.asarray(x, dtype=float) ** 2, axis=-1)


# --------------------------------------------------------------------------
# Optimizer
# --------------------------------------------------------------------------

def pso(f, dim, bounds, n_particles=30, iters=100, w=0.7298, c1=1.49618, c2=1.49618,
        w_end=None, vmax_frac=0.2, seed=None, record=False):
    """Minimize f with a synchronous, vectorized global-best PSO.

    w_end: if given, inertia decreases linearly from w to w_end.
    vmax_frac: velocity clamp as a fraction of the range; None means no clamp.
    record: if True, keep every position/velocity/pbest/gbest and the random
    factors r1, r2 so a single update step can be replayed for animation.
    Index t of a history array is the state after t iterations.
    """
    rng = np.random.default_rng(seed)
    lo = np.broadcast_to(np.asarray(bounds[0], dtype=float), (dim,))
    hi = np.broadcast_to(np.asarray(bounds[1], dtype=float), (dim,))
    # vmax_frac=None disables the velocity clamp; start velocities still use 0.2 * range.
    vmax = vmax_frac * (hi - lo) if vmax_frac is not None else np.full(dim, np.inf)
    v0 = vmax if vmax_frac is not None else 0.2 * (hi - lo)

    x = rng.uniform(lo, hi, (n_particles, dim))
    v = rng.uniform(-v0, v0, (n_particles, dim))
    p, fp = x.copy(), f(x)
    g_idx = np.argmin(fp)
    g, fg = p[g_idx].copy(), fp[g_idx]

    gbest_f, diversity, w_used = [fg], [np.linalg.norm(x - g, axis=1).mean()], []
    hist = {k: [] for k in ("X", "V", "P", "G", "R1", "R2")} if record else None
    if record:
        for k, a in zip("XVPG", (x, v, p, g)):
            hist[k].append(a.copy())

    for t in range(iters):
        wt = w if w_end is None else w + (w_end - w) * t / max(iters - 1, 1)
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        v = wt * v + c1 * r1 * (p - x) + c2 * r2 * (g - x)
        v = np.clip(v, -vmax, vmax)
        x = np.clip(x + v, lo, hi)
        fx = f(x)

        better = fx < fp
        p[better], fp[better] = x[better], fx[better]
        g_idx = np.argmin(fp)
        if fp[g_idx] < fg:
            g, fg = p[g_idx].copy(), fp[g_idx]

        gbest_f.append(fg)
        diversity.append(np.linalg.norm(x - g, axis=1).mean())
        w_used.append(wt)
        if record:
            for k, a in zip("XVPG", (x, v, p, g)):
                hist[k].append(a.copy())
            hist["R1"].append(r1)
            hist["R2"].append(r2)

    out = dict(gbest=g, gbest_f=fg, curve=np.array(gbest_f), diversity=np.array(diversity),
               W=np.array(w_used), params=dict(c1=c1, c2=c2, vmax=vmax, lo=lo, hi=hi))
    if record:
        out.update({k: np.array(a) for k, a in hist.items()})
    return out


def run_experiment(configs, f, dim, bounds, n_particles, iters, seeds):
    """Run every config on the same seeds; return {name: (curves, diversity)}."""
    results = {}
    for cfg in configs:
        runs = [pso(f, dim, bounds, n_particles, iters, cfg["w"], cfg["c1"], cfg["c2"],
                    cfg["w_end"], seed=s) for s in seeds]
        results[cfg["name"]] = (np.array([r["curve"] for r in runs]),
                                np.array([r["diversity"] for r in runs]))
    return results


# --------------------------------------------------------------------------
# Static plotting
# --------------------------------------------------------------------------

def apply_style():
    """Explainer look: white background, no frame, no grid, light-grey labels."""
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.spines.left": False, "axes.spines.right": False,
        "axes.spines.top": False, "axes.spines.bottom": False,
        "axes.grid": False, "axes.labelcolor": C_LABEL, "axes.titlecolor": C_TEXT,
        "xtick.color": C_LABEL, "ytick.color": C_LABEL,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "text.color": C_TEXT,
        "font.size": 10, "axes.titlesize": 10, "legend.frameon": False,
        # JPEG frames keep the inline to_jshtml players small.
        "animation.frame_format": "jpeg", "animation.embed_limit": 40,
    })


def draw_landscape(ax, f, bounds, optimum=(0.0, 0.0), res=300):
    lo, hi = bounds
    grid = np.linspace(lo, hi, res)
    X, Y = np.meshgrid(grid, grid)
    Z = f(np.stack([X, Y], axis=-1))
    ax.contourf(X, Y, Z, levels=40, cmap=CMAP, zorder=0)
    ax.contour(X, Y, Z, levels=12, colors="white", linewidths=0.4, alpha=0.7, zorder=1)
    ax.plot(*optimum, marker="x", color="white", ms=8, mew=2.2, zorder=2)
    ax.plot(*optimum, marker="x", color=C_PARTICLE, ms=6, mew=1.0, zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ticks = [round(lo * 0.98), 0, round(hi * 0.98)]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)


def plot_snapshots(hist, f, bounds, at=(0, 5, 20, 50), title=None):
    fig, axes = plt.subplots(1, len(at), figsize=(3.2 * len(at), 3.4))
    for ax, t in zip(axes, at):
        t = min(t, len(hist["X"]) - 1)
        draw_landscape(ax, f, bounds)
        ax.scatter(*hist["X"][t].T, s=12, c=C_PARTICLE, zorder=4)
        ax.scatter(*hist["G"][t], s=90, marker="*", c=C_SOC, edgecolors="white", lw=0.6, zorder=5)
        ax.set_title(f"iterasi {t}   gbest={hist['curve'][t]:.3g}", color=C_LABEL, fontsize=9)
    if title:
        fig.suptitle(title, color=C_TEXT)
    fig.tight_layout()
    return fig


def plot_stability(configs, clerc=(0.7298, 2 * 1.49618)):
    """Poli (2009) region -1<w<1, 0<c1+c2<24(1-w^2)/(7-5w) with configs marked."""
    w = np.linspace(-1, 1, 400)
    bound = 24 * (1 - w**2) / (7 - 5 * w)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.fill_between(w, 0, bound, color=C_INERTIA, alpha=0.12, lw=0)
    ax.plot(w, bound, color=C_INERTIA, lw=1.2)
    ax.text(-0.35, 1.2, "stabil (orde-2)", color=C_INERTIA, fontsize=10)
    ax.text(0.35, 4.6, r"$c_1+c_2=\frac{24(1-w^2)}{7-5w}$", color=C_INERTIA, fontsize=11)

    # Hand-placed labels (points/offset, ha): the points share y = 3 and sit close together.
    place = {"small_w": ((0, -16), "center"), "large_w": ((0, -16), "center"),
             "balanced": ((-12, 10), "right"), "ldw": ((-2, 9), "left")}
    grouped = {}
    for cfg in configs:
        if cfg["w_end"] is not None:
            c = cfg["c1"] + cfg["c2"]
            ax.annotate("", xy=(cfg["w_end"], c), xytext=(cfg["w"], c),
                        arrowprops=dict(arrowstyle="-|>", color=cfg["color"], lw=1.6))
            (dx, dy), ha = place.get(cfg["name"], ((0, 9), "center"))
            ax.annotate(cfg["label"], (cfg["w"] - 0.16, c), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, fontsize=9, color=cfg["color"])
            continue
        grouped.setdefault((cfg["w"], cfg["c1"] + cfg["c2"]), []).append(cfg)
    for (wv, c), cfgs in grouped.items():
        ax.scatter(wv, c, s=40, color=cfgs[0]["color"], zorder=3, edgecolors="white")
        (dx, dy), ha = place.get(cfgs[0]["name"], ((0, 9), "center"))
        ax.annotate(", ".join(k["label"] for k in cfgs), (wv, c), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, fontsize=9, color=cfgs[0]["color"])
    ax.scatter(*clerc, s=40, marker="D", color=C_TEXT, zorder=3, edgecolors="white")
    ax.annotate("Clerc χ", clerc, xytext=(-2, -16), textcoords="offset points", ha="right",
                fontsize=9, color=C_TEXT)
    ax.set_xlim(-1.05, 1.15)
    ax.set_ylim(0, 5.2)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_xlabel("w")
    ax.set_ylabel("c₁ + c₂")
    fig.tight_layout()
    return fig


def plot_spring(ws=(0.2, 0.7, 1.0), c1=1.5, c2=1.5, steps=40, x0=4.0, n_noisy=6, seed=0):
    """One particle in 1D with p = g = 0 fixed: x(t+1) = (1 + w - k) x(t) - w x(t-1).

    Thick line: r1, r2 replaced by their mean 0.5. Thin lines: real random r1, r2.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(steps + 1)
    fig, axes = plt.subplots(1, len(ws), figsize=(3.6 * len(ws), 2.8), sharey=True)
    for ax, w in zip(axes, ws):
        for noisy in [False] + [True] * n_noisy:
            x = np.empty(steps + 1)
            x[0] = x_prev = x0
            for k in range(steps):
                r1, r2 = rng.random(2) if noisy else (0.5, 0.5)
                kappa = c1 * r1 + c2 * r2
                x[k + 1], x_prev = (1 + w - kappa) * x[k] - w * x_prev, x[k]
            ax.plot(t, x, color=C_LABEL if noisy else C_INERTIA, lw=0.6 if noisy else 2,
                    alpha=0.6 if noisy else 1, zorder=1 if noisy else 2)
        ax.axhline(0, color=C_LABEL, lw=0.6, ls=":")
        ax.set_title(f"w = {w}", color=C_TEXT)
        ax.set_xlabel("iterasi")
        ax.set_ylim(-8, 8)
    axes[0].set_ylabel("posisi x")
    fig.tight_layout()
    return fig


def plot_speed(series):
    """series: list of (label, color, speeds array (seeds, iters+1)); median over seeds."""
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for label, color, speeds in series:
        med = np.median(speeds, axis=0)
        ax.plot(med, color=color, lw=1.8)
        ax.text(len(med) - 1 + 3, med[-1], label, color=color, fontsize=9, va="center")
    ax.set_yscale("log")
    ax.set_xlabel("iterasi")
    ax.set_title("rata-rata |v| (median dari 10 seed)", color=C_TEXT, loc="left")
    ax.set_xlim(0, len(med) * 1.45)
    ax.set_xticks(range(0, len(med), 50))
    fig.tight_layout()
    return fig


def plot_experiment(results, configs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for cfg in configs:
        curves, div = results[cfg["name"]]
        it = np.arange(curves.shape[1])
        for ax, data in zip(axes, (curves, div)):
            q25, med, q75 = np.percentile(data, [25, 50, 75], axis=0)
            ax.fill_between(it, q25, q75, color=cfg["color"], alpha=0.15, lw=0)
            ax.plot(it, med, color=cfg["color"], lw=1.6, label=cfg["label"])
    for ax, title in zip(axes, ("gbest f(x)  (median, pita 25–75%)",
                                "diversitas: rata-rata jarak ke gbest")):
        ax.set_yscale("log")
        ax.set_xlabel("iterasi")
        ax.set_title(title, color=C_TEXT, loc="left")
    axes[0].legend(fontsize=8, labelcolor="linecolor", loc="upper right", ncol=2)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Animation
# --------------------------------------------------------------------------

def _ease(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3 - 2 * s)


def _interp(X, k, substeps):
    t, s = divmod(k, substeps)
    if t >= len(X) - 1:
        return X[-1]
    return X[t] + (X[t + 1] - X[t]) * (s / substeps)


def animate_swarms(hists, f, bounds, titles=None, ncols=None, iters=None, substeps=3,
                   fps=20, trail=6, panel=4.4, dpi=80, hold=1.0):
    """Swarm(s) moving over the landscape; several hists share frame numbers."""
    n = len(hists)
    ncols = ncols or n
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel * ncols, panel * nrows), dpi=dpi,
                             squeeze=False)
    T = iters or min(len(h["X"]) for h in hists) - 1
    n_frames = T * substeps + 1 + int(hold * fps)
    artists = []
    for idx, ax in enumerate(axes.flat):
        if idx >= n:
            ax.set_visible(False)
            continue
        draw_landscape(ax, f, bounds)
        if titles:
            ax.set_title(titles[idx], fontsize=9, color=C_TEXT)
        lc = LineCollection([], linewidths=1.1, zorder=3)
        ax.add_collection(lc)
        sc = ax.scatter([], [], s=14, c=C_PARTICLE, zorder=4)
        gb = ax.scatter([], [], s=90, marker="*", c=C_SOC, edgecolors="white", lw=0.6, zorder=5)
        txt = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=8, color=C_TEXT,
                      bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.5), zorder=6)
        artists.append((lc, sc, gb, txt))
    fig.tight_layout()

    def update(k):
        k = min(k, T * substeps)
        t = k // substeps
        for h, (lc, sc, gb, txt) in zip(hists, artists):
            X = h["X"][: T + 1]
            pts = _interp(X, k, substeps)
            sc.set_offsets(pts)
            path = np.stack([_interp(X, j, substeps) for j in range(max(0, k - trail), k + 1)])
            if len(path) > 1:
                segs = np.stack([path[:-1], path[1:]], axis=2)  # (L-1, N, 2, 2)
                L = len(segs)
                alpha = np.repeat(np.linspace(0.05, 0.55, L), segs.shape[1])
                rgba = np.zeros((alpha.size, 4))
                rgba[:, :3] = 0.1
                rgba[:, 3] = alpha
                lc.set_segments(segs.reshape(-1, 2, 2))
                lc.set_color(rgba)
            else:
                lc.set_segments([])
            gb.set_offsets(h["G"][t][None])
            txt.set_text(f"iterasi {t}   gbest = {h['curve'][t]:.3g}")
        return []

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)
    plt.close(fig)
    return anim


def pick_particle(hist, t):
    """Pick a particle whose three velocity terms are similar in size and unclipped."""
    c1, c2, vmax, lo, hi = (hist["params"][k] for k in ("c1", "c2", "vmax", "lo", "hi"))
    x = hist["X"][t]
    a = hist["W"][t] * hist["V"][t]
    b = c1 * hist["R1"][t] * (hist["P"][t] - x)
    c = c2 * hist["R2"][t] * (hist["G"][t] - x)
    raw = a + b + c
    norms = np.stack([np.linalg.norm(v, axis=1) for v in (a, b, c)], axis=1)
    score = norms.min(axis=1) / (norms.max(axis=1) + 1e-12)
    ok = (np.all(np.abs(raw) <= vmax, axis=1)
          & np.all((x + raw >= lo) & (x + raw <= hi), axis=1)
          & (np.linalg.norm(hist["P"][t] - hist["G"][t], axis=1) > 0.3))
    score[~ok] = -1
    return int(np.argmax(score))


def animate_update_step(hist, f, bounds, t, i=None, fps=20, dpi=80):
    """Freeze the swarm at iteration t, zoom on particle i and build v(t+1) tip-to-tail."""
    i = pick_particle(hist, t) if i is None else i
    c1, c2 = hist["params"]["c1"], hist["params"]["c2"]
    x = hist["X"][t][i]
    p, g = hist["P"][t][i], hist["G"][t]
    a = hist["W"][t] * hist["V"][t][i]
    b = c1 * hist["R1"][t][i] * (p - x)
    c = c2 * hist["R2"][t][i] * (g - x)
    v_new = a + b + c
    assert np.allclose(x + v_new, hist["X"][t + 1][i]), "picked particle was clipped"

    pts = np.array([x, x + a, x + a + b, x + v_new, p, g])
    center = (pts.min(0) + pts.max(0)) / 2
    span = max(np.ptp(pts, axis=0).max() * 1.45, 1.2)
    lo, hi = bounds
    center = np.clip(center, lo + span / 2, hi - span / 2) if span < hi - lo else center
    full = np.array([lo, hi, lo, hi], dtype=float)
    zoom = np.array([center[0] - span / 2, center[0] + span / 2,
                     center[1] - span / 2, center[1] + span / 2])

    phases = [("hold", 0.8), ("zoom", 1.2), ("inertia", 1.1), ("cog", 1.1), ("soc", 1.1),
              ("res", 1.1), ("move", 1.1), ("end", 1.2)]
    starts = np.cumsum([0] + [d for _, d in phases])
    n_frames = int(starts[-1] * fps)
    captions = {"hold": "kawanan dibekukan pada iterasi %d" % t, "zoom": "fokus pada satu partikel",
                "inertia": "1. inersia: lanjutkan arah lama",
                "cog": "2. kognitif: tarik ke pbest sendiri",
                "soc": "3. sosial: tarik ke gbest kawanan",
                "res": "4. jumlahkan: kecepatan baru", "move": "5. pindah ke posisi baru",
                "end": "5. pindah ke posisi baru"}

    fig, ax = plt.subplots(figsize=(6, 6.3), dpi=dpi)
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.06, top=0.9)
    draw_landscape(ax, f, bounds)
    ax.set_xticks([])
    ax.set_yticks([])
    caption = fig.text(0.5, 0.95, "", ha="center", va="center", fontsize=13, color=C_TEXT)
    others = ax.scatter(*np.delete(hist["X"][t], i, axis=0).T, s=14, c=C_PARTICLE, zorder=4)
    pb = ax.scatter(*p, s=110, facecolors="none", edgecolors=C_COG, lw=2, zorder=5)
    gb = ax.scatter(*g, s=200, marker="*", c=C_SOC, edgecolors="white", lw=0.8, zorder=5)
    me = ax.scatter(*x, s=55, c=C_PARTICLE, edgecolors="white", lw=1, zorder=8)
    ghost = ax.scatter(*x, s=55, facecolors="none", edgecolors=C_PARTICLE, lw=1, zorder=7)
    lead = dict(lw=1.4, ls=(0, (4, 3)), zorder=3)
    lp, = ax.plot(*np.c_[x, p], color=C_COG, **lead)
    lg, = ax.plot(*np.c_[x, g], color=C_SOC, **lead)

    off = 0.045 * span
    lbl = dict(fontsize=12, ha="center", va="center", zorder=9,
               bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
    pb_t = ax.text(*(p + [0, 1.6 * off]), "pbest", color=C_COG, **{**lbl, "fontsize": 10})
    gb_t = ax.text(*(g + [0, 1.6 * off]), "gbest", color=C_SOC, **{**lbl, "fontsize": 10})

    centroid = pts[:4].mean(0)
    arrows = {}
    for name, start, vec, color, tex in [
        ("inertia", x, a, C_INERTIA, r"$w\,\mathbf{v}_i$"),
        ("cog", x + a, b, C_COG, r"$c_1 r_1(\mathbf{p}_i-\mathbf{x}_i)$"),
        ("soc", x + a + b, c, C_SOC, r"$c_2 r_2(\mathbf{g}-\mathbf{x}_i)$"),
        ("res", x, v_new, C_RES, r"$\mathbf{v}_i^{t+1}$"),
    ]:
        arr = FancyArrowPatch(start, start, arrowstyle="-|>", mutation_scale=16, lw=2.4,
                              color=color, shrinkA=0, shrinkB=0, zorder=6)
        ax.add_patch(arr)
        mid = start + vec / 2
        normal = np.array([-vec[1], vec[0]]) / (np.linalg.norm(vec) + 1e-12)
        if np.dot(normal, mid - centroid) < 0:
            normal = -normal
        txt = ax.text(*(mid + 2.2 * off * normal), tex, color=color, **lbl)
        arrows[name] = (arr, txt, start, vec)

    def prog(name, time):
        k = [n for n, _ in phases].index(name)
        return float(np.clip((time - starts[k]) / dict(phases)[name], 0, 1))

    def update(k):
        time = k / fps
        current = min(np.searchsorted(starts, time, side="right") - 1, len(phases) - 1)
        caption.set_text(captions[phases[current][0]])
        z = _ease(prog("zoom", time))
        lim = full + z * (zoom - full)
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
        others.set_alpha(1 - 0.75 * z)

        fade = 1 - 0.65 * _ease(prog("move", time))
        for name, (arr, txt, start, vec) in arrows.items():
            s = _ease(prog(name, time))
            arr.set_visible(s > 0)
            arr.set_positions(start, start + s * vec)
            alpha = s if name == "res" else s * fade
            arr.set_alpha(alpha)
            txt.set_alpha(_ease(2 * prog(name, time) - 1) * (1 if name == "res" else fade))
            txt.get_bbox_patch().set_alpha(0.75 * txt.get_alpha())
        for art, name in ((pb, "cog"), (pb_t, "cog"), (lp, "cog"), (gb, "soc"), (gb_t, "soc"), (lg, "soc")):
            s = _ease(2 * prog(name, time))
            art.set_alpha(s * (0.9 if art in (lp, lg) else 1))
            if art in (pb_t, gb_t):
                art.get_bbox_patch().set_alpha(0.75 * s)
        m = _ease(prog("move", time))
        me.set_offsets((x + m * v_new)[None])
        ghost.set_alpha(0.6 * (m > 0))
        return []

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)
    plt.close(fig)
    return anim, i


def show_and_export(anim, name, fps=20):
    """Save figures/<name>.mp4 (if ffmpeg exists) and .gif, return the inline player."""
    FIG_DIR.mkdir(exist_ok=True)
    if HAS_FFMPEG:
        # yuv420p so the mp4 plays in browsers and QuickTime.
        anim.save(FIG_DIR / f"{name}.mp4",
                  writer=animation.FFMpegWriter(fps=fps, bitrate=1800, extra_args=["-pix_fmt", "yuv420p"]))
    anim.save(FIG_DIR / f"{name}.gif", writer=animation.PillowWriter(fps=fps))
    return HTML(anim.to_jshtml(fps=fps, default_mode="loop"))
