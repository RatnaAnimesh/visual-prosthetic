"""Render the model's results as figures.

One figure per claim, in the layout a visual physiologist would recognise.
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

RESULTS = pathlib.Path("results")
FIGS = pathlib.Path("figures")
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 9, "figure.facecolor": "white",
})

LAYER_COLOURS = {
    "L4Cb_ss": "#1b6ca8", "L4Ca_ss": "#d1495b", "L4B_pyr": "#e07a5f",
    "L23_pyr": "#2a9d8f", "L5_pyr": "#7b539b", "L6_pyr": "#8a6d3b",
    "L4_pv": "#999999", "L23_pv": "#bbbbbb", "L5_ib": "#a05195",
}
NICE = {
    "L4Cb_ss": "IVCβ spiny stellate", "L4Ca_ss": "IVCα spiny stellate",
    "L4B_pyr": "IVB pyramid", "L23_pyr": "II/III pyramid",
    "L5_pyr": "V pyramid", "L6_pyr": "VI pyramid",
    "L4_pv": "IVC basket (FS)", "L23_pv": "II/III basket (FS)",
    "L5_ib": "V bursting pyramid",
}


def _load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def fig_orientation():
    d = _load("orientation_tuning.json")
    if d is None:
        return
    oris = np.array(d["orientations"])
    # Aligned curves are stored with the preferred orientation first; put it
    # in the middle and wrap, which is how tuning curves are normally drawn.
    n = len(oris)
    x = (np.arange(n) - n // 2) * (180.0 / n)

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.9))

    ax = axes[0]
    for b in ["L4Cb_ss", "L23_pyr", "L5_pyr", "L6_pyr"]:
        if b not in d["layers"]:
            continue
        c = np.array(d["layers"][b]["aligned_curve"])
        ax.plot(x, np.roll(c, n // 2), "o-", ms=3, lw=1.4,
                color=LAYER_COLOURS[b], label=NICE[b])
    ax.set_xlabel("orientation relative to preferred (deg)")
    ax.set_ylabel("firing rate (spikes/s)")
    ax.set_title("Orientation tuning by layer")
    ax.legend(frameon=False, fontsize=6.5)

    ax = axes[1]
    # Normalise by the peak only.  Subtracting the minimum as well would make
    # every curve span the full range and hide the very thing this panel is
    # about: the fast-spiking cells keep firing hard at the null orientation,
    # the excitatory cells nearly stop.
    for b in ["L4Cb_ss", "L23_pyr", "L4_pv"]:
        if b not in d["layers"]:
            continue
        c = np.roll(np.array(d["layers"][b]["aligned_curve"]), n // 2)
        osi = d["layers"][b]["osi_cell_mean"]
        ax.plot(x, c / max(c.max(), 1e-9), "o-", ms=3, lw=1.4,
                color=LAYER_COLOURS[b], label=f"{NICE[b]}  (OSI {osi:.2f})")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("orientation relative to preferred (deg)")
    ax.set_ylabel("response / peak response")
    ax.set_title("Excitation is sharply tuned,\nfast-spiking inhibition is not")
    ax.legend(frameon=False, fontsize=6.5)

    ax = axes[2]
    bases = [b for b in ["L4Cb_ss", "L23_pyr", "L5_pyr", "L6_pyr", "L4_pv",
                         "L23_pv"] if b in d["layers"]]
    vals = [d["layers"][b]["osi_cell_mean"] for b in bases]
    ax.bar(range(len(bases)), vals,
           color=[LAYER_COLOURS[b] for b in bases])
    ax.set_xticks(range(len(bases)))
    ax.set_xticklabels([NICE[b].replace(" ", "\n") for b in bases],
                       fontsize=5.5)
    ax.set_ylabel("orientation selectivity index")
    ax.set_title("Single-cell selectivity")

    fig.tight_layout()
    fig.savefig(FIGS / "01_orientation_tuning.png", bbox_inches="tight")
    plt.close(fig)


def fig_dissection():
    d = _load("circuit_dissection.json")
    if d is None:
        return
    modes = ["intact", "feedforward_only", "no_inhibition", "no_pv",
             "no_horizontal"]
    labels = ["intact", "feedforward\nonly", "no\ninhibition", "no basket\noutput",
              "no horizontal\nconnections"]
    layers = ["L4Cb_ss", "L23_pyr", "L5_pyr"]

    # "no horizontal" lies on top of "intact" -- that overlap is the result,
    # so it is drawn dashed and wider to stay visible underneath.
    style = {
        "intact": dict(color="#1b6ca8", lw=1.6, ls="-", zorder=5),
        "feedforward_only": dict(color="#e08214", lw=1.2, ls="-"),
        "no_inhibition": dict(color="#2a9d3f", lw=1.2, ls="-"),
        "no_pv": dict(color="#c44536", lw=1.2, ls="-"),
        "no_horizontal": dict(color="#7b539b", lw=2.6, ls=":", zorder=2),
    }
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.9))
    for ax, b in zip(axes, layers):
        n = len(d["intact"][b]["aligned_curve"])
        x = (np.arange(n) - n // 2) * (180.0 / n)
        for m, lab in zip(modes, labels):
            if b not in d.get(m, {}):
                continue
            c = np.roll(np.array(d[m][b]["aligned_curve"]), n // 2)
            osi = d[m][b]["osi"]
            ax.plot(x, c, "o", ms=2.5, label=None, **{
                k: v for k, v in style[m].items() if k == "color"})
            ax.plot(x, c, label=f"{lab.replace(chr(10), ' ')} (OSI {osi:.2f})",
                    **style[m])
        ax.set_title(NICE[b])
        ax.set_xlabel("orientation rel. preferred (deg)")
        ax.set_ylabel("spikes/s")
    axes[0].legend(frameon=False, fontsize=6)
    fig.suptitle("Removing each circuit element in turn", y=1.04)
    fig.tight_layout()
    fig.savefig(FIGS / "02_circuit_dissection.png", bbox_inches="tight")
    plt.close(fig)


def fig_simple_complex():
    d = _load("simple_vs_complex.json")
    if d is None:
        return
    layers = [b for b in ["L4Cb_ss", "L4B_pyr", "L23_pyr", "L5_pyr", "L6_pyr"]
              if b in d["layers"] and d["layers"][b]["n"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.9))

    ax = axes[0]
    data = [np.array(d["layers"][b]["ratio"]) for b in layers]
    parts = ax.violinplot(data, showmedians=True, widths=0.8)
    for pc, b in zip(parts["bodies"], layers):
        pc.set_facecolor(LAYER_COLOURS[b])
        pc.set_alpha(0.65)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xticks(range(1, len(layers) + 1))
    ax.set_xticklabels([NICE[b].replace(" ", "\n") for b in layers],
                       fontsize=5.5)
    ax.set_ylabel("F1 / F0")
    ax.set_title("Response modulation\n(>1 simple, <1 complex)")

    ax = axes[1]
    frac = [100 * d["layers"][b]["frac_simple"] for b in layers]
    ax.bar(range(len(layers)), frac,
           color=[LAYER_COLOURS[b] for b in layers])
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([NICE[b].replace(" ", "\n") for b in layers],
                       fontsize=5.5)
    ax.set_ylabel("% classified simple")
    ax.set_title("Simple cells are concentrated\nin the input layer")
    fig.tight_layout()
    fig.savefig(FIGS / "03_simple_vs_complex.png", bbox_inches="tight")
    plt.close(fig)


def fig_ocular_dominance():
    d = _load("ocular_dominance.json")
    if d is None:
        return
    layers = [b for b in ["L4Cb_ss", "L4Ca_ss", "L23_pyr", "L5_pyr", "L6_pyr"]
              if b in d["layers"]]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.9))

    ax = axes[0]
    for b in ("L4Cb_ss", "L23_pyr"):
        if b not in d["layers"]:
            continue
        h = np.array(d["layers"][b]["histogram"], dtype=float)
        h = h / max(h.sum(), 1)
        ax.bar(np.arange(1, 8) + (0.18 if b == "L23_pyr" else -0.18), h,
               width=0.36, color=LAYER_COLOURS[b], label=NICE[b])
    ax.set_xlabel("ocular-dominance group (1 = contra, 7 = ipsi)")
    ax.set_ylabel("fraction of cells")
    ax.set_title("Hubel-Wiesel ocular-dominance histogram")
    ax.legend(frameon=False, fontsize=6.5)

    ax = axes[1]
    vals = [d["layers"][b]["mean_abs_odr"] for b in layers]
    ax.bar(range(len(layers)), vals,
           color=[LAYER_COLOURS[b] for b in layers])
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([NICE[b].replace(" ", "\n") for b in layers],
                       fontsize=5.5)
    ax.set_ylabel("mean |ocular dominance|")
    ax.set_title("Monocularity is highest\nin the input layer")

    ax = axes[2]
    od = np.array(d["column_od_index"])
    for b in ("L4Cb_ss", "L23_pyr"):
        if b not in d["layers"]:
            continue
        ax.plot(od, d["layers"][b]["odr_by_column"], "o-", ms=4,
                color=LAYER_COLOURS[b], label=NICE[b])
    ax.set_xlabel("column's position in the ocular-dominance stripe")
    ax.set_ylabel("measured eye preference")
    ax.set_title("Measured preference follows\nthe anatomical stripe")
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIGS / "04_ocular_dominance.png", bbox_inches="tight")
    plt.close(fig)


def fig_streams():
    d = _load("spatial_frequency.json")
    if d is None:
        return
    sfs = np.array(d["spatial_freqs"])
    cs = np.array(d["contrasts"])
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.9))

    ax = axes[0]
    for b in ["L4Ca_ss", "L4Cb_ss", "L4B_pyr", "L23_pyr"]:
        if b not in d["sf_tuning"]:
            continue
        v = np.array(d["sf_tuning"][b], dtype=float)
        if v.max() <= 0:
            continue
        ax.plot(sfs, v / v.max(), "o-", ms=3, lw=1.4,
                color=LAYER_COLOURS[b], label=NICE[b])
    ax.set_xscale("log")
    ax.set_xlabel("spatial frequency (cycles/deg)")
    ax.set_ylabel("normalised response")
    ax.set_title("The magnocellular sublayer prefers\ncoarser gratings")
    ax.legend(frameon=False, fontsize=6.5)

    ax = axes[1]
    for b in ["L4Ca_ss", "L4Cb_ss"]:
        if b not in d["contrast_response"]:
            continue
        v = np.array(d["contrast_response"][b]["response"], dtype=float)
        if v.max() <= 0:
            continue
        ax.plot(cs, v / v.max(), "o-", ms=3, lw=1.4,
                color=LAYER_COLOURS[b],
                label=f"{NICE[b]}  c50={d['contrast_response'][b]['fit']['c50']:.2f}")
    ax.set_xlabel("contrast")
    ax.set_ylabel("normalised response")
    ax.set_title("...and saturates at lower contrast")
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIGS / "05_magno_parvo_streams.png", bbox_inches="tight")
    plt.close(fig)


def fig_contrast_invariance():
    d = _load("contrast_response.json")
    if d is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.9))

    ax = axes[0]
    cs = np.array(d["contrasts"])
    for b, dd in d["layers"].items():
        v = np.array(dd["response"], dtype=float)
        ax.plot(cs, v, "o-", ms=3, lw=1.3,
                color=LAYER_COLOURS.get(b, "k"), label=NICE.get(b, b))
    ax.set_xlabel("contrast")
    ax.set_ylabel("spikes/s")
    ax.set_title("Contrast-response functions")
    ax.legend(frameon=False, fontsize=6)

    ax = axes[1]
    for c, dd in sorted(d["tuning"].items(), key=lambda kv: float(kv[0])):
        n = len(dd["aligned_curve"])
        x = (np.arange(n) - n // 2) * (180.0 / n)
        v = np.roll(np.array(dd["aligned_curve"]), n // 2)
        v = v / max(v.max(), 1e-9)
        ax.plot(x, v, "o-", ms=3, lw=1.3,
                label=f"contrast {float(c):.2f}  (OSI {dd['osi']:.2f}, "
                      f"HWHH {dd['hwhh']:.0f}°)")
    ax.set_xlabel("orientation rel. preferred (deg)")
    ax.set_ylabel("response / peak response")
    # The model sharpens with contrast; real V1 does not.  Say so on the plot.
    ax.set_title("Tuning is NOT contrast invariant\n"
                 "(the model's main failure)")
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(FIGS / "06_contrast.png", bbox_inches="tight")
    plt.close(fig)


def fig_architecture():
    """The column's anatomy: who is where and who talks to whom."""
    from column import ColumnConfig, CorticalColumn
    from column.column import LAYER_DEPTH, LAYER_PLAN

    col = CorticalColumn(ColumnConfig(n_columns=8))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))

    # (a) pinwheel of orientation columns
    ax = axes[0]
    sc = ax.scatter(col.pos[:, 0], col.pos[:, 1], c=col.theta_pref, s=1.5,
                    cmap="hsv", vmin=0, vmax=180)
    ax.set_aspect("equal")
    ax.set_xlabel("cortical surface (µm)")
    ax.set_ylabel("cortical surface (µm)")
    ax.set_title("Orientation columns\n(one hypercolumn, pinwheel layout)")
    plt.colorbar(sc, ax=ax, label="preferred orientation (deg)",
                 fraction=0.046)

    # (b) laminar profile
    ax = axes[1]
    for base, layer, ct, cnt in LAYER_PLAN:
        idx = col.group(base)
        colour = "#c44536" if not ct.excitatory else "#1b6ca8"
        ax.scatter(col.pos[idx, 0] + np.random.normal(0, 6, len(idx)),
                   col.depth[idx], s=0.7, c=colour, alpha=0.5)
    for layer, dpt in sorted(LAYER_DEPTH.items(), key=lambda kv: kv[1]):
        ax.axhline(dpt, color="k", lw=0.3, alpha=0.25)
        ax.text(340, dpt, layer, fontsize=5.5, va="center")
    ax.invert_yaxis()
    ax.set_ylabel("depth below pia (µm)")
    ax.set_xlabel("cortical surface (µm)")
    ax.set_title("Laminar organisation\n(blue excitatory, red inhibitory)")

    # (c) receptive fields of four layer IV cells
    ax = axes[2]
    from column.connectivity import gabor
    g = col.grid
    idx = col.pops[col.key("L4Cb_ss", 0)].idx[:4]
    panel = np.zeros((g.n_pix * 2, g.n_pix * 2))
    for i, gid in enumerate(idx):
        rf = gabor(g.x, g.y, col.rf_x[gid], col.rf_y[gid],
                   col.theta_pref[gid], col.cfg.sf_parvo, col.rf_phase[gid],
                   1 / (3 * col.cfg.sf_parvo),
                   col.cfg.rf_aspect / (3 * col.cfg.sf_parvo))
        r, c = divmod(i, 2)
        panel[r * g.n_pix:(r + 1) * g.n_pix,
              c * g.n_pix:(c + 1) * g.n_pix] = rf / np.abs(rf).max()
    ax.imshow(panel, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Four IVCβ receptive fields\n(same column: one orientation,"
                 "\nfour spatial phases)")

    fig.tight_layout()
    fig.savefig(FIGS / "00_architecture.png", bbox_inches="tight")
    plt.close(fig)


def fig_population_decoding():
    d = _load("population_decoding.json")
    if d is None:
        return
    oris = np.array(d["orientations"])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    ax = axes[0]
    ax.plot([0, 180], [0, 180], "k--", lw=0.8, label="perfect decoding")
    for b in ["L4Cb_ss", "L23_pyr", "L5_pyr"]:
        if b not in d["layers"]:
            continue
        est = np.array(d["layers"][b]["decoded"])
        # Orientation is circular modulo 180: plot the representative nearest
        # the stimulus, so a 5 degree error near zero does not appear as 175.
        est = oris + ((est - oris + 90.0) % 180.0) - 90.0
        ax.plot(oris, est, "o", ms=4, color=LAYER_COLOURS[b], label=NICE[b],
                alpha=0.8)
    ax.set_xlabel("stimulus orientation (deg)")
    ax.set_ylabel("decoded orientation (deg)")
    ax.set_title("Population vector decoding")
    ax.legend(frameon=False, fontsize=6.5)

    ax = axes[1]
    bases = [b for b in ["L4Ca_ss", "L4Cb_ss", "L23_pyr", "L6_pyr", "L5_pyr",
                         "L4_pv"] if b in d["layers"]]
    vals = [d["layers"][b]["mean_error_deg"] for b in bases]
    ax.bar(range(len(bases)), vals,
           color=[LAYER_COLOURS.get(b, "k") for b in bases])
    ax.set_xticks(range(len(bases)))
    ax.set_xticklabels([NICE[b].replace(" ", "\n") for b in bases],
                       fontsize=5.5)
    ax.set_ylabel("mean decoding error (deg)")
    ax.set_title("Accuracy of the population code")
    fig.tight_layout()
    fig.savefig(FIGS / "07_population_decoding.png", bbox_inches="tight")
    plt.close(fig)


def fig_moving_bar():
    d = _load("moving_bar.json")
    if d is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.9))

    ax = axes[0]
    r = d["conditions"]["preferred"]["raster"]
    if r:
        ax.plot(r[0], r[1], "|", ms=2.5, color="#1b6ca8", mew=0.6)
    ax.set_xlabel("time from bar onset (ms)")
    ax.set_ylabel("IVCβ cell")
    ax.set_title("Raster, bar at preferred orientation")

    for ax, cond in zip(axes[1:], ("preferred", "orthogonal")):
        c = d["conditions"][cond]
        for b in ["L4Cb_ss", "L23_pyr", "L5_pyr"]:
            if b not in c["psth"]:
                continue
            ax.plot(c["psth"][b]["t"], c["psth"][b]["rate"], lw=1.3,
                    color=LAYER_COLOURS[b], label=NICE[b])
        ax.set_xlabel("time from bar onset (ms)")
        ax.set_ylabel("spikes/s")
        ax.set_title(f"PSTH, bar {cond}")
    axes[1].legend(frameon=False, fontsize=6.5)
    ymax = max(a.get_ylim()[1] for a in axes[1:])
    for a in axes[1:]:
        a.set_ylim(0, ymax)
    fig.tight_layout()
    fig.savefig(FIGS / "08_moving_bar.png", bbox_inches="tight")
    plt.close(fig)


def main():
    fig_architecture()
    fig_orientation()
    fig_dissection()
    fig_simple_complex()
    fig_ocular_dominance()
    fig_streams()
    fig_contrast_invariance()
    fig_population_decoding()
    fig_moving_bar()
    print("figures written to", FIGS.resolve())
    for p in sorted(FIGS.glob("*.png")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
