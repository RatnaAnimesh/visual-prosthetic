"""Publication figures for the manuscript.

Numbered, multi-panel, self-contained captions in the manuscript.  These are
deliberately separate from the analysis figures in ``figures/``: those exist to
look at while working, these exist to be read once, in print, by someone who
has not seen the code.

Every figure after the first shows the forward model built from measured
parts (``prosthesis/cortex_ssn.py``).  The phenomenological pooled-divisor
model (``prosthesis/cortex_circuit.py``) appears in orange where it is the
alternative being compared against, and on its own only in figure 11, which
collects the predictions that would follow if cortex had a pooled divisor.

Output: paper/figures/fig1.png ... fig11.png

Run:  python3 paper/make_figures.py [fig2 fig5 ...]
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pulse2percept.implants import ArgusII
from pulse2percept.implants.cortex import Cortivis
from pulse2percept.models import AxonMapModel, ScoreboardModel
from pulse2percept.models.cortex import DynaphosModel
from pulse2percept.stimuli import BiphasicPulseTrain

from prosthesis.cortex_ssn import BiophysicalCortex
from prosthesis.v2_experiments import X0, Y0, pair

OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "results"

# Journal figures: single column ~85 mm, double ~175 mm. Work in inches.
ONE_COL, TWO_COL = 3.35, 6.9

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "lines.linewidth": 1.2,
    "figure.facecolor": "white", "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

C_BIO = "#1d4f91"                      # the model built from measured parts
C_OCC, C_NET = "#7b539b", "#1b7f79"    # its occupancy and network parts
C_POOL = "#e08214"                     # the pooled-divisor alternative
C_LIN, C_HORIZ = "#7a7a7a", "#1b7f79"  # linear spread, horizontal connections
C_E, C_I = "#b2182b", "#2166ac"        # excitatory / inhibitory cells


def load(name):
    return json.loads((RESULTS / name).read_text())


def panel(ax, letter, dx=-0.16, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="top", ha="left")


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.4)


def _row(rows, **kw):
    return next(r for r in rows if all(r[k] == v for k, v in kw.items()))


# --------------------------------------------------------------------------

def fig1():
    """The independence assumption, and the retinal precedent for doubting it."""
    implant = ArgusII()
    letter = {e: 1 for e in ["B5", "C5", "D5", "E5", "F5", "D6", "D7"]}
    implant.stim = letter
    sb = ScoreboardModel(xrange=(-14, 12), yrange=(-12, 12), rho=200).build()
    ax_m = AxonMapModel(xrange=(-14, 12), yrange=(-12, 12), rho=200,
                        axlambda=800).build()
    img_sb = sb.predict_percept(implant).data[..., 0]
    img_ax = ax_m.predict_percept(implant).data[..., 0]

    cort = Cortivis()
    dyn = DynaphosModel(xrange=(-3.2, -0.4), yrange=(-0.2, 2.0),
                        xystep=0.02).build()

    def peak(stim):
        cort.stim = stim
        d = dyn.predict_percept(cort).data
        return d[..., int(np.argmax(d.reshape(-1, d.shape[-1]).max(axis=0)))]

    tr = lambda a: BiphasicPulseTrain(300, a, 0.17, stim_dur=100.0)
    a, b = cort.electrode_names[0], cort.electrode_names[95]
    pa, pb = peak({a: tr(200.0)}), peak({b: tr(200.0)})
    pab = peak({a: tr(200.0), b: tr(200.0)})
    resid = pab - (pa + pb)
    rel = np.abs(resid).max() / max(np.abs(pa + pb).max(), 1e-12)

    fig, axes = plt.subplots(1, 4, figsize=(TWO_COL, 1.75))
    fig.subplots_adjust(wspace=0.08, top=0.80, bottom=0.14)
    for ax, (img, ttl, lt) in zip(axes, [
            (img_sb, "independent pixels", "a"),
            (img_ax, "with axon activation", "b"),
            (pab, "both electrodes on", "c"),
            (resid, f"minus sum of singles\n(max {100 * rel:.0e}%)", "d")]):
        div = lt == "d"
        m = float(np.abs(img).max()) or 1.0
        ax.imshow(img, cmap="RdBu_r" if div else "gray",
                  vmin=-m if div else 0, vmax=m, origin="lower",
                  aspect="equal")
        bare(ax); panel(ax, lt, dx=-0.02, dy=1.14)
        ax.set_title(ttl, pad=2, fontsize=6.5)
    fig.text(0.30, 0.03, "retinal implant (Argus II)", ha="center",
             fontsize=6.5, style="italic")
    fig.text(0.73, 0.03, "cortical implant (Dynaphos / Cortivis)",
             ha="center", fontsize=6.5, style="italic")
    fig.savefig(OUT / "fig1.png")
    plt.close(fig)
    return {"dynaphos_residual": float(rel)}


def fig2():
    """The forward model built from measured parts, and the size it predicts."""
    z = np.load(RESULTS / "v2_fields_size.npz")
    rows = load("v2_paper_size.json")["rows"]
    xs, ys = z["xs"], z["ys"]
    ex, ey = z["electrode"]
    w = 2400.0
    sx, sy = np.abs(xs - ex) <= w, np.abs(ys - ey) <= w
    crop = lambda a: a[np.ix_(sy, sx)]
    ext = ((xs[sx][0] - ex) / 1000, (xs[sx][-1] - ex) / 1000,
           (ys[sy][0] - ey) / 1000, (ys[sy][-1] - ey) / 1000)

    fig, axes = plt.subplots(1, 4, figsize=(TWO_COL, 1.75),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.3]})
    maps = [
        (np.rad2deg(crop(z["theta"])), "hsv", 0, 180, "orientation map",
         "deg"),
        (crop(z["p_e_100ua"]), "magma", 0, None,
         "E cells driven directly", "fraction"),
        (crop(z["cortex_100ua"]), "RdBu_r", None, None,
         "evoked E rate", "Hz"),
    ]
    for ax, (img, cmap, lo, hi, ttl, unit), lt in zip(axes, maps, "abc"):
        if lo is None:
            m = float(np.abs(img).max())
            lo, hi = -m, m
        im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi if hi is not None
                       else float(img.max()), origin="lower", extent=ext)
        ax.plot(0, 0, "+", color="k" if lt != "b" else "w", ms=5, mew=0.8)
        ax.set_title(ttl, pad=2, fontsize=6.5)
        ax.set_xlabel("mm from tip", fontsize=6)
        if lt == "a":
            ax.set_ylabel("mm from tip", fontsize=6)
        else:
            ax.set_yticklabels([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=5)
        cb.set_label(unit, fontsize=5.5, labelpad=1)
        panel(ax, lt, dx=-0.34 if lt == "a" else -0.14, dy=1.30)

    ax = axes[3]
    amps = [r["amp_ua"] for r in rows]
    dyn = [(r["amp_ua"], r["dynaphos_diameter_deg"]) for r in rows
           if r["dynaphos_diameter_deg"] is not None]
    ax.plot(*zip(*dyn), "o-", color="k", ms=2.6, lw=1.2,
            label="Dynaphos baseline")
    ax.plot(amps, [r["halfmax_diameter_deg"] for r in rows], "o-",
            color=C_BIO, ms=2.6, lw=1.2, label="measured parts, half max")
    vis = [(r["amp_ua"], r["visible_diameter_deg"]) for r in rows
           if r["visible_diameter_deg"] > 0]
    ax.plot(*zip(*vis), "s--", color=C_BIO, mfc="white", ms=2.6, lw=1.0,
            label="measured parts, rate doubled")
    ax.set_xscale("log")
    ax.set_xlabel("current (µA)")
    ax.set_ylabel("phosphene\ndiameter (deg)")
    ax.set_ylim(0, 0.62)  # headroom so the legend clears the data
    ax.legend(frameon=False, fontsize=4.9, loc="upper left",
              labelspacing=0.2, handlelength=1.6)
    panel(ax, "d", dx=-0.34, dy=1.30)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(OUT / "fig2.png")
    plt.close(fig)


def fig3():
    """The model built from measured parts against measurements."""
    v = load("v2_validation.json")
    fig, axes = plt.subplots(1, 4, figsize=(TWO_COL, 1.95))

    ax = axes[0]
    rows = v["recruitment"]["rows"]
    amps = np.array([r["amp_ua"] for r in rows])
    ax.loglog(amps, [r["cells_in_plane"] for r in rows], "-o", color=C_E,
              ms=2.4, lw=1.2, label="model")
    ax.plot([8.5], [5.0], "s", mfc="none", mec="k", ms=5, mew=0.9,
            label="Histed et al.\n(calibration)")
    ax.set_xlabel("current (µA)")
    ax.set_ylabel("cells recruited in one\noptical section")
    ax.legend(frameon=False, fontsize=5.2, loc="upper left")
    panel(ax, "a", dx=-0.42)

    ax = axes[1]
    inh = v["inhibition"]["rows"]
    rk = [r for r in rows if r["amp_ua"] <= 100]
    ax.semilogx([r["amp_ua"] for r in rk],
                [r["median_distance_um"] for r in rk], "-o", color=C_E,
                ms=2.4, lw=1.2, label="model, E")
    ax.semilogx([r["amp_ua"] for r in inh],
                [r["median_distance_i_um"] for r in inh], "-o", color=C_I,
                ms=2.4, lw=1.2, label="model, I")
    kum = v["recruitment"]["kumaravelu_median_um"]
    ax.plot([float(a) for a in kum], list(kum.values()), "D", mfc="none",
            mec="k", ms=3.5, mew=0.9, label="Kumaravelu et al.")
    ax.set_ylim(0, 520)
    ax.set_xlabel("current (µA)")
    ax.set_ylabel("median distance of\nrecruited somata (µm)")
    ax.legend(frameon=False, fontsize=5.2, loc="lower right")
    panel(ax, "b", dx=-0.42)

    ax = axes[2]
    vs = v["visual_plus_stim"]
    for amp, mk in ((50.0, "o"), (150.0, "s")):
        rr = [r for r in vs if r["amp_ua"] == amp]
        ax.plot([r["contrast"] for r in rr], [r["ratio_to_sum"] for r in rr],
                mk + "-", color="0.35" if amp == 50 else C_BIO, ms=2.6,
                lw=1.2, label=f"{amp:.0f} µA")
    ax.axhline(1.0, color="k", lw=0.7, ls=":")
    ax.set_ylim(0.5, 1.1)
    ax.set_xlabel("grating contrast")
    ax.set_ylabel("both / (visual + electrical)")
    ax.legend(frameon=False, fontsize=5.5, loc="lower left")
    panel(ax, "c", dx=-0.38)

    ax = axes[3]
    du = [r for r in v["dual"] if r["separation_um"] == 485.0]
    a = [r["amp_ua"] for r in du]
    ax.axvspan(6.8, 15.9, color="0.9", lw=0)
    ax.semilogx(a, [100 * r["direct_fraction"] for r in du], "-o",
                color=C_OCC, ms=2.4, lw=1.1, label="directly driven")
    ax.semilogx(a, [100 * r["network_fraction"] for r in du], "-o",
                color=C_NET, ms=2.4, lw=1.1, label="network-driven")
    ax.semilogx(a, [100 * r["net_fraction"] for r in du], "-", color="k",
                lw=1.3, label="net")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("current (µA)")
    ax.set_ylabel("both − sum of singles\n(% of sum)")
    ax.legend(frameon=False, fontsize=5.2, loc="lower left")
    panel(ax, "d", dx=-0.42)

    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / "fig3.png")
    plt.close(fig)


def fig4(sep=800.0, amp=300.0):
    """Only a nonlinearity after the electrodes combine breaks independence."""
    sw = load("phase1_circuit.json")["separation_sweep"]
    pairs = load("v2_experiments.json")["pairs"]

    fig = plt.figure(figsize=(TWO_COL, 2.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.6, 1, 1, 1], wspace=0.18,
                          left=0.10, right=0.99, top=0.86, bottom=0.19)
    ax = fig.add_subplot(gs[0, 0])
    for k, c, ls, lw, lab in [
            ("circuit off", C_LIN, "-", 2.4, "linear spread"),
            ("horizontal only", C_HORIZ, "--", 1.1,
             "+ horizontal connections"),
            ("pointwise saturation", C_POOL, ":", 1.3,
             "pointwise saturation"),
            ("full circuit", C_POOL, "-", 1.2, "pooled divisor")]:
        x = [r["separation_um"] / 1000 for r in sw[k]]
        y = [100 * r["max_dev"] for r in sw[k]]
        ax.plot(x, y, ls, color=c, lw=lw, label=lab)
    seps = sorted({r["separation_um"] for r in pairs})
    x = np.array(seps) / 1000
    ax.plot(x, [100 * _row(pairs, amp_ua=amp, separation_um=s)["max_dev"]
                for s in seps], "o-", color=C_BIO, ms=2.4, lw=1.4,
            label="measured parts")
    ax.plot(x, [100 * _row(pairs, amp_ua=amp,
                           separation_um=s)["max_dev_direct"]
                for s in seps], "--", color=C_OCC, lw=1.1,
            label="measured parts,\noccupancy only")
    ax.set_xlabel("electrode separation (mm)")
    ax.set_ylabel("departure from\nindependence (%)")
    ax.legend(frameon=False, fontsize=5.2, labelspacing=0.25)
    panel(ax, "a", dx=-0.34)

    m = BiophysicalCortex()
    f = pair(m, sep, amp)
    g = m._ensure_cortical_grid(np.array([X0, X0 + sep]), np.array([Y0, Y0]))
    lin, both = f["a"] + f["b"], f["ab"]
    res = both - lin
    xs, ys = g["xs"], g["ys"]
    cx = X0 + sep / 2
    sx = np.abs(xs - cx) <= 2000.0
    sy = np.abs(ys - Y0) <= 1400.0
    crop = lambda a: a[np.ix_(sy, sx)]
    vmax = float(crop(lin).max())
    for i, (img, ttl, lt, div) in enumerate([
            (crop(lin), "predicted if\nindependent", "b", False),
            (crop(both), "measured-parts\nmodel", "c", False),
            (crop(res), "difference", "d", True)]):
        a = fig.add_subplot(gs[0, i + 1])
        mm = float(np.abs(img).max()) or 1.0
        a.imshow(img, cmap="RdBu_r" if div else "inferno",
                 vmin=-mm if div else 0, vmax=mm if div else vmax,
                 origin="lower")
        bare(a); panel(a, lt, dx=-0.10, dy=1.34)
        a.set_title(ttl, pad=3)
    fig.savefig(OUT / "fig4.png")
    plt.close(fig)
    return {"fig4_example_separation_um": sep}


def fig5():
    """How large the interaction is in measured cortex, and its shape."""
    e = load("v2_experiments.json")
    pairs = e["pairs"]
    seps = sorted({r["separation_um"] for r in pairs})
    x = np.array(seps) / 1000
    fig, axes = plt.subplots(1, 4, figsize=(TWO_COL, 1.95))

    def curve(amp, key):
        return [100 * _row(pairs, amp_ua=amp, separation_um=s)[key]
                for s in seps]

    amps = (30.0, 100.0, 300.0, 600.0)
    cm = plt.cm.viridis(np.linspace(0.1, 0.85, len(amps)))
    ax = axes[0]
    for amp, c in zip(amps, cm):
        ax.plot(x, curve(amp, "signed_total"), "-o", color=c, ms=2.2,
                lw=1.1, label=f"{amp:.0f} µA")
    ax.plot(x, curve(300.0, "signed_direct"), "--", color=cm[2], lw=1.0,
            label="300 µA, occupancy")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("separation (mm)")
    ax.set_ylabel("signed interaction (%)")
    ax.legend(frameon=False, fontsize=5.0, loc="lower right",
              labelspacing=0.2)
    panel(ax, "a", dx=-0.36)

    ax = axes[1]
    all_amps = sorted({r["amp_ua"] for r in pairs})
    show = (400.0, 800.0, 1600.0, 2400.0)
    sc = plt.cm.cividis(np.linspace(0.05, 0.85, len(show)))
    for s, c in zip(show, sc):
        ax.semilogx(all_amps, [100 * _row(pairs, amp_ua=a, separation_um=s)
                               ["signed_total"] for a in all_amps], "-o",
                    color=c, ms=2.0, lw=1.1, label=f"{s/1000:g} mm")
    surf = load("phase7_regimes.json")["surface"]["full circuit"]
    r = next(q for q in surf if abs(q["separation_um"] - 400.0) < 1)
    ax.semilogx(r["amps_ua"], 100 * np.array(r["signed_total"]), "-",
                color=C_POOL, lw=1.0, label="pooled, 0.4 mm")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylim(-64, 78)
    ax.set_xlabel("current (µA)")
    ax.set_ylabel("signed interaction (%)")
    # the band below -25 is clear of every curve
    ax.legend(frameon=False, fontsize=4.8, loc="lower left",
              labelspacing=0.2, ncol=2, columnspacing=0.8,
              handlelength=1.2)
    panel(ax, "b", dx=-0.36)

    ax = axes[2]
    for amp, c in zip(amps[1:], cm[1:]):
        ax.plot(x, curve(amp, "max_dev"), "-o", color=c, ms=2.2, lw=1.1,
                label=f"{amp:.0f} µA")
    sw = load("phase1_circuit.json")["separation_sweep"]["full circuit"]
    ax.plot([q["separation_um"] / 1000 for q in sw],
            [100 * q["max_dev"] for q in sw], "-", color=C_POOL, lw=1.0,
            label="pooled divisor,\n300 µA")
    ax.set_xlabel("separation (mm)")
    ax.set_ylabel("peak departure (%)")
    ax.legend(frameon=False, fontsize=5.0, labelspacing=0.2)
    panel(ax, "c", dx=-0.34)

    ax = axes[3]
    hs = (0.4, 0.8, 1.6, 2.4)
    for name, h in e["sensitivity"].items():
        y = [100 * h[f"peak_300ua_{s:.1f}mm"] for s in hs]
        if name == "baseline":
            continue
        ax.plot(hs, y, "-", color="0.65", lw=0.6)
    h = e["sensitivity"]["baseline"]
    ax.plot(hs, [100 * h[f"peak_300ua_{s:.1f}mm"] for s in hs], "-o",
            color=C_BIO, ms=2.2, lw=1.4, label="calibrated")
    ax.plot([], [], "-", color="0.65", lw=0.6, label="18 variants")
    ax.set_xlabel("separation (mm)")
    ax.set_ylabel("peak departure (%)")
    ax.legend(frameon=False, fontsize=5.2, title="300 µA",
              title_fontsize=5.2, alignment="left")
    panel(ax, "d", dx=-0.34)

    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT / "fig5.png")
    plt.close(fig)


def fig6():
    """The human multi-electrode data."""
    b = load("v2_paper_bosking.json")
    rows = b["rows"]
    seps = sorted({r["separation_um"] for r in rows})
    amps = sorted({r["amp_ua"] for r in rows})
    fig = plt.figure(figsize=(TWO_COL, 3.7))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.55], hspace=0.55,
                          wspace=0.42, left=0.08, right=0.99, top=0.96,
                          bottom=0.10)
    axes = [fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1])]

    ax = axes[0]
    cm = plt.cm.viridis(np.linspace(0.05, 0.9, len(amps)))
    for a, c in zip(amps, cm):
        ax.plot(np.array(seps) / 1000,
                [100 * _row(rows, amp_ua=a, separation_um=s)["max_dev"]
                 for s in seps], "-o", color=c, ms=2.0, lw=1.1,
                label=(f"{a/1000:g} mA" if a >= 1000 else f"{a:.0f} µA"))
    p5 = load("phase5_scale.json")
    ax.plot(np.array(p5["separations_um"]) / 1000,
            100 * np.array(p5["by_amplitude"]["4000.0"]["deviations"]),
            "--", color=C_POOL, lw=1.0, label="pooled divisor,\n4 mA")
    for mm, lab in [(4.11, "merged"), (13.47, "one unreported"),
                    (26.36, "two phosphenes")]:
        ax.axvline(mm, color="k", ls=":", lw=0.7)
        ax.text(mm * 1.06, 58, lab, fontsize=5.0, rotation=90, va="top")
    ax.set_xscale("log")
    ax.set_xticks([0.4, 1, 2, 4, 10, 26])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0, 60)
    ax.set_xlabel("electrode separation (mm)")
    ax.set_ylabel("peak departure (%)")
    ax.legend(frameon=False, fontsize=5.2, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), labelspacing=0.3)
    panel(ax, "a", dx=-0.09)

    rp = load("v2_paper_repulsion.json")
    # the same colour for the same current as in (a)
    shades = [cm[amps.index(a)] for a in (100.0, 300.0, 600.0)]
    pool_ls = {100.0: ":", 300.0: "--", 600.0: "-."}
    for ax, (bio_key, pool_key, val, ylab, lt) in zip(axes[1:], [
            ("measured_parts", "pooled_divisor", "repulsion",
             "change in distance between\nthe two centroids (%)", "b"),
            ("peaks_measured_parts", "peaks_pooled_divisor", "peak_shift",
             "change in distance between\nthe two maxima (%)", "c")]):
        for a, c in zip((100.0, 300.0, 600.0), shades):
            rr = [r for r in rp[bio_key]
                  if r["amp_ua"] == a and r[val] is not None]
            ax.plot([r["separation_um"] / 1000 for r in rr],
                    [100 * r[val] for r in rr], "-o", color=c, ms=2.0,
                    lw=1.1, label=f"measured parts, {a:.0f} µA")
        for a in (100.0, 300.0, 600.0):
            rr = [r for r in rp[pool_key]
                  if r["amp_ua"] == a and r[val] is not None]
            ax.plot([r["separation_um"] / 1000 for r in rr],
                    [100 * r[val] for r in rr], pool_ls[a], color=C_POOL,
                    lw=1.0, label=f"pooled divisor, {a:.0f} µA")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("electrode separation (mm)")
        ax.set_ylabel(ylab)
        panel(ax, lt, dx=-0.30, dy=1.16)
    axes[2].legend(frameon=False, fontsize=5.2, loc="upper left",
                   bbox_to_anchor=(1.05, 1.0), labelspacing=0.3)
    fig.savefig(OUT / "fig6.png")
    plt.close(fig)


def fig7():
    """What the person would see."""
    z = np.load(RESULTS / "v2_fields_percepts.npz")
    p = load("v2_paper_percepts.json")
    test, rec = z["test"][:4], z["recon"][:4]
    # Both rows come from prosthesis/v2_paper.py, each decoded through its
    # own model's orientation map with the same fixed linear operator.
    pool = (z["pooled_alone"], z["pooled_together"])
    bio = (z["line_alone"], z["line_together"])

    fig = plt.figure(figsize=(TWO_COL, 2.15))
    gs = fig.add_gridspec(2, 8, hspace=0.10, wspace=0.10,
                          width_ratios=[1, 1, 1, 1, 0.35, 1, 1, 1],
                          left=0.07, right=0.99, top=0.84, bottom=0.03)
    for j in range(4):
        a = fig.add_subplot(gs[0, j])
        a.imshow(test[j], cmap="gray", origin="lower"); bare(a)
        if j == 0:
            a.set_ylabel("shown", fontsize=6); panel(a, "a", dx=-0.34, dy=1.22)
        if j == 1:
            a.set_title(f"held-out $R^2$ = {p['decoder_r2']:.2f}",
                        fontsize=6.5, pad=3, loc="left")
        b2 = fig.add_subplot(gs[1, j])
        b2.imshow(rec[j], cmap="gray", origin="lower"); bare(b2)
        if j == 0:
            b2.set_ylabel("decoded", fontsize=6)

    for row, (pr, lab) in enumerate([(bio, "measured parts"),
                                     (pool, "pooled divisor")]):
        lin, both = pr
        mm = float(np.abs(lin).max()) or 1.0
        for col, (img, ttl) in enumerate([(lin, "sum of singles"),
                                          (both, "driven together"),
                                          (both - lin, "difference")]):
            a = fig.add_subplot(gs[row, 5 + col])
            a.imshow(img, cmap="RdBu_r", vmin=-mm, vmax=mm, origin="lower")
            bare(a)
            if row == 0:
                a.set_title(ttl, pad=3, fontsize=6)
                if col == 0:
                    panel(a, "b", dx=-0.40, dy=1.22)
            if col == 0:
                a.set_ylabel(lab, fontsize=6)
    fig.savefig(OUT / "fig7.png")
    plt.close(fig)
    return {"fig7_decoder_r2": float(p["decoder_r2"])}


def fig8():
    """What an encoder programmed under independence delivers."""
    q = load("v2_paper_encoder.json")
    old = load("phase4_encoder.json")
    z = np.load(RESULTS / "v2_fields_encoder.npz")
    names = list(q["results"])
    fig = plt.figure(figsize=(TWO_COL, 2.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.5, 0.95],
                          wspace=0.45, hspace=0.12, left=0.08, right=0.99,
                          top=0.84, bottom=0.17)
    # panel (c) gets its own tight grid, so its images sit together
    gc = gs[0, 2].subgridspec(2, 2, wspace=0.08, hspace=0.12)

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(names))
    ax.bar(x - 0.2, [q["results"][n]["full_on_full"] for n in names], 0.4,
           color=C_BIO, label="programmed for measured parts")
    ax.bar(x + 0.2, [q["results"][n]["indep_on_full"] for n in names], 0.4,
           color=C_LIN, label="programmed as independent")
    lo = min(min(q["results"][n]["indep_on_full"] for n in names), 0.9)
    ax.set_ylim(np.floor(lo * 50) / 50, 1.005)
    ax.set_xticks(x); ax.set_xticklabels([f"P{i + 1}" for i in x])
    ax.set_ylabel("pattern fidelity ($r^2$)")
    ax.set_xlabel("target pattern")
    ax.legend(frameon=False, ncol=1, loc="lower left",
              bbox_to_anchor=(0.0, 1.0), borderaxespad=0.0, fontsize=5.3)
    panel(ax, "a", dx=-0.24, dy=1.30)

    ax = fig.add_subplot(gs[0, 1])
    mq = q["mean"]
    vals = [1.0, mq["contrast_ratio_mismatched"], mq["contrast_ratio_mse"],
            old["mean_contrast_ratio"], old["mean_contrast_ratio_mse"]]
    xb = [0.0, 1.25, 1.95, 3.15, 3.85]
    ax.bar(xb, vals, 0.62, color=["k", C_BIO, C_BIO, C_POOL, C_POOL])
    for k in (2, 4):
        ax.patches[k].set_hatch("////"); ax.patches[k].set_edgecolor("white")
    ax.set_xticks([0.0, 1.6, 3.5])
    ax.set_xticklabels(["target", "measured\nparts", "pooled\ndivisor"],
                       fontsize=5.6)
    ax.set_xlim(-0.55, 4.4)
    ax.legend(handles=[
        matplotlib.patches.Patch(facecolor="0.45",
                                 label="programmed as independent"),
        matplotlib.patches.Patch(facecolor="0.45", hatch="////",
                                 edgecolor="white",
                                 label="re-optimised for brightness")],
        frameon=False, fontsize=5.3, loc="lower left",
        bbox_to_anchor=(0.0, 1.0), borderaxespad=0.0)
    ax.set_ylabel("percept contrast\n(RMS, relative to target)")
    ax.axhline(1.0, color="k", lw=0.5, ls=":")
    panel(ax, "b", dx=-0.36, dy=1.30)

    for j, (key, ttl) in enumerate([("target", "target"),
                                    ("mismatch", "delivered")]):
        for i in range(2):
            a = fig.add_subplot(gc[i, j])
            img = z[f"t{i}_{key}"]
            mm = float(np.abs(z[f"t{i}_target"]).max()) or 1.0
            a.imshow(img, cmap="RdBu_r", vmin=-mm, vmax=mm, origin="lower")
            bare(a)
            if i == 0:
                a.set_title(ttl, pad=3, fontsize=6)
                if j == 0:
                    panel(a, "c", dx=-0.30, dy=1.30)
    fig.savefig(OUT / "fig8.png")
    plt.close(fig)


def fig9():
    """Arrays: no ceiling, and a modest charge bill."""
    e = load("v2_experiments.json")
    d = load("phase8_channels.json")
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL * 0.72, 2.05))

    ax = axes[0]
    # blues for the measured parts; orange stays the pooled divisor's
    pc = plt.cm.Blues(np.linspace(0.5, 0.95, 3))
    for pitch, c in zip((400.0, 800.0, 1600.0), pc):
        for amp, ls in ((100.0, "-"), (300.0, "--")):
            rr = [r for r in e["channels"] if r["pitch_um"] == pitch
                  and r["amp_ua"] == amp]
            ax.plot([r["n"] for r in rr], [r["efficiency"] for r in rr], ls,
                    marker="o", color=c, ms=2.0, lw=1.1,
                    label=f"{pitch/1000:g} mm" if amp == 100 else None)
    counts = np.array(d["counts"], float)
    rel = np.array(d["ceilings"]["circuit"]["400.0"]["relative_total"])
    ax.plot(counts, rel / counts, ":", color=C_POOL, lw=1.2,
            label="pooled divisor,\n0.4 mm")
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("electrodes driven")
    ax.set_ylabel("delivered / sum of singles")
    ax.legend(frameon=False, fontsize=5.0, loc="lower left",
              title="pitch", title_fontsize=5.0, labelspacing=0.2)
    panel(ax, "a", dx=-0.34)

    ax = axes[1]
    for pitch, mk in ((400.0, "o"), (800.0, "s")):
        rr = [r for r in e["charge"] if r["pitch_um"] == pitch]
        ax.plot([1] + [r["n"] for r in rr], [0.0] + [r["delta_k"] for r in rr],
                "-" + mk, color=C_BIO, ms=2.6, lw=1.2,
                label=f"measured parts, {pitch/1000:g} mm")
        rows = [r for r in d["shannon"]["by_pitch"][str(pitch)]
                if r.get("delta_k") is not None]
        ax.plot([r["n"] for r in rows], [r["delta_k"] for r in rows],
                "--" + mk, color=C_POOL, ms=2.2, lw=1.0, mfc="white",
                label=f"pooled divisor, {pitch/1000:g} mm")
    ax.set_xscale("log")
    ax.set_xlabel("electrodes driven")
    ax.set_ylabel("increase in Shannon $k$")
    ax.legend(frameon=False, fontsize=5.0, loc="upper left")
    panel(ax, "b", dx=-0.30)
    fig.tight_layout(w_pad=1.4)
    fig.savefig(OUT / "fig9.png")
    plt.close(fig)


def fig10():
    """Sequential stimulation, and what the shape experiment costs."""
    e = load("v2_experiments.json")
    old = load("phase9_sequence.json")
    pw = load("v2_paper_power.json")
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL * 0.72, 2.05))

    ax = axes[0]
    for amp, ls in (("300.0", "--"), ("600.0", "-")):
        rows = old["sequence_bias"][amp]
        ax.plot([r["delay_ms"] for r in rows],
                [r["ratio_second_over_first"] for r in rows], ls,
                color=C_POOL, lw=1.0,
                label=f"pooled divisor, {float(amp):.0f} µA")
    for label, mk, mfc in (("fast inhibition only", "o", C_BIO),
                           ("30% slow inhibition", "s", "white")):
        r = next(q for q in e["sequence"][label]
                 if q["separation_um"] == 1600.0 and q["amp_ua"] == 300.0)
        ax.plot([q["delay_ms"] for q in r["rows"]],
                [q["second_over_solo"] for q in r["rows"]], "-",
                marker=mk, mfc=mfc, color=C_BIO, ms=2.4, lw=1.1,
                label="measured parts" + ("" if mk == "o" else ", slow inh."))
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set_ylim(0.5, 1.1)
    ax.set_xlabel("inter-electrode delay (ms)")
    ax.set_ylabel("second phosphene / alone")
    ax.legend(frameon=False, fontsize=5.0, loc="lower right")
    panel(ax, "a", dx=-0.32)

    ax = axes[1]
    rows = pw["rows"]
    ax.plot([r["sem_pp"] for r in rows], [r["power_if_measured"] for r in rows],
            "-o", color=C_BIO, ms=2.6, lw=1.2, label="truth = measured parts")
    ax.plot([r["sem_pp"] for r in rows], [r["power_if_pooled"] for r in rows],
            "-s", color=C_POOL, ms=2.6, lw=1.2, label="truth = pooled divisor")
    ax.axhline(0.8, color="k", lw=0.6, ls=":")
    ax.set_xlabel("standard error per separation (pp)")
    ax.set_ylabel("probability of\nidentifying the mechanism")
    ax.set_ylim(0.45, 1.02)
    ax.legend(frameon=False, fontsize=5.3)
    panel(ax, "b", dx=-0.40)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(OUT / "fig10.png")
    plt.close(fig)


def fig11():
    """What a pooled divisor would add, if cortex had one."""
    s = load("phase1_sensitivity.json")
    surf = load("phase7_regimes.json")["surface"]
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL, 2.05))

    ax = axes[0]
    cm = plt.cm.Oranges(np.linspace(0.35, 0.95, len(s["norm_gain_sweep"])))
    for r, c in zip(s["norm_gain_sweep"], cm):
        ax.plot(np.array(r["separations"]) / 1000,
                100 * np.array(r["deviations"]), color=c, lw=1.0,
                label=f"{r['norm_gain']:g}")
    ax.set_xlabel("separation (mm)")
    ax.set_ylabel("departure from\nindependence (%)")
    ax.legend(frameon=False, ncol=2, title="pool gain", title_fontsize=5.5,
              loc="upper right", handlelength=1.0, columnspacing=1.0,
              labelspacing=0.3, fontsize=5.2)
    panel(ax, "a", dx=-0.34)

    ax = axes[1]
    rows = surf["full circuit"]
    cmv = plt.cm.viridis(np.linspace(0.05, 0.85, len(rows)))
    for r, c in zip(rows, cmv):
        ax.semilogx(r["amps_ua"], 100 * np.array(r["signed_total"]),
                    color=c, lw=1.1, label=f"{r['separation_um']/1000:g}")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("current (µA)")
    ax.set_ylabel("signed interaction (%)")
    ax.legend(frameon=False, title="sep. (mm)", title_fontsize=5.5,
              fontsize=5.2, ncol=2, handlelength=1.0, columnspacing=0.9,
              labelspacing=0.25)
    panel(ax, "b", dx=-0.34)

    ax = axes[2]
    for label, ls, mk in [("pointwise saturation", ":", "o"),
                          ("full circuit", "-", "s")]:
        xs = [r["separation_um"] / 1000 for r in surf[label]
              if r["crossover_ua"]]
        ys = [r["crossover_ua"] for r in surf[label] if r["crossover_ua"]]
        ax.plot(xs, ys, mk + ls, color=C_POOL, ms=3, lw=1.2,
                label="pooled divisor" if label == "full circuit"
                else "pointwise saturation")
    ax.set_yscale("log")
    ax.set_xlabel("separation (mm)")
    ax.set_ylabel("crossover current (µA)")
    ax.legend(frameon=False, fontsize=5.5)
    panel(ax, "c", dx=-0.36)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(OUT / "fig11.png")
    plt.close(fig)


FIGS = (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11)


def main() -> None:
    print("building manuscript figures")
    info = {}
    only = set(sys.argv[1:])
    for fn in FIGS:
        if only and fn.__name__ not in only:
            continue
        r = fn()
        if r:
            info.update(r)
        print(f"  {fn.__name__}  {fn.__doc__.splitlines()[0]}")
    if not only:
        for stale in OUT.glob("fig*.png"):
            if stale.stem not in {f.__name__ for f in FIGS}:
                stale.unlink()
        (OUT / "_figure_info.json").write_text(json.dumps(info, indent=2))
    print(f"\n{len(list(OUT.glob('fig*.png')))} figures in {OUT}")
    for k, v in info.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
