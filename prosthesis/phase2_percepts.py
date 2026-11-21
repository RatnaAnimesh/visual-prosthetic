"""Phase 2: what would the person actually see?

Everything up to here has been cortical activation.  The step to a percept was
always an assumption -- that what you see is the activation painted back
through the retinotopic map.  This replaces it with a decoder fitted on the
normal-vision pathway and then applied to stimulation.

  step 1  validate the decoder by round-tripping held-out images.  If it
          cannot reconstruct an image the cortex actually saw, it has no
          business interpreting one the cortex was made to hallucinate.
  step 2  decode single-electrode stimulation.
  step 3  the letter test -- the cortical counterpart of the retinal figure
          in Phase 0.  Drive a pattern of electrodes meant to draw a shape and
          ask what the shape decodes to, under the independent baseline and
          under the circuit.

Run:  python3 prosthesis/phase2_percepts.py
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

from pulse2percept.implants.cortex import Cortivis

from prosthesis.cortex_circuit import CircuitSpatial
from prosthesis.phase1_circuit import calibrated
from prosthesis.percept_decoder import (CorticalCode, InversePercept,
                                        pink_noise, reconstruction_r2)

FIGS = pathlib.Path("figures")
RESULTS = pathlib.Path("results")
FIGS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

CTX_X, CTX_Y = (16200, 23800), (-8800, -1200)
TEST_AMP = 300.0
N_TRAIN = 3000

# Calibrated in Phase 1 so both configurations reproduce the same published
# single-electrode size-vs-current curve.
def configs():
    return {"independent (baseline-like)": calibrated("circuit off"),
            "circuit": calibrated("full circuit")}


def build_decoder():
    code = CorticalCode(CTX_X, CTX_Y)
    dec = InversePercept(code, lam=0.02)
    rng = np.random.default_rng(1234)
    test = pink_noise(24, code.n_px, rng)
    r2 = reconstruction_r2(test, dec.round_trip(test))
    return code, dec, {"r2": r2, "n_sites": code.n_sites,
                       "n_px": code.n_px, "test": test}


def activation_at_sites(model, code, amps, xe, ye):
    """Run the circuit model and read it out on the decoder's sheet."""
    g = model._ensure_cortical_grid(xe, ye)
    field = model._activation_field(np.asarray(amps, float), xe, ye, g)
    return code.sample_field(field, g["xs"], g["ys"])


def electrode_subset(implant, names):
    xe = np.array([implant[n].x for n in names], float)
    ye = np.array([implant[n].y for n in names], float)
    return xe, ye


def pick_line(implant, n=5, spacing_um=1400.0):
    """Five electrodes at growing distances from the first.

    The k-th pick is the electrode closest to k * spacing_um from the
    reference, in any direction, so the group is not a straight line.  On
    the Cortivis layout with the defaults it zig-zags across the array, 1.4,
    2.8, 4.3 and 4.6 mm from the reference and 0.9 to 1.8 mm from each
    electrode's nearest neighbour in the group.
    """
    names = list(implant.electrode_names)
    ref = names[0]
    x0, y0 = implant[ref].x, implant[ref].y
    chosen = [ref]
    while len(chosen) < n:
        target = spacing_um * len(chosen)
        best, bd = None, 1e18
        for nm in names:
            if nm in chosen:
                continue
            d = abs(np.hypot(implant[nm].x - x0, implant[nm].y - y0) - target)
            if d < bd:
                best, bd = nm, d
        if best is None:
            break
        chosen.append(best)
    return chosen


def main() -> None:
    implant = Cortivis()

    print("[1/3] fitting and validating the percept decoder")
    code, dec, val = build_decoder()
    print(f"      {val['n_sites']} cortical sites -> "
          f"{val['n_px']}x{val['n_px']} image")
    print(f"      held-out reconstruction R^2 = {val['r2']:.3f}")

    recon = dec.round_trip(val["test"])
    fig, axes = plt.subplots(2, 6, figsize=(11, 4))
    for j in range(6):
        axes[0, j].imshow(val["test"][j], cmap="gray", origin="lower")
        axes[1, j].imshow(recon[j], cmap="gray", origin="lower")
        for i in range(2):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
    axes[0, 0].set_ylabel("shown", fontsize=9)
    axes[1, 0].set_ylabel("decoded", fontsize=9)
    fig.suptitle(f"The inverse recovers images from the normal-vision code "
                 f"(held-out R² = {val['r2']:.2f})", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "p2_decoder_validation.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    # -- single electrode -------------------------------------------------
    print("\n[2/3] decoding single-electrode stimulation")
    single = {}
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.4))
    for row, (label, cfg) in enumerate(configs().items()):
        m = CircuitSpatial(xrange=(-3.2, -0.4), yrange=(-0.2, 2.0),
                           xystep=0.05, **cfg)
        m.build()
        xe, ye = electrode_subset(implant, [implant.electrode_names[0]])
        a = activation_at_sites(m, code, [TEST_AMP], xe, ye)
        img = dec.decode(a)[0]
        single[label] = {"peak": float(img.max()),
                         "energy": float(np.abs(img).sum())}
        axes[row, 0].scatter(code.ctx_x / 1000, code.ctx_y / 1000, c=a,
                             s=2, cmap="inferno", marker="s")
        axes[row, 0].set_aspect("equal")
        axes[row, 0].set_title("cortical activity" if row == 0 else "",
                               fontsize=9)
        axes[row, 0].set_xlabel("cortex x (mm)", fontsize=7)
        axes[row, 0].tick_params(labelsize=6)
        axes[row, 0].set_ylabel(f"{label}\ncortex y (mm)", fontsize=7)
        mx = float(np.abs(img).max()) or 1.0
        axes[row, 1].imshow(img, cmap="RdBu_r", vmin=-mx, vmax=mx,
                            origin="lower")
        axes[row, 1].set_title("decoded percept" if row == 0 else "",
                               fontsize=9)
        axes[row, 1].set_xticks([]); axes[row, 1].set_yticks([])
    fig.suptitle("One electrode, decoded through the normal-vision code",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "p2_single_electrode.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    for k, v in single.items():
        print(f"      {k:28s} peak {v['peak']:+.3f}  energy {v['energy']:.2f}")

    # -- the letter test --------------------------------------------------
    print("\n[3/3] multi-electrode pattern (the letter test)")
    names = pick_line(implant, n=5, spacing_um=1400.0)
    xe, ye = electrode_subset(implant, names)
    seps = [float(np.hypot(x - xe[0], y - ye[0])) for x, y in zip(xe, ye)]
    print("      electrodes at cortical separations (um): " +
          " ".join(f"{s:.0f}" for s in seps))

    pattern = {}
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.4))
    for row, (label, cfg) in enumerate(configs().items()):
        m = CircuitSpatial(xrange=(-3.2, -0.4), yrange=(-0.2, 2.0),
                           xystep=0.05, **cfg)
        m.build()
        # each electrode alone, summed -- what an independent model predicts
        alone = np.zeros(code.n_sites)
        for i in range(len(names)):
            amps = np.zeros(len(names)); amps[i] = TEST_AMP
            alone += activation_at_sites(m, code, amps, xe, ye)
        together = activation_at_sites(m, code,
                                       np.full(len(names), TEST_AMP), xe, ye)

        img_alone = dec.decode(alone)[0]
        img_together = dec.decode(together)[0]
        diff = img_together - img_alone
        rel = float(np.abs(diff).max() / max(np.abs(img_alone).max(), 1e-12))
        pattern[label] = {
            "percept_change": rel,
            "activation_change": float(
                np.abs(together - alone).max() / max(np.abs(alone).max(), 1e-12)),
        }
        for col, (img, t) in enumerate([
                (img_alone, "sum of singles"),
                (img_together, "all on together"),
                (diff, f"difference ({100 * rel:.0f}%)")]):
            mx = float(np.abs(img).max()) or 1.0
            ax = axes[row, col]
            ax.imshow(img, cmap="RdBu_r", vmin=-mx, vmax=mx, origin="lower")
            if row == 0:
                ax.set_title(t, fontsize=9)
            if col == 0:
                ax.set_ylabel(label, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{len(names)} electrodes driven together, decoded as a "
                 f"percept", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "p2_pattern.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    for k, v in pattern.items():
        print(f"      {k:28s} activation changes {100 * v['activation_change']:5.1f}%"
              f"  ->  percept changes {100 * v['percept_change']:5.1f}%")

    (RESULTS / "phase2_percepts.json").write_text(json.dumps({
        "decoder": {k: v for k, v in val.items() if k != "test"},
        "single_electrode": single, "pattern": pattern,
        "electrode_separations_um": seps}, indent=2))
    print(f"\nwrote {RESULTS / 'phase2_percepts.json'}")


if __name__ == "__main__":
    main()
