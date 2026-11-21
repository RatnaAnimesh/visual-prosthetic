"""Phase 0: reproduce the existing models before building anything.

Two things are being established here, and neither is our own work.

**The retinal cautionary tale.** The naive model of a retinal implant treats
each electrode as an independent pixel -- a round blob of light at the
electrode's retinotopic position (``ScoreboardModel``).  Beyeler and colleagues
showed that is wrong: an electrode also activates the *axon bundles* passing
over it, so the percept smears along the nerve fibre trajectory and comes out
as an elongated comet (``AxonMapModel``).  The pixel assumption failed, and it
failed because of neuroanatomy the geometric model did not contain.

**The cortical baseline.** The state of the art for cortical prostheses is
Dynaphos (van der Grinten et al., eLife 2023), now shipped inside
pulse2percept as ``cortex.DynaphosModel``.  Its own docstring states the
assumption we intend to test: "Percepts from each electrode are Gaussian
blobs, with the size dictated by a magnification factor M determined by the
electrode's position in the visual cortex."  Electrodes are independent, and
V1 contributes a retinotopic map and nothing else.

Run:  python3 prosthesis/phase0_baselines.py
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pulse2percept as p2p
from pulse2percept.implants import ArgusII
from pulse2percept.implants.cortex import Cortivis
from pulse2percept.models import AxonMapModel, ScoreboardModel
from pulse2percept.models.cortex import DynaphosModel
from pulse2percept.stimuli import BiphasicPulseTrain

FIGS = pathlib.Path("figures")
FIGS.mkdir(exist_ok=True)

XR, YR = (-14, 12), (-12, 12)          # retinal figures: full implant field
XYSTEP = 0.25
# The Cortivis array maps to about 1.7-2.3 deg eccentricity in the upper-left
# quadrant, so the cortical figures need their own, much tighter window --
# at the full field of view its phosphenes are sub-pixel specks.
CORTEX_XR, CORTEX_YR, CORTEX_STEP = (-3.2, -0.4), (-0.2, 2.0), 0.02


def retinal_pixel_vs_axon() -> dict:
    """Same electrodes, two models: independent pixels vs. axon activation."""
    implant = ArgusII()

    scoreboard = ScoreboardModel(xrange=XR, yrange=YR, rho=200).build()
    axonmap = AxonMapModel(xrange=XR, yrange=YR, rho=200, axlambda=800).build()

    conditions = {
        "single electrode (C7)": {"C7": 1},
        "two electrodes (C7 + E10)": {"C7": 1, "E10": 1},
        "letter-like pattern": {e: 1 for e in
                                ["B5", "C5", "D5", "E5", "F5", "D6", "D7"]},
    }

    fig, axes = plt.subplots(2, len(conditions), figsize=(4.0 * len(conditions), 7))
    out = {}
    for j, (label, stim) in enumerate(conditions.items()):
        implant.stim = stim
        for i, (mname, model) in enumerate((("scoreboard\n(independent pixels)",
                                             scoreboard),
                                            ("axon map\n(axon bundles included)",
                                             axonmap))):
            percept = model.predict_percept(implant)
            img = percept.data[..., 0]
            extent = (percept.xdva[0], percept.xdva[-1],
                      percept.ydva[0], percept.ydva[-1])
            ax = axes[i, j]
            ax.imshow(img, cmap="gray", extent=extent, origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(label, fontsize=9)
            if j == 0:
                ax.set_ylabel(mname, fontsize=8)
            out[(mname.split("\n")[0], label)] = img

    fig.suptitle("Retinal prosthesis: the pixel assumption fails\n"
                 "(same electrodes, same positions — only the model differs)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "p0_retinal_pixel_vs_axon.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)
    return out


def cortical_baseline() -> dict:
    """The Dynaphos cortical model, and its independence assumption.

    Superposition test: is the percept from two electrodes together equal to
    the sum of each alone?  For a model that assumes independence the answer
    is yes by construction, and measuring it here establishes the baseline
    that a circuit model would have to depart from to be worth building.
    """
    implant = Cortivis()
    model = DynaphosModel(xrange=CORTEX_XR, yrange=CORTEX_YR,
                          xystep=CORTEX_STEP).build()

    names = [e for e in implant.electrode_names]
    # Two electrodes ~0.8 deg apart in the visual field, far enough that the
    # phosphenes are visibly separate.
    a, b = names[0], names[95]
    amp = 200.0  # uA, above the model's rheobase

    def train(amp_ua):
        # Dynaphos models temporal dynamics, so it needs a real pulse train:
        # 300 Hz biphasic, 170 us phase, a 100 ms burst.
        return BiphasicPulseTrain(300, amp_ua, 0.17, stim_dur=100.0)

    def percept(stim):
        implant.stim = stim
        # Take the frame at peak response rather than the last frame, which
        # may have decayed back toward baseline.
        d = model.predict_percept(implant).data
        return d[..., int(np.argmax(d.reshape(-1, d.shape[-1]).max(axis=0)))]

    p_a = percept({a: train(amp)})
    p_b = percept({b: train(amp)})
    p_ab = percept({a: train(amp), b: train(amp)})
    linear = p_a + p_b

    resid = p_ab - linear
    denom = max(float(np.abs(linear).max()), 1e-12)
    nonlin = float(np.abs(resid).max() / denom)

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, (img, t) in zip(axes, [
        (p_a, f"electrode {a}"),
        (p_b, f"electrode {b}"),
        (p_ab, "both together"),
        (resid, f"both − (sum of each)\nmax deviation {100 * nonlin:.3f}%"),
    ]):
        m = float(np.abs(img).max()) or 1.0
        ax.imshow(img, cmap="gray" if "−" not in t else "RdBu_r",
                  vmin=-m if "−" in t else 0, vmax=m, origin="lower",
                  extent=(CORTEX_XR[0], CORTEX_XR[1],
                          CORTEX_YR[0], CORTEX_YR[1]))
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("visual field x (deg)", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle("Cortical baseline (Dynaphos): electrodes are assumed "
                 "independent", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "p0_cortical_superposition.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    return {"nonlinearity": nonlin, "electrodes": (a, b)}


def phosphene_size_vs_current() -> dict:
    """Phosphene size against stimulus amplitude, the Phase 1 calibration gate.

    Any circuit model we build has to reproduce this before it has standing to
    criticise a geometric one.  Recording the baseline's own curve here gives
    us the target.
    """
    implant = Cortivis()
    model = DynaphosModel(xrange=CORTEX_XR, yrange=CORTEX_YR,
                          xystep=CORTEX_STEP).build()
    elec = implant.electrode_names[0]

    amps = np.array([50, 100, 150, 200, 300, 400, 600])
    sizes, brightness = [], []
    for a in amps:
        implant.stim = {elec: BiphasicPulseTrain(300, float(a), 0.17,
                                                 stim_dur=100.0)}
        d = model.predict_percept(implant).data
        img = d[..., int(np.argmax(d.reshape(-1, d.shape[-1]).max(axis=0)))]
        peak = float(img.max())
        brightness.append(peak)
        # Area above half-maximum, converted to an equivalent disc diameter.
        if peak > 0:
            area_px = int((img >= peak / 2).sum())
            area_deg2 = area_px * (CORTEX_STEP ** 2)
            sizes.append(2.0 * np.sqrt(area_deg2 / np.pi))
        else:
            sizes.append(0.0)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
    axes[0].plot(amps, sizes, "o-", color="#1b6ca8")
    axes[0].set_xlabel("stimulus amplitude (µA)")
    axes[0].set_ylabel("phosphene diameter (deg)")
    axes[0].set_title("Phosphene size vs. current")
    axes[1].plot(amps, brightness, "o-", color="#c44536")
    axes[1].set_xlabel("stimulus amplitude (µA)")
    axes[1].set_ylabel("peak brightness (a.u.)")
    axes[1].set_title("Brightness vs. current")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Dynaphos baseline — the curves a circuit model must match",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "p0_phosphene_size_current.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    return {"amps": amps.tolist(), "sizes": sizes, "brightness": brightness}


def main() -> None:
    print(f"pulse2percept {p2p.__version__}\n")

    print("[1/3] retinal: independent pixels vs. axon activation")
    retinal_pixel_vs_axon()

    print("[2/3] cortical: Dynaphos superposition test")
    cort = cortical_baseline()
    print(f"      electrodes {cort['electrodes']}: deviation from linear "
          f"superposition = {100 * cort['nonlinearity']:.2f}%")

    print("[3/3] cortical: phosphene size and brightness vs. current")
    sz = phosphene_size_vs_current()
    print("      amplitude (uA): " + " ".join(f"{a:6.0f}" for a in sz["amps"]))
    print("      diameter (deg): " + " ".join(f"{s:6.2f}" for s in sz["sizes"]))

    print(f"\nfigures written to {FIGS.resolve()}")


if __name__ == "__main__":
    main()
