"""The model against raw cells from awake cortex, and against human thresholds.

Every recruitment number in the main model comes from published summaries.
This script reads the one dataset with raw cells shared, Dadarlat et al.'s
6,698 neurons in awake mouse V1 (``prosthesis/dadarlat.py``), and one human
curve, Fernandez et al.'s thresholds for groups of contiguous electrodes.

    cells      reproduces the published fractions from the raw traces, fits the
               stimulation site of each mouse from its responses, and sets the
               model's recruitment against every cell at every current: the
               paper's calibration as it stands, its two densities refitted,
               and the same with a local term added.  Also the inhibitory to
               excitatory ratio near the tip at each current, with a
               hierarchical bootstrap.
    human      per-electrode threshold for 1, 2, 4, 9 and 16 contiguous
               electrodes at 400 um (Fernandez et al. 2021, their
               Supplementary Figure 1), for sparse and awake recruitment and
               readouts from a local peak to the total
    pairs      the pair experiment of the main paper with recruitment fitted to
               the awake cells, as an upper bound: it counts every cell that
               responds as driven directly
    maps       the headline pair numbers over twelve orientation maps, for
               sparse and awake recruitment
    spaced     the human groups on every second or third electrode, where
               occupancy and probability summation part
    dense      the paper's discriminating predictions rerun with awake
               recruitment on the paper's own grid and map: the current
               sweep and where the sign turns, array output, the charge to
               hold it, and sequential phosphenes
    kim        Kim et al.'s paired net modulation (2025) with awake recruitment,
               scaling the inhibitory densities, and the I/E ratio each scale
               gives near the tip at 20 uA, read as the data are read

Run:  python3 prosthesis/v2_awake.py [part ...]      (default: all)
      The data must be fetched first:
      python3 -c "from prosthesis import dadarlat as D; D.fetch(); D.extract()"
"""

from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import replace

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from prosthesis import dadarlat as D
from prosthesis.cortex_ssn import BiophysicalCortex, RecruitmentParams
from prosthesis.v2_experiments import interaction, pair
from prosthesis import v2_experiments as E
from prosthesis.v2_validate import dual_electrode

RESULTS = pathlib.Path("results")
RESULTS.mkdir(exist_ok=True)


def _save(part, data):
    (RESULTS / f"v2_awake_{part}.json").write_text(json.dumps(data, indent=1))
    print(f"wrote {RESULTS / f'v2_awake_{part}.json'}", flush=True)


def _load(part):
    return json.loads((RESULTS / f"v2_awake_{part}.json").read_text())


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

H0_EDGES = np.arange(0.0, 701.0, 25.0)


def unit_profiles(n_tips: int = 16) -> dict:
    """The model's expected sites per cell at unit density, against distance.

    Averaged over tips scattered across the orientation map, since the data
    pool five electrodes on five maps.
    """
    rp = replace(RecruitmentParams(), site_density_e=1.0, site_density_i=1.0)
    m = BiophysicalCortex(step_um=26.75, margin_um=1600.0, recruit=rp)
    g = m._ensure_cortical_grid([0.0], [0.0])
    X, Y = np.meshgrid(g["xs"], g["ys"])
    tips = np.random.default_rng(1).uniform(-500, 500, size=(n_tips, 2))
    nb = len(H0_EDGES) - 1
    out = {"r": (0.5 * (H0_EDGES[1:] + H0_EDGES[:-1])).tolist(), "e": {}, "i": {}}
    for a in D.AMPS:
        acc = {"e": np.zeros(nb), "i": np.zeros(nb)}
        for tx, ty in tips:
            he, hi = m.hits([a], [tx], [ty], g)
            idx = np.digitize(np.hypot(X - tx, Y - ty).ravel(), H0_EDGES) - 1
            ok = (idx >= 0) & (idx < nb)
            cnt = np.maximum(np.bincount(idx[ok], minlength=nb), 1)
            for k, h in (("e", he), ("i", hi)):
                acc[k] += np.bincount(idx[ok], h.ravel()[ok], minlength=nb) / cnt
        for k in acc:
            out[k][str(a)] = (acc[k] / n_tips).tolist()
    return out


def cells_part():
    cs = D.cells()
    out = {"published_fractions": {str(a): v for a, v in D.published_fractions(cs).items()},
           "n_cells": int(sum(len(c["d"]) for c in cs)),
           "n_inhibitory": int(sum(c["red"].sum() for c in cs))}

    # Where is the electrode?  Fitted per mouse, against both readings of epos.
    sites = []
    for c in cs:
        z = D.load(c["m"])
        t = D.cell_table(z)
        col, row = D.fit_site(z, t)
        u, e = z["umpp"], z["epos"]
        sx, sy = u[0] / z["mag"], u[1] / z["mag"]
        code = (e[1] * u[1] / u[0], e[0] * u[0] / u[1])
        sites.append({"mouse": c["m"], "fitted_col_row": [col, row],
                      "from_code_um": float(np.hypot((col - code[0]) * sx, (row - code[1]) * sy)),
                      "as_col_row_um": float(np.hypot((col - e[0]) * sx, (row - e[1]) * sy))})
    out["sites"] = sites

    out["profiles"] = {lab: {f"{k}{a}": v for (k, a), v in D.profile(cs, lab).items()}
                       for lab in ("sig", "early")}

    ratio = {}
    for lab in ("sig", "early"):
        for a in (10, 15, 20, 25, 30, 40, 50):
            lo, hi = D.bootstrap(cs, lambda s: D.site_ratio(s, a, lab))
            ratio[f"{lab}{a}"] = {"ratio": D.site_ratio(cs, a, lab), "ci": [lo, hi],
                                 "ratio_without_GADC": D.site_ratio(
                                     [c for c in cs if c["m"] != "GADC"], a, lab)}
    out["site_ratio"] = ratio

    H = unit_profiles()
    out["unit_profiles"] = H
    r = np.array(H["r"])

    def h0(kind, a, d):
        return np.interp(d, r, np.array(H[kind][str(a)]))

    fits = {}
    for lab in ("sig", "early"):
        fixed = D.fit_recruitment(cs, h0, lab, local=False,
                                  x0=[np.log(9.0), np.log(6.0), -5.0])
        fits[f"{lab}_paper"] = _fixed_densities(cs, h0, lab)
        fits[f"{lab}_densities"] = fixed
        fits[f"{lab}_local"] = D.fit_recruitment(cs, h0, lab, local=True)
    out["fits"] = fits
    out["field_medians"] = field_medians(cs, H, fits)
    _save("cells", out)


def field_medians(cs, H, fits, amps=(20, 50)):
    """Median distance of responding cells over the imaged field, data and model.

    The model's median is weighted by its response probability at each
    imaged cell's distance, so both are read over the same cells.
    """
    r = np.array(H["r"])

    def wmed(x, w):
        o = np.argsort(x)
        c = np.cumsum(w[o])
        return float(x[o][np.searchsorted(c, c[-1] / 2)])

    out = {}
    for a in amps:
        for kind, want in (("e", False), ("i", True)):
            d = np.concatenate([c["d"][c["red"] == want] for c in cs])
            y = np.concatenate([c["sig"][a][c["red"] == want] for c in cs])
            out[f"data_{kind}{a}"] = float(np.median(d[y]))
            for name, f in (("calibration", {"s_e": 9.0, "s_i": 6.0}), ("refit", fits["sig_local"])):
                h = f[f"s_{kind}"] * np.interp(d, r, np.array(H[kind][str(a)]))
                if "L_um" in f:
                    h = h + f[f"s_loc_{kind}"] * (a / 20.0) ** 1.5 * np.exp(-d ** 2 / (2 * f["L_um"] ** 2))
                out[f"{name}_{kind}{a}"] = wmed(d, 1 - np.exp(-h))
    return out


def _fixed_densities(cs, h0, lab, se=9.0, si=6.0, edges=(0, 100, 200, 300, 500)):
    """The paper's calibration as it stands, nothing fitted.

    The floor is the fraction that responds at 3 uA, where the model drives
    almost nobody; fitting it instead would let it absorb the whole misfit.
    """
    f = float(np.mean(np.concatenate([c[lab][3] for c in cs])))
    nll, pred = 0.0, {}
    for kind, want, s in (("e", False, se), ("i", True, si)):
        for a in D.AMPS:
            d = np.concatenate([c["d"][c["red"] == want] for c in cs])
            y = np.concatenate([c[lab][a][c["red"] == want] for c in cs]).astype(float)
            p = np.clip(f + (1 - f) * (1 - np.exp(-s * h0(kind, a, d))), 1e-9, 1 - 1e-9)
            nll -= float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
            pred[f"{kind}{a}"] = [float(p[(d >= lo) & (d < hi)].mean())
                                  for lo, hi in zip(edges[:-1], edges[1:])]
    return {"s_e": se, "s_i": si, "floor": f, "nll": nll, "aic": 2 * nll,
            "binned_prediction": pred}


def awake_params(label: str = "sig") -> RecruitmentParams:
    """Recruitment fitted to every awake cell that responds (an upper bound)."""
    f = _load("cells")["fits"][f"{label}_local"]
    return replace(RecruitmentParams(), site_density_e=f["s_e"], site_density_i=f["s_i"],
                   local_e=f["s_loc_e"], local_i=f["s_loc_i"], local_um=f["L_um"])


def variants() -> dict:
    return {"sparse": RecruitmentParams(),
            "awake_all": awake_params("sig"),
            "awake_fast": awake_params("early")}


# ---------------------------------------------------------------------------
# Human thresholds for groups of contiguous electrodes
# ---------------------------------------------------------------------------

PITCH = 400.0
GROUPS = {1: [(0, 0)], 2: [(0, 0), (1, 0)],
          4: [(i, j) for i in range(2) for j in range(2)],
          9: [(i, j) for i in range(3) for j in range(3)],
          16: [(i, j) for i in range(4) for j in range(4)]}
# Fernandez et al. 2021, Supplementary Figure 1, digitised: mean and SD (uA)
FERNANDEZ = {1: (73.5, 7.7), 2: (70.8, 5.5), 4: (47.7, 13.9),
             9: (42.3, 9.3), 16: (28.1, 5.3)}
SIGMAS = (0.0, 50.0, 100.0, 200.0, 400.0, 800.0, np.inf)
CURRENTS = np.geomspace(3.0, 1500.0, 70)
ORIGINS = ((0.0, 0.0), (130.0, 270.0), (-310.0, 90.0), (220.0, -180.0))


def _readout(pe, sigma, step):
    if np.isinf(sigma):
        return pe.sum()
    return (gaussian_filter(pe, sigma / step) if sigma > 0 else pe).max()


def _curves(m, g, additive, pitch=PITCH):
    """Readout against current for each group and placement."""
    out = {n: np.zeros((len(ORIGINS), len(CURRENTS), len(SIGMAS))) for n in GROUPS}
    for oi, (ox, oy) in enumerate(ORIGINS):
        for n, cells_ in GROUPS.items():
            xs = [ox + i * pitch for i, j in cells_]
            ys = [oy + j * pitch for i, j in cells_]
            for ci, a in enumerate(CURRENTS):
                if additive:
                    pe = sum(1 - np.exp(-m.hits([a], [x], [y], g)[0]) for x, y in zip(xs, ys))
                else:
                    pe = 1 - np.exp(-m.hits([a] * n, xs, ys, g)[0])
                out[n][oi, ci] = [_readout(pe, s, m.step) for s in SIGMAS]
    return out


def group_thresholds(curves, single):
    """Per-electrode threshold of each group over the single threshold."""
    res = {}
    for si, s in enumerate(SIGMAS):
        r = {}
        for n in GROUPS:
            v = []
            for oi in range(len(ORIGINS)):
                target = np.interp(np.log(single), np.log(CURRENTS), curves[1][oi, :, si])
                v.append(np.exp(np.interp(target, curves[n][oi, :, si], np.log(CURRENTS))) / single)
            r[n] = float(np.mean(v))
        res[s] = r
    return res


def chi2(r):
    """Against the data's ratios, with the bars on both thresholds as error.

    The legend does not say what the bars are.  As errors of the mean this is
    the usual test; as spreads across groups it is lenient, and a small value
    is weak evidence.
    """
    m1, s1 = FERNANDEZ[1]
    tot = 0.0
    for n in (2, 4, 9, 16):
        mn, sn = FERNANDEZ[n]
        obs = mn / m1
        tot += ((r[n] - obs) / (obs * np.hypot(sn / mn, s1 / m1))) ** 2
    return float(tot)


def human_part():
    out = {"data": {str(n): v for n, v in FERNANDEZ.items()}, "rows": []}
    for name, rp in variants().items():
        # Recruitment alone: the network moves cells it does not drive by
        # under half a hertz at these currents, far below detection.
        m = BiophysicalCortex(step_um=26.75, margin_um=1400.0, recruit=rp, network=False)
        g = m._ensure_cortical_grid([-400.0, 1600.0], [-400.0, 1600.0])
        for additive in (False, True):
            C = _curves(m, g, additive)
            for single in (35.0, FERNANDEZ[1][0], 150.0):
                for s, r in group_thresholds(C, single).items():
                    out["rows"].append({"variant": name, "additive": additive,
                                        "single_ua": single,
                                        "sigma_um": None if np.isinf(s) else s,
                                        "ratios": {str(n): v for n, v in r.items()},
                                        "chi2": chi2(r)})
        print(f"human: {name} done", flush=True)
    _save("human", out)


SPACED = (800.0, 1200.0)
WEIBULL = 3.3   # the slope at which probability summation fits the 400 um curve best


def spaced_part(pitches=SPACED):
    """The same groups on every second or third electrode of the array.

    Occupancy needs overlap, which fades with spacing; probability summation
    over independent electrodes does not depend on it.  The two predictions
    part here, on an array already implanted.
    """
    out = {"weibull": WEIBULL, "probability_summation": {
        str(n): n ** (-1 / WEIBULL) for n in GROUPS}, "rows": []}
    for name in ("sparse", "awake_all"):
        m = BiophysicalCortex(step_um=26.75, margin_um=1400.0, recruit=variants()[name],
                              network=False)
        for pitch in pitches:
            g = m._ensure_cortical_grid([-400.0, 3 * pitch + 400.0], [-400.0, 3 * pitch + 400.0])
            for s, r in group_thresholds(_curves(m, g, False, pitch), FERNANDEZ[1][0]).items():
                out["rows"].append({"variant": name, "pitch_um": pitch,
                                    "sigma_um": None if np.isinf(s) else s,
                                    "ratios": {str(n): v for n, v in r.items()}})
            print(f"spaced: {name} {pitch:.0f} um done", flush=True)
    _save("spaced", out)


# ---------------------------------------------------------------------------
# Pairs with awake recruitment
# ---------------------------------------------------------------------------

PAIR_AMPS = (10.0, 50.0, 100.0, 300.0)
PAIR_SEPS = (400.0, 800.0, 1200.0, 1600.0, 2400.0)


def pairs_part(step: float = 53.5):
    rows = []
    for name, rp in variants().items():
        m = BiophysicalCortex(step_um=step, recruit=rp)
        for amp in PAIR_AMPS:
            for sep in PAIR_SEPS:
                rows.append({"variant": name, "amp_ua": amp, "separation_um": sep,
                             **interaction(pair(m, sep, amp))})
                print(f"pairs: {name} {amp:.0f} uA {sep:.0f} um "
                      f"{100 * rows[-1]['signed_total']:+.2f}%", flush=True)
    _save("pairs", {"step_um": step, "rows": rows})


def maps_part(step: float = 53.5, seeds=range(12)):
    """The headline pair numbers across orientation maps.

    The map is drawn from a seed and the grid size, so one number from one
    map is one sample: the footprint of each electrode depends on the
    orientation domain its tip lands in.
    """
    rows = []
    for name in ("sparse", "awake_all"):
        rp = variants()[name]
        for seed in seeds:
            m = BiophysicalCortex(step_um=step, recruit=rp, seed=seed)
            for amp in (100.0, 300.0):
                rows.append({"variant": name, "seed": seed, "amp_ua": amp,
                             **interaction(pair(m, 400.0, amp))})
        print(f"maps: {name} done", flush=True)
    summary = {}
    for name in ("sparse", "awake_all"):
        for amp in (100.0, 300.0):
            for key in ("signed_total", "max_dev"):
                v = np.array([r[key] for r in rows
                              if r["variant"] == name and r["amp_ua"] == amp])
                summary[f"{name}_{amp:.0f}_{key}"] = {
                    "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                    "min": float(v.min()), "max": float(v.max())}
    _save("maps", {"step_um": step, "rows": rows, "summary": summary})


# ---------------------------------------------------------------------------
# Kim et al.'s paired electrodes with awake recruitment
# ---------------------------------------------------------------------------

KIM_MULTS = (1.0, 1.25, 1.5, 2.0)


def model_site_ratio(rp: RecruitmentParams, a: int = 20, rmax: float = 300.0) -> float:
    """The model's I/E ratio near the tip, read as ``D.site_ratio`` reads the data.

    Each cell of the dataset gets the model's probability at its own distance,
    the probabilities are pooled within ``rmax`` and -ln(1 - f) is taken.
    """
    H, cs = _load("cells")["unit_profiles"], D.cells()
    r = np.array(H["r"])
    sites = {}
    for kind, want, s, l in (("e", False, rp.site_density_e, rp.local_e),
                             ("i", True, rp.site_density_i, rp.local_i)):
        d = np.concatenate([c["d"][(c["red"] == want) & (c["d"] < rmax)] for c in cs])
        h = s * np.interp(d, r, np.array(H[kind][str(a)])) \
            + l * (a / 20.0) ** 1.5 * np.exp(-d ** 2 / (2 * rp.local_um ** 2))
        sites[kind] = -np.log(np.mean(np.exp(-h)))
    return float(sites["i"] / sites["e"])


def kim_part(step: float = 53.5, mults=KIM_MULTS):
    base, out = awake_params("sig"), []
    for k in mults:
        rp = replace(base, site_density_i=base.site_density_i * k, local_i=base.local_i * k)
        m = BiophysicalCortex(step_um=step, pulse_hz=200.0, recruit=rp)
        out.append({"mult": k, "ie_ratio_20ua": model_site_ratio(rp),
                    "rows": dual_electrode((370.0, 485.0, 558.0, 850.0), (6.8, 12.7, 15.9), m)})
        print(f"kim: x{k} ratio {out[-1]['ie_ratio_20ua']:.2f} net at 485 um " + " ".join(
            f"{100 * x['net_fraction']:+.2f}" for x in out[-1]["rows"]
            if x["separation_um"] == 485.0), flush=True)
    _save("kim", {"step_um": step, "sparse_ie_ratio_20ua": model_site_ratio(RecruitmentParams()),
                  "by_mult": out})


def dense_part():
    """Same calls and grid as v2_experiments.py, with recruitment fitted to awake cells.

    Both fits bracket the directly driven fraction: awake_all counts every
    responder as driven, awake_fast only the short-latency ones.
    """
    out = {}
    for name in ("awake_all", "awake_fast"):
        m = BiophysicalCortex(recruit=variants()[name])
        rows = E.pair_sweep(m)
        out[name] = {"pairs": rows, "crossover_ua": E.crossovers(rows),
                     "interaction_range_um": E.interaction_range(rows),
                     "signature": E.signatures(rows),
                     "channels": E.channels(m), "charge": E.charge_to_hold(m)}
        print(f"dense: {name} sweep, channels and charge done", flush=True)
        _save("dense", out)
    m = BiophysicalCortex(recruit=variants()["awake_all"])
    out["awake_all"]["sequence"] = [E.sequence(m, sep, amp) for sep in (800.0, 1600.0)
                                    for amp in (100.0, 300.0)]
    _save("dense", out)


PARTS = {"cells": cells_part, "human": human_part, "pairs": pairs_part,
         "maps": maps_part, "kim": kim_part, "spaced": spaced_part,
         "dense": dense_part}

if __name__ == "__main__":
    parts = sys.argv[1:] or list(PARTS)
    unknown = set(parts) - set(PARTS)
    if unknown:
        raise SystemExit(f"unknown part(s) {unknown}; choose from {', '.join(PARTS)}")
    if not D.available():
        # shortcut: the raw data are 1.8 GB, so run_all keeps the committed results instead
        print("skipped: Dadarlat et al.'s data not fetched (see the docstring); "
              "keeping the results already in results/")
        raise SystemExit(0)
    for p in parts:
        PARTS[p]()
