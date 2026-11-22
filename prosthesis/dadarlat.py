"""Raw awake-mouse V1 microstimulation data, read cell by cell.

Dadarlat, Sun & Stryker (2024, Neuron 112:821) imaged 6,698 layer 2/3 neurons
(GCaMP6s, inhibitory cells labelled red in Gad2-Cre x Ai14 mice) in five awake
mice while one electrode delivered 25 pulses at 250 Hz, ten times at each of
3--50 uA.  The authors shared Suite2p output and MATLAB analysis code on
Mendeley Data (doi:10.17632/tzwps2dykh.2, CC BY 4.0).

This module downloads the data (the running-wheel video, 3.6 GB, is skipped),
reduces each mouse to a small npz, and reclassifies every cell exactly as
``compute_sig_respspub.m`` does.  ``published_fractions`` reproduces the
fractions in their paper (49.0% of excitatory and 54.0% of inhibitory cells
modulated at 40 uA, 11.1% of inhibitory cells at 10 uA, 2.7% of excitatory
cells at 5 uA), which is the check that the reading is right.

Distances follow ``InitAnalysispub.m``.  The stimulation site that best
explains each mouse's responses, fitted from the data alone, lies 64 um on
average from the site that convention gives, against 100 um for the other
reading of the labelled coordinates (``fit_site``).
"""

from __future__ import annotations

import json
import pathlib
import subprocess

import numpy as np
from scipy.optimize import minimize
from scipy.stats import mannwhitneyu

DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "dadarlat2024"
API = "https://data.mendeley.com/public-api/datasets/tzwps2dykh"
# Mouse -> (scanbox info file, stimulus block file), in the row order of epos.mat
MICE = {"GAD1": ("mgad_000_002", "Block-198"), "GADA": ("mGADA_000_002", "Block-200"),
        "GADB": ("mGADb_000_004", "Block-206"), "GADD": ("mGADd_000_002", "Block-202"),
        "GADC": ("mGADC_000_002", "Block-208")}
AMPS = (3, 5, 10, 15, 20, 25, 30, 40, 50)
NS = 20                    # samples per trial window (InitAnalysispub.m: nsamp)
S1, S2 = slice(1, 4), slice(4, 20)   # stage 1 = samples 2:4, stage 2 = 5:20 (1-based)
P_SIG = 0.02
DROP_AFTER = (30, 40, 50)  # trials following these amplitudes are excluded


# -- download and reduce ------------------------------------------------------

def _get(url: str) -> bytes:
    # The API refuses Python's default client string; curl is accepted.
    return subprocess.run(["curl", "-sfL", "--retry", "3", url], check=True,
                          capture_output=True).stdout


def fetch(dest: pathlib.Path = DATA / "raw") -> None:
    """Download every file except the running-wheel video (about 1.9 GB)."""
    dest.mkdir(parents=True, exist_ok=True)
    listing = [("", f) for f in json.loads(_get(f"{API}/files?folder_id=root&version=2"))]
    for folder in json.loads(_get(f"{API}/folders/2")):
        listing += [(folder["name"], f) for f in
                    json.loads(_get(f"{API}/files?folder_id={folder['id']}&version=2"))]
    for sub, f in listing:
        if "_ball" in f["filename"]:
            continue
        out = dest / sub / f["filename"]
        if out.exists() and out.stat().st_size == f["size"]:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sfL", "--retry", "3", "-o", str(out),
                        f["content_details"]["download_url"]], check=True)
        if out.stat().st_size != f["size"]:
            raise IOError(f"{out}: expected {f['size']} bytes")


def extract(raw: pathlib.Path = DATA / "raw", out: pathlib.Path = DATA) -> None:
    """One npz per mouse: neuropil-corrected traces, positions, labels, stimuli."""
    import scipy.io as sio
    ep = sio.loadmat(raw / "epos.mat", squeeze_me=True)
    for row, (m, (sbx, blk)) in enumerate(MICE.items()):
        info = sio.loadmat(raw / m / f"{sbx}.mat", squeeze_me=True,
                           struct_as_record=False)["info"]
        frm = np.asarray(info.frame).astype(int)
        onsets = np.round((frm[np.r_[np.diff(frm) > 100, False]] + 1) / 3).astype(int)
        A = np.asarray(sio.loadmat(raw / m / f"{blk}.mat", squeeze_me=True,
                                   struct_as_record=False)["params"].Ampl, float)
        amps = A[np.where(np.diff(A) > 1)[0] + 1][:-1]
        d = sio.loadmat(raw / m / f"F_stim_{m}_allplanes.mat", squeeze_me=True,
                        struct_as_record=False,
                        variable_names=["Fcell", "FcellNeu", "stat"])
        st = d["stat"]
        keep = np.array([s.iscell for s in st], bool)
        np.savez_compressed(
            out / f"{m}.npz",
            F=(d["Fcell"] - 0.8 * d["FcellNeu"])[keep].astype(np.float32),
            red=np.array([s.redcell for s in st])[keep].astype(int),
            plane=np.array([s.iplane for s in st])[keep].astype(int),
            med=np.array([s.med for s in st])[keep].astype(float),
            onsets=onsets, amps=amps, epos=ep["epos"][row].astype(float),
            mag=float(ep["mag"][row]), umpp=ep["micronppixel"].astype(float))


def load(m: str, root: pathlib.Path = DATA) -> dict:
    return dict(np.load(root / f"{m}.npz"))


def available(root: pathlib.Path = DATA) -> bool:
    return all((root / f"{m}.npz").exists() for m in MICE)


# -- per-cell classification ---------------------------------------------------

def cell_table(z: dict) -> dict:
    """Significance and response class per amplitude, as compute_sig_respspub.m.

    Rank-sum across trials of the window means against the pre-stimulus
    baseline, p <= 0.02, in either stage.  Class 1 is early excitation (with
    or without late excitation), the short-latency response.
    """
    F, on, amps = z["F"], z["onsets"], z["amps"]
    n = min(len(on), len(amps))
    on, amps = on[:n], amps[:n]
    resp = np.stack([F[:, o - 2:o + NS - 2] for o in on])     # MATLAB o-1..o+18
    base = np.stack([F[:, o - 17:o - 2] for o in on])          # MATLAB o-16..o-2
    resp = resp - resp[:, :, :1]
    base = base - base[:, :, :1]
    keep = ~np.isin(np.r_[np.nan, amps[:-1]], DROP_AFTER)
    out = {}
    for a in np.unique(amps):
        sel = (amps == a) & keep
        r, b = resp[sel], base[sel]
        bm, r1, r2 = b.mean(2), r[:, :, S1].mean(2), r[:, :, S2].mean(2)
        # MATLAB's ranksum is exact for these sample sizes
        meth = "exact" if len(bm) < 10 else "asymptotic"
        e = mannwhitneyu(bm, r1, axis=0, method=meth).pvalue <= P_SIG
        late = mannwhitneyu(bm, r2, axis=0, method=meth).pvalue <= P_SIG
        m1, m2 = r[:, :, S1].mean((0, 2)), r[:, :, S2].mean((0, 2))
        cls = np.zeros(F.shape[0], int)
        cls[e & (m1 > 0)] = 1
        cls[~e & late & (m2 > 0)] = 2
        cls[e & (m1 < 0)] = 3
        cls[(e & (m1 > 0) & late & (m2 < 0)) | (~e & late & (m2 < 0))] = 4
        out[int(a)] = {"sig": e | late, "early": cls == 1}
    return out


def dist(z: dict) -> np.ndarray:
    """3-D distance to the tip (um), with InitAnalysispub.m's conventions."""
    u, mag = z["umpp"], z["mag"]
    y, x = z["med"][:, 0], z["med"][:, 1]
    a = y * u[1] / mag - z["epos"][0] * u[0] / mag
    b = x * u[0] / mag - z["epos"][1] * u[1] / mag
    return np.sqrt(a ** 2 + b ** 2 + ((z["plane"] - z["epos"][2]) * 20.0) ** 2)


def cells(root: pathlib.Path = DATA) -> list[dict]:
    """Every mouse: distance, label and classification at every amplitude."""
    out = []
    for m in MICE:
        z = load(m, root)
        t = cell_table(z)
        # Four cells in GADC carry labels other than 0 and 1; the authors'
        # code counts neither as excitatory nor inhibitory, leaving 6162 + 532.
        k = np.isin(z["red"], (0, 1))
        out.append({"m": m, "d": dist(z)[k], "red": z["red"][k] == 1,
                    "sig": {a: t[a]["sig"][k] for a in AMPS},
                    "early": {a: t[a]["early"][k] for a in AMPS}})
    return out


def published_fractions(cs: list[dict]) -> dict:
    """Mean over mice of the fraction modulated, excitatory and inhibitory."""
    out = {}
    for a in AMPS:
        fe = [c["sig"][a][~c["red"]].mean() for c in cs]
        fi = [c["sig"][a][c["red"]].mean() for c in cs]
        out[a] = {"e": float(np.mean(fe)), "e_sem": float(np.std(fe, ddof=1) / np.sqrt(len(fe))),
                  "i": float(np.mean(fi)), "i_sem": float(np.std(fi, ddof=1) / np.sqrt(len(fi)))}
    return out


def profile(cs: list[dict], label: str, edges=(0, 100, 200, 300, 500)) -> dict:
    """Fraction responding by distance bin, per amplitude and cell type."""
    out = {}
    for kind, want in (("e", False), ("i", True)):
        for a in AMPS:
            row = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                k = n = 0
                for c in cs:
                    w = (c["red"] == want) & (c["d"] >= lo) & (c["d"] < hi)
                    k += int(c[label][a][w].sum())
                    n += int(w.sum())
                row.append((k, n))
            out[(kind, a)] = row
    return out


# -- the inhibitory to excitatory ratio ---------------------------------------

def site_ratio(cs: list[dict], a: int, label: str, rmax: float = 300.0) -> float:
    """I/E ratio of expected sites near the tip, -ln(1 - f) for each type.

    f is the fraction responding within ``rmax``, less the fraction that
    responds at 3 uA (the false-positive floor).  -ln(1 - f) undoes the
    saturation of a fraction, so the result compares with the model's ratio
    of recruitment probabilities, measured where they are far from saturation.
    """
    def frac(want):
        sel = [(c["red"] == want) & (c["d"] < rmax) for c in cs]
        n = sum(int(s.sum()) for s in sel)
        k = sum(int(c[label][a][s].sum()) for c, s in zip(cs, sel))
        k0 = sum(int(c[label][3][s].sum()) for c, s in zip(cs, sel))
        f0 = k0 / n
        return max((k / n - f0) / (1 - f0), 1e-6)
    return float(np.log(1 - frac(True)) / np.log(1 - frac(False)))


def bootstrap(cs: list[dict], stat, n: int = 1000, seed: int = 0) -> tuple:
    """Hierarchical bootstrap: mice with replacement, then cells within each."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sample = []
        for c in (cs[k] for k in rng.integers(0, len(cs), len(cs))):
            i = rng.integers(0, len(c["d"]), len(c["d"]))
            sample.append({"d": c["d"][i], "red": c["red"][i],
                           "sig": {a: v[i] for a, v in c["sig"].items()},
                           "early": {a: v[i] for a, v in c["early"].items()}})
        vals.append(stat(sample))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# -- recruitment fitted to every cell -----------------------------------------

def _p(th, a, kind, d, h0, local):
    """Probability that a cell responds: floor f plus recruitment.

    h = s_ax h0(r, I)  [+ s_loc (I/20)^1.5 exp(-r^2 / 2 L^2)]
    h0 is the model's own expected sites at unit density, so s_ax is a site
    density in the model's units (9 for excitatory cells in the paper).
    """
    f = 1.0 / (1.0 + np.exp(-th[2]))
    h = np.exp(th[0] if kind == "e" else th[1]) * h0
    if local:
        sl = np.exp(th[3] if kind == "e" else th[4])
        h = h + sl * (a / 20.0) ** 1.5 * np.exp(-d ** 2 / (2 * np.exp(th[5]) ** 2))
    return np.clip(f + (1 - f) * (1 - np.exp(-h)), 1e-9, 1 - 1e-9)


def fit_recruitment(cs: list[dict], h0, label: str, local: bool,
                    x0=None) -> dict:
    """Binomial maximum likelihood over every cell and amplitude.

    ``h0(kind, amp, r)`` returns the model's expected sites at unit density.
    """
    rows = []
    for c in cs:
        for a in AMPS:
            for kind, want in (("e", False), ("i", True)):
                w = c["red"] == want
                rows.append((a, kind, c["d"][w], c[label][a][w].astype(float),
                             h0(kind, a, c["d"][w])))

    def nll(th):
        return -sum(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
                    for a, k, d, y, h in rows
                    for p in (_p(th, a, k, d, h, local),))

    if x0 is None:
        x0 = [np.log(60.0), np.log(40.0), -5.0] + \
             ([np.log(0.3), np.log(0.1), np.log(90.0)] if local else [])
    r = minimize(nll, np.asarray(x0, float), method="Nelder-Mead",
                 options={"maxiter": 20000, "maxfev": 20000,
                          "xatol": 1e-4, "fatol": 1e-3})
    th = r.x
    out = {"label": label, "local": local, "nll": float(r.fun),
           "aic": float(2 * r.fun + 2 * len(th)),
           "s_e": float(np.exp(th[0])), "s_i": float(np.exp(th[1])),
           "floor": float(1 / (1 + np.exp(-th[2]))), "theta": th.tolist()}
    if local:
        out.update({"s_loc_e": float(np.exp(th[3])), "s_loc_i": float(np.exp(th[4])),
                    "L_um": float(np.exp(th[5]))})
    edges = (0, 100, 200, 300, 500)
    pred = {}
    for kind in ("e", "i"):
        for a in AMPS:
            row = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                ps = [_p(th, aa, k, d[(d >= lo) & (d < hi)], h[(d >= lo) & (d < hi)], local)
                      for aa, k, d, y, h in rows if aa == a and k == kind]
                row.append(float(np.concatenate(ps).mean()))
            pred[f"{kind}{a}"] = row
    out["binned_prediction"] = pred
    return out


def fit_site(z: dict, t: dict, amps=(10, 15, 20, 25, 30)) -> tuple:
    """The stimulation site that best explains a mouse's early excitation.

    p(r) = f + (1 - f) A_I exp(-r^2 / 2 s^2), binomial likelihood, from a
    7 x 7 grid of starting points.  Returns (column, row) in pixels.
    """
    sx, sy = z["umpp"][0] / z["mag"], z["umpp"][1] / z["mag"]
    X, Y = z["med"][:, 1] * sx, z["med"][:, 0] * sy
    ys = np.stack([t[a]["early"] for a in amps])

    def nll(th):
        r2 = (X - th[0]) ** 2 + (Y - th[1]) ** 2
        f = 1 / (1 + np.exp(-np.clip(th[3], -30, 30)))
        tot = 0.0
        for k in range(len(amps)):
            A = 1 / (1 + np.exp(-np.clip(th[4 + k], -30, 30)))
            p = np.clip(f + (1 - f) * A * np.exp(-r2 / (2 * np.exp(th[2]) ** 2)),
                        1e-6, 1 - 1e-6)
            tot -= np.sum(ys[k] * np.log(p) + (1 - ys[k]) * np.log(1 - p))
        return tot

    best = None
    for gx in np.linspace(X.min(), X.max(), 7):
        for gy in np.linspace(Y.min(), Y.max(), 7):
            r = minimize(nll, np.r_[gx, gy, np.log(250), -4, np.full(len(amps), -1.0)],
                         method="Nelder-Mead",
                         options={"maxiter": 6000, "xatol": 0.5, "fatol": 1e-3})
            if best is None or r.fun < best.fun:
                best = r
    return best.x[0] / sx, best.x[1] / sy
