"""Measurements a visual physiologist would make on these neurons.

Everything here operates on spike trains, so the same routine works on the
model and would work unchanged on real recordings.  That is deliberate: the
point of the model is to be falsifiable by the same measurements that were
used to characterise the real cortex.
"""

from __future__ import annotations

import numpy as np


def circular_mean_orientation(rates: np.ndarray, oris_deg: np.ndarray
                              ) -> tuple[float, float]:
    """Population-vector estimate of orientation, and its resultant length.

    Kandel Fig. 25-17: each cell's preferred orientation is a "line label",
    its firing rate is the weight of its vote, and the population's estimate
    is the vector sum.  Orientation is defined modulo 180 degrees, so the
    angles are doubled before averaging and halved afterwards -- otherwise a
    cell preferring 175 degrees and one preferring 5 degrees would cancel
    instead of reinforcing.
    """
    r = np.asarray(rates, dtype=float)
    if r.sum() <= 0:
        return float("nan"), 0.0
    ang = 2 * np.deg2rad(oris_deg)
    v = np.sum(r * np.exp(1j * ang)) / np.sum(r)
    return float((np.rad2deg(np.angle(v)) / 2) % 180.0), float(np.abs(v))


def orientation_selectivity_index(rates: np.ndarray, oris_deg: np.ndarray
                                  ) -> float:
    """1 - circular variance: 0 for a flat cell, 1 for a perfectly tuned one.

    Preferred over the classic (R_pref - R_orth)/(R_pref + R_orth) because it
    uses the whole tuning curve rather than two points, so a single noisy
    sample cannot dominate it.
    """
    _, resultant = circular_mean_orientation(rates, oris_deg)
    return resultant


def tuning_width_hwhh(rates: np.ndarray, oris_deg: np.ndarray) -> float:
    """Half-width at half-height of the tuning curve, in degrees.

    Interpolated on the circular curve after subtracting the baseline, which
    is how tuning widths are reported in the experimental literature (V1
    orientation tuning is typically 20-40 degrees HWHH).
    """
    r = np.asarray(rates, dtype=float)
    o = np.asarray(oris_deg, dtype=float)
    if r.max() <= r.min():
        return float("nan")
    k = int(np.argmax(r))
    # Recentre so the peak sits in the middle, respecting the 180 deg wrap.
    shift = np.argsort(((o - o[k] + 90.0) % 180.0))
    oo = ((o[shift] - o[k] + 90.0) % 180.0)
    rr = r[shift]
    half = (rr.max() + rr.min()) / 2.0
    above = rr >= half
    if above.all():
        return float("nan")
    # Width of the contiguous above-half region around the peak.
    peak = int(np.argmax(rr))
    lo = peak
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = peak
    while hi < len(rr) - 1 and above[hi + 1]:
        hi += 1
    width = oo[hi] - oo[lo]
    return float(width / 2.0)


def f0_f1(spike_times: np.ndarray, t0: float, t1: float, freq_hz: float
          ) -> tuple[float, float]:
    """Mean rate (F0) and the amplitude of modulation at ``freq_hz`` (F1).

    The F1/F0 ratio is the standard operational definition separating simple
    from complex cells: a simple cell's response follows the bright and dark
    bars of a drifting grating in and out of its subregions, giving F1/F0 > 1,
    whereas a complex cell responds throughout the cycle, giving F1/F0 < 1.
    Both are computed in spikes/s so the ratio is dimensionless.
    """
    m = (spike_times >= t0) & (spike_times < t1)
    st = spike_times[m]
    dur = (t1 - t0) / 1000.0
    if dur <= 0:
        return 0.0, 0.0
    f0 = st.size / dur
    if st.size == 0:
        return 0.0, 0.0
    ph = 2 * np.pi * freq_hz * (st - t0) / 1000.0
    # Amplitude of the first Fourier component of the spike train, in spikes/s.
    f1 = 2.0 * np.abs(np.sum(np.exp(1j * ph))) / dur
    return float(f0), float(f1)


def modulation_ratio(spike_times: np.ndarray, t0: float, t1: float,
                     freq_hz: float) -> float:
    f0, f1 = f0_f1(spike_times, t0, t1, freq_hz)
    return float(f1 / f0) if f0 > 0 else float("nan")


def ocular_dominance_ratio(rate_left: float, rate_right: float,
                           spont: float = 0.0) -> float:
    """Contralateral bias in [-1, +1]; 0 means the eyes drive the cell equally.

    Hubel and Wiesel's seven-group ocular-dominance histogram is a binned
    version of this quantity.
    """
    l = max(rate_left - spont, 0.0)
    r = max(rate_right - spont, 0.0)
    if l + r <= 0:
        return 0.0
    return float((r - l) / (r + l))


def naka_rushton(c: np.ndarray, r_max: float, c50: float, n: float,
                 base: float = 0.0) -> np.ndarray:
    """Saturating contrast-response function."""
    c = np.asarray(c, dtype=float)
    return base + r_max * c ** n / (c ** n + c50 ** n)


def fit_naka_rushton(contrasts: np.ndarray, rates: np.ndarray) -> dict:
    """Least-squares fit of the contrast-response function.

    Returns the semisaturation contrast c50 and exponent n.  A cortical cell
    whose c50 shifts with surround stimulation but whose R_max does not is
    showing contrast gain control rather than simple subtraction.
    """
    from scipy.optimize import curve_fit

    c = np.asarray(contrasts, dtype=float)
    r = np.asarray(rates, dtype=float)
    base0 = float(r.min())
    p0 = [max(r.max() - base0, 1e-3), 0.3, 2.0, base0]
    try:
        popt, _ = curve_fit(
            naka_rushton, c, r, p0=p0, maxfev=20000,
            bounds=([0, 1e-3, 0.5, 0], [np.inf, 1.0, 6.0, np.inf]),
        )
        return dict(r_max=popt[0], c50=popt[1], n=popt[2], base=popt[3])
    except Exception:
        return dict(r_max=np.nan, c50=np.nan, n=np.nan, base=np.nan)


def psth(spike_times: np.ndarray, t0: float, t1: float, bin_ms: float = 10.0,
         n_trials: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Peri-stimulus time histogram in spikes/s."""
    edges = np.arange(t0, t1 + bin_ms, bin_ms)
    counts, _ = np.histogram(spike_times, bins=edges)
    rate = counts * 1000.0 / (bin_ms * max(n_trials, 1))
    centres = (edges[:-1] + edges[1:]) / 2.0
    return centres, rate


def per_neuron_spike_times(spikes, idx: np.ndarray) -> dict:
    """Split a SpikeRecorder into per-neuron spike-time arrays."""
    t, i = spikes.arrays()
    out = {}
    if t.size == 0:
        return {int(k): np.empty(0) for k in idx}
    order = np.argsort(i, kind="stable")
    t, i = t[order], i[order]
    wanted = set(int(k) for k in idx)
    starts = np.searchsorted(i, np.asarray(sorted(wanted)), side="left")
    ends = np.searchsorted(i, np.asarray(sorted(wanted)), side="right")
    for k, s, e in zip(sorted(wanted), starts, ends):
        out[int(k)] = t[s:e]
    return out


def fano_factor(counts: np.ndarray) -> float:
    """Variance-to-mean ratio of spike counts across repeats.

    Kandel's point about response variability: repeated presentations of the
    same stimulus give a range of responses, and that variability sets the
    discrimination threshold.  A Poisson process gives exactly 1.
    """
    c = np.asarray(counts, dtype=float)
    m = c.mean()
    return float(c.var() / m) if m > 0 else float("nan")


def population_vector_decode(rates: np.ndarray, prefs_deg: np.ndarray
                             ) -> tuple[float, float]:
    """Decode the stimulus orientation from a population of tuned cells."""
    return circular_mean_orientation(rates, prefs_deg)
