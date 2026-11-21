"""Checks on the biophysical forward model, ``prosthesis/cortex_ssn.py``.

Three kinds of test, in roughly increasing order of how much they would
matter if they failed:

* bookkeeping -- the orientation-harmonic expansion reconstructs the kernel,
  the weights do not depend on grid resolution, contributions from separate
  electrodes add where physics says they must;
* mechanism -- the network is an inhibition-stabilised supralinear network
  (paradoxical inhibitory response, supralinear-to-sublinear summation,
  surround suppression), and directly driven cells are not vetoed;
* data -- recruitment is sparse and fills in rather than spreads (Histed et
  al. 2009), stimulation suppresses ongoing activity (Butovas & Schwarz
  2003), and electrical plus visual drive sums sub-additively (Nassi et al.
  2015).

The resolution test exists because it failed: the weights were once scaled
by the inverse of the right factor, which is invisible on the 107 um grid
every experiment uses and wrong by 16x on a grid half as fine.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
from scipy import fft as sfft

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from prosthesis.cortex_ssn import (BiophysicalCortex, RecruitmentParams,
                                   SSNParams, SSNSheet, _ori_harmonics,
                                   effective_rate, orientation_map_periodic)
from prosthesis.v2_validate import grating_patch

X0, Y0 = 20000.0, -5000.0


@pytest.fixture(scope="module")
def small():
    """A 5 mm sheet: large enough for the far kernel, fast enough to test."""
    return BiophysicalCortex(min_n=48, margin_um=1500.0)


@pytest.fixture(scope="module")
def fine():
    """Fine grid for recruitment alone (no network is run on it)."""
    return BiophysicalCortex(step_um=26.75, margin_um=2600.0)


def _pair(m, sep, amp):
    xe, ye = np.array([X0, X0 + sep]), np.array([Y0, Y0])
    g = m._ensure_cortical_grid(xe, ye)
    fa = m._activation_field([amp, 0.0], xe, ye, g)
    fb = m._activation_field([0.0, amp], xe, ye, g)
    fab = m._activation_field([amp, amp], xe, ye, g)
    return fa, fb, fab


# ---------------------------------------------------------------------------
# Synaptic depression
# ---------------------------------------------------------------------------

def test_effective_rate_is_identity_at_the_reference_rate():
    assert effective_rate(20.0, 0.45, 300.0, 20.0) == pytest.approx(20.0)
    assert effective_rate(30.0, 0.35, 150.0, 30.0) == pytest.approx(30.0)


def test_effective_rate_rises_and_saturates_at_ref_plus_one_over_U_tau():
    U, tau, ref = 0.45, 300.0, 20.0
    r_d = 1000.0 / (U * tau)
    f = [5.0, 20.0, 50.0, 100.0, 300.0, 1000.0, 1e6]
    v = [effective_rate(x, U, tau, ref) for x in f]
    assert np.all(np.diff(v) > 0)
    assert v[-1] < ref + r_d
    assert v[-1] == pytest.approx(ref + r_d, rel=1e-4)
    assert effective_rate(0.0, U, tau, ref) == 0.0
    # Driving at 300 Hz buys little more than driving at 50 Hz.
    assert v[4] / v[2] < 1.25


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_orientation_map_range_and_column_period():
    n, step, period = 128, 50.0, 1000.0
    th = orientation_map_periodic(n, step, period, seed=1)
    assert th.min() >= 0.0 and th.max() < np.pi
    z = np.exp(2j * th)
    P = np.abs(sfft.fft2(z - z.mean())) ** 2
    f = np.hypot(*np.meshgrid(sfft.fftfreq(n, step), sfft.fftfreq(n, step)))
    df = 1.0 / (n * step)
    k = np.rint(f / df).astype(int).ravel()
    radial = np.bincount(k, P.ravel()) / np.maximum(np.bincount(k), 1)
    peak = radial[1:].argmax() + 1
    assert peak * df == pytest.approx(1.0 / period, rel=0.15)


@pytest.mark.parametrize("A,B,s,tol", [(0.20, 0.80, 55.0, 0.03),
                                       (0.14, 0.86, 25.0, 0.002)])
def test_orientation_harmonics_reconstruct_the_kernel(A, B, s, tol):
    # The wide kernel has a kink where it wraps at +-90 deg, so its series
    # converges slowly; four harmonics leave ~3% of its range.
    c = _ori_harmonics(A, B, s, 4)
    d = np.linspace(-np.pi / 2, np.pi / 2, 721)
    q = A + B * np.exp(-np.rad2deg(d) ** 2 / (2 * s ** 2))
    rec = sum(c[m] * np.cos(2 * m * d) for m in range(5))
    assert np.abs(rec - q).max() < tol


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------

def test_recurrent_input_is_linear_in_rates():
    sh = SSNSheet(48, 107.0, 0)
    rng = np.random.default_rng(0)
    a, b = rng.random((2, 48, 48)), rng.random((2, 48, 48))
    lhs = sh.recurrent(a[0] + b[0], a[1] + b[1])
    ra, rb = sh.recurrent(*a), sh.recurrent(*b)
    for x, y, z in zip(lhs, ra, rb):
        np.testing.assert_allclose(x, y + z, atol=1e-10)


def test_recurrent_input_does_not_depend_on_grid_resolution():
    coarse, finer = SSNSheet(48, 107.0, 0), SSNSheet(96, 53.5, 0)
    for re, ri in ((1.0, 0.0), (0.0, 1.0)):
        a = coarse.recurrent(np.full((48, 48), re), np.full((48, 48), ri))
        b = finer.recurrent(np.full((96, 96), re), np.full((96, 96), ri))
        for x, y in zip(a, b):
            assert y.mean() == pytest.approx(x.mean(), rel=0.02)


def test_spontaneous_state(small):
    g = small._ensure_cortical_grid([X0], [Y0])
    rE, rI, _ = small.baseline(g)
    # Obeid & Miller's spiking network: about 1.5 Hz (E) and 3 Hz (I).
    assert 0.8 < rE.mean() < 2.5
    assert 2.5 < rI.mean() < 7.0
    assert rI.mean() > rE.mean()


def test_network_is_inhibition_stabilised_at_spontaneous_rates():
    """Paradoxical effect: more drive to I cells lowers I rates."""
    sh = SSNSheet(48, 107.0, 0)
    _, rI = sh.run(10.0, 10.0, t_ms=800.0, tol=1e-7)
    _, rI_up = sh.run(10.0, 11.0, t_ms=800.0, tol=1e-7)
    assert rI_up.mean() < rI.mean()


def test_summation_is_supralinear_when_weak_and_sublinear_when_strong():
    p = SSNParams(r_max_e=1e9, r_max_i=1e9)      # no ceiling to help it
    sh = SSNSheet(75, 107.0, 3, params=p)
    i = j = 40
    h0 = grating_patch(sh, i, j, 1.0, 100.0) / 50.0

    def r(c):
        return sh.run(c * h0, c * h0, t_ms=400.0)[0][i, j]

    weak = np.log(r(4.0) / r(2.0)) / np.log(2.0)
    strong = np.log(r(128.0) / r(64.0)) / np.log(2.0)
    assert weak > 1.5
    assert strong < 1.0


def test_surround_suppression_of_excitatory_cells():
    sh = SSNSheet(75, 107.0, 3)
    i = j = 40
    resp = [sh.run(grating_patch(sh, i, j, L, 16.4),
                   grating_patch(sh, i, j, L, 16.4), t_ms=300.0)[0][i, j]
            for L in (0.6, 1.2, 8.0)]
    si = (max(resp) - resp[-1]) / max(resp)
    assert si > 0.4                               # published mean 0.79


def test_slow_inhibition_leaves_steady_states_unchanged():
    fast = SSNSheet(48, 107.0, 0)
    slow = SSNSheet(48, 107.0, 0, params=SSNParams(slow_inh_frac=0.3))
    a = fast.run(10.0, 10.0, t_ms=800.0, tol=1e-7)
    b = slow.run(10.0, 10.0, t_ms=1500.0, tol=None)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, rtol=1e-3)


# ---------------------------------------------------------------------------
# Direct recruitment
# ---------------------------------------------------------------------------

def test_initiation_radius_matches_kumaravelu():
    m = BiophysicalCortex()
    # Median of a uniform ball is 0.794 R.
    for amp, med in ((5.0, 26.0), (20.0, 56.0), (50.0, 105.0)):
        assert 0.794 * m.init_radius_um(amp) == pytest.approx(med, rel=0.30)


def test_hits_add_across_electrodes(small):
    xe, ye = np.array([X0, X0 + 800.0]), np.array([Y0, Y0])
    g = small._ensure_cortical_grid(xe, ye)
    ha = small.hits([60.0, 0.0], xe, ye, g)
    hb = small.hits([0.0, 90.0], xe, ye, g)
    hab = small.hits([60.0, 90.0], xe, ye, g)
    for k in range(2):
        np.testing.assert_allclose(hab[k], ha[k] + hb[k], rtol=1e-12)


def test_occupancy_is_subadditive_but_never_below_either_alone(small):
    xe, ye = np.array([X0, X0 + 400.0]), np.array([Y0, Y0])
    g = small._ensure_cortical_grid(xe, ye)
    pa, _ = small.recruited([200.0, 0.0], xe, ye, g)
    pb, _ = small.recruited([0.0, 200.0], xe, ye, g)
    pab, _ = small.recruited([200.0, 200.0], xe, ye, g)
    assert np.all(pab <= pa + pb + 1e-12)
    assert np.all(pab >= np.maximum(pa, pb) - 1e-12)
    assert (pa + pb - pab).max() > 1e-3


def test_recruitment_is_sparse_and_calibrated_to_histed(fine):
    g = fine._ensure_cortical_grid([X0], [Y0])
    pe, _ = fine.recruited([8.5], [X0], [Y0], g)
    assert pe.max() < 0.02                        # sparse: no ball of cells
    cells = (pe * 1300.0 * (fine.step / 1000.0) ** 2).sum()
    assert 3.5 < cells < 7.5                      # Histed: ~5 cells


def _median_distance(p, d):
    w, dd = p.ravel(), d.ravel()
    o = np.argsort(dd)
    cw = np.cumsum(w[o])
    return dd[o][np.searchsorted(cw, cw[-1] / 2)]


def test_inhibitory_recruitment_matches_dadarlat(fine):
    """Similar I and E fractions (~1.1), I recruited closer to the tip."""
    g = fine._ensure_cortical_grid([X0], [Y0])
    X, Y = np.meshgrid(g["xs"], g["ys"])
    d = np.hypot(X - X0, Y - Y0)
    for amp in (10.0, 50.0):
        pe, pi = fine.recruited([amp], [X0], [Y0], g)
        ratio = pi[d < 300].mean() / pe[d < 300].mean()
        assert 0.9 < ratio < 1.4
        assert _median_distance(pi, d) < 0.8 * _median_distance(pe, d)


def test_recruitment_fills_in_rather_than_spreads(fine):
    g = fine._ensure_cortical_grid([X0], [Y0])
    X, Y = np.meshgrid(g["xs"], g["ys"])
    d = np.hypot(X - X0, Y - Y0)

    def median(p):
        w, dd = p.ravel(), d.ravel()
        o = np.argsort(dd)
        cw = np.cumsum(w[o])
        return dd[o][np.searchsorted(cw, cw[-1] / 2)]

    p5, _ = fine.recruited([5.0], [X0], [Y0], g)
    p50, _ = fine.recruited([50.0], [X0], [Y0], g)
    assert p50.sum() / p5.sum() > 20.0            # many more cells
    assert median(p50) / median(p5) < 1.3         # at about the same distance
    assert fine.init_radius_um(50.0) / fine.init_radius_um(5.0) > 3.0


# ---------------------------------------------------------------------------
# The forward model
# ---------------------------------------------------------------------------

def test_no_current_evokes_nothing(small):
    g = small._ensure_cortical_grid([X0], [Y0])
    ev = small._activation_field([0.0], [X0], [Y0], g)
    assert np.abs(ev).max() < 1e-3


def test_direct_drive_is_a_floor_that_inhibition_cannot_remove(small):
    g = small._ensure_cortical_grid([X0], [Y0])
    rho_e, _ = small._rho()
    small._activation_field([150.0], [X0], [Y0], g, h_e=0.0, h_i=40.0)
    L = small.last
    assert np.all(L["R_e"] >= L["p_e"] * rho_e - 1e-9)


def test_occupancy_alone_is_subadditive():
    m = BiophysicalCortex(min_n=48, margin_um=1500.0, network=False)
    fa, fb, fab = _pair(m, 400.0, 300.0)
    assert (fab - fa - fb).max() <= 1e-12
    assert (fab - fa - fb).sum() / (fa + fb).sum() < -0.01


def test_low_currents_superpose(small):
    fa, fb, fab = _pair(small, 1600.0, 10.0)
    lin = fa + fb
    assert abs((fab - lin).sum() / lin.sum()) < 0.002


def test_distant_electrodes_do_not_interact_numerically():
    """Guards the convergence tolerance.

    Electrodes 4.5 mm apart barely share any cortex.  With the original
    tolerance the summed residual of ~10^4 unconverged sites read as a 1%
    interaction at 10 uA; converged, it is a few hundredths of a percent.
    """
    m = BiophysicalCortex()
    fa, fb, fab = _pair(m, 4500.0, 10.0)
    lin = fa + fb
    assert abs((fab - lin).sum() / lin.sum()) < 1e-3


def test_stimulation_suppresses_ongoing_activity_nearby(small):
    g = small._ensure_cortical_grid([X0], [Y0])
    small._activation_field([100.0], [X0], [Y0], g)
    L = small.last
    assert (L["indirect_e"] / L["baseline_e"]).min() < -0.2
    assert L["indirect_e"].sum() < 0


def test_visual_and_electrical_drive_sum_subadditively(small):
    g = small._ensure_cortical_grid([X0], [Y0])
    sheet = g["sheet"]
    X, Y = np.meshgrid(g["xs"], g["ys"])
    i, j = np.unravel_index(np.argmin(np.hypot(X - X0, Y - Y0)), X.shape)
    h = grating_patch(sheet, i, j, 1.5, 16.4)
    vis = small._activation_field([0.0], [X0], [Y0], g, h, h)[i, j]
    stim = small._activation_field([150.0], [X0], [Y0], g)[i, j]
    both = small._activation_field([150.0], [X0], [Y0], g, h, h)[i, j]
    assert vis > 0 and stim > 0
    assert both < vis + stim


def test_time_course_settles_to_the_steady_state(small):
    xe, ye = np.array([X0]), np.array([Y0])
    g = small._ensure_cortical_grid(xe, ye)
    ss = small._activation_field([100.0], xe, ye, g)
    i, j = small.site_index(g, X0, Y0)
    assert g["xs"][j] == pytest.approx(X0) and g["ys"][i] == pytest.approx(Y0)
    ts, out = small.simulate([(0.0, 250.0, (100.0,))], xe, ye, g, t_end=250.0)
    assert out[-1, 0] == pytest.approx(ss[i, j], rel=0.03)
    # And relaxes back to spontaneous once the train stops.
    ts2, out2 = small.simulate([(0.0, 50.0, (100.0,))], xe, ye, g,
                               t_end=400.0)
    assert abs(out2[-1, 0]) < 0.05 * abs(ss[i, j])


def test_recruitment_parameters_are_frozen():
    r = RecruitmentParams()
    with pytest.raises(Exception):
        r.K_ax = 1.0                              # type: ignore[misc]
