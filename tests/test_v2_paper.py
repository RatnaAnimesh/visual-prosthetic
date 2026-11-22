"""Checks on the measurement code in ``prosthesis/v2_paper.py``.

None of these run an experiment.  They check that each readout measures what
the manuscript says it measures:

* the visual-field readout samples cortex where the implant's retinotopy
  says it should, and the disc diameter of a disc is its diameter;
* the repulsion measure is exactly zero for any linear model, so a nonzero
  value is the model's doing and not the estimator's;
* the decoder built for this model reads this model's orientation map;
* the independent forward model used for optimal stimulation is right
  about each electrode on its own and adds them, nothing else.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from prosthesis.cortex_ssn import BiophysicalCortex
from prosthesis.v2_paper import (AMP_GRID, Forward, Retinotopy, bio_code,
                                 centroid_separation, crossing, disc_diameter,
                                 repulsion)


@pytest.fixture(scope="module")
def vf():
    return Retinotopy()


def test_retinotopy_reads_a_linear_field_exactly(vf):
    """Bilinear sampling of a field linear in x and y returns x and y."""
    xs = np.arange(15000.0, 26000.0, 100.0)
    ys = np.arange(-10000.0, 0.0, 100.0)
    g = {"xs": xs, "ys": ys}
    X, Y = np.meshgrid(xs, ys)
    inside = ((vf.cx >= xs[0]) & (vf.cx <= xs[-1]) & (vf.cy >= ys[0])
              & (vf.cy <= ys[-1])).reshape(vf.shape)
    assert inside.mean() > 0.5
    sx = vf.sample(X, g)
    sy = vf.sample(Y, g)
    np.testing.assert_allclose(sx[inside], vf.cx.reshape(vf.shape)[inside],
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(sy[inside], vf.cy.reshape(vf.shape)[inside],
                               rtol=0, atol=1e-6)
    assert np.all(sx[~inside] == 0.0)


def test_disc_diameter_of_a_disc(vf):
    step = 0.01
    x = np.arange(-1, 1, step)
    X, Y = np.meshgrid(x, x)
    for d in (0.2, 0.5, 1.0):
        img = (np.hypot(X, Y) <= d / 2).astype(float)
        assert disc_diameter(img, step) == pytest.approx(d, rel=0.03)
    # a Gaussian's half-max diameter is its FWHM
    sig = 0.15
    g = np.exp(-(X ** 2 + Y ** 2) / (2 * sig ** 2))
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sig
    assert disc_diameter(g, step) == pytest.approx(fwhm, rel=0.03)
    assert disc_diameter(np.zeros_like(g), step) == 0.0


def _blob(X, Y, x0, w=300.0):
    return np.exp(-((X - x0) ** 2 + Y ** 2) / (2 * w ** 2))


@pytest.mark.parametrize("sep", [400.0, 800.0, 1600.0])
def test_repulsion_is_zero_for_a_linear_model(sep):
    from prosthesis.v2_experiments import X0
    xs = np.arange(X0 - 3000, X0 + sep + 3000, 20.0)
    ys = np.arange(-2000.0, 2000.0, 20.0)
    X, Y = np.meshgrid(xs, ys)
    a, b = _blob(X, Y, X0), _blob(X, Y, X0 + sep)
    g = {"xs": xs, "ys": ys}
    assert repulsion({"a": a, "b": b, "ab": a + b}, g, sep) == 0.0
    # ... and a pair whose blobs sit further apart scores positive
    a2, b2 = _blob(X, Y, X0 - 50), _blob(X, Y, X0 + sep + 50)
    assert repulsion({"a": a, "b": b, "ab": a2 + b2}, g, sep) > 0.0
    # ... while suppression between them, which pushes each centroid
    # outward, also scores positive, and only excitation counts
    mid = _blob(X, Y, X0 + sep / 2, w=sep / 4)
    assert repulsion({"a": a, "b": b, "ab": a + b - 0.3 * mid}, g, sep) > 0.0


def test_centroid_separation_of_two_points():
    xs = np.arange(0.0, 1000.0, 10.0)
    ys = np.arange(0.0, 100.0, 10.0)
    f = np.zeros((ys.size, xs.size))
    f[5, 20] = 1.0
    f[5, 70] = 2.0
    assert centroid_separation(f, xs, ys, 200.0, 700.0) == pytest.approx(500.0)
    assert np.isnan(centroid_separation(np.zeros_like(f), xs, ys, 200, 700))


def test_decoder_reads_this_models_own_orientation_map():
    m = BiophysicalCortex(min_n=48, margin_um=1500.0)
    xe, ye = np.array([19000.0, 21000.0]), np.array([-5000.0, -5000.0])
    g = m._ensure_cortical_grid(xe, ye)
    code = bio_code(g)
    th = np.rad2deg(g["sheet"].theta) % 180.0
    h = g["xs"][1] - g["xs"][0]
    inside = ((code.ctx_x >= g["xs"][0]) & (code.ctx_x <= g["xs"][-1])
              & (code.ctx_y >= g["ys"][0]) & (code.ctx_y <= g["ys"][-1]))
    assert inside.sum() > 100
    i = np.rint((code.ctx_y[inside] - g["ys"][0]) / h).astype(int)
    j = np.rint((code.ctx_x[inside] - g["xs"][0]) / h).astype(int)
    np.testing.assert_allclose(code.theta[inside], th[i, j], atol=1e-9)
    assert np.all((code.theta >= 0) & (code.theta < 180))


def _table_forward(n=4, npx=6, seed=0):
    """A Forward with a random single-electrode table and no model."""
    rng = np.random.default_rng(seed)
    F = Forward.__new__(Forward)
    F.n = n
    F.table = rng.normal(size=(n, AMP_GRID.size, npx, npx))
    F.table[:, 0] = 0.0
    F.scale = 1.7
    return F


def test_independent_model_is_exact_for_each_electrode_alone():
    F = _table_forward()
    for i in range(F.n):
        for k in (0, 1, 7, AMP_GRID.size - 1):
            v = np.zeros(F.n)
            v[i] = AMP_GRID[k]
            np.testing.assert_allclose(F.independent(v),
                                       F.table[i, k] * F.scale, atol=1e-12)


def test_independent_model_adds_and_interpolates():
    F = _table_forward()
    v = np.array([110.0, 0.0, 333.0, 600.0])
    total = np.zeros_like(F.table[0, 0])
    for i, a in enumerate(v):
        u = np.zeros(F.n)
        u[i] = a
        total += F.independent(u)
    np.testing.assert_allclose(F.independent(v), total, atol=1e-12)
    # halfway between two table entries is their mean
    u = np.zeros(F.n)
    u[1] = 0.5 * (AMP_GRID[3] + AMP_GRID[4])
    np.testing.assert_allclose(
        F.independent(u), 0.5 * (F.table[1, 3] + F.table[1, 4]) * F.scale,
        atol=1e-12)
    # currents are clipped to the allowed range
    np.testing.assert_allclose(F.independent(np.full(F.n, 1e4)),
                               F.independent(np.full(F.n, AMP_GRID[-1])))
    np.testing.assert_allclose(F.independent(np.full(F.n, -50.0)), 0.0)


def test_two_peak_separation_finds_one_maximum_per_side():
    from prosthesis.phase6_controls import axis_profile
    from prosthesis.phase6_controls import two_peak_separation as half_peak_separation
    t = np.linspace(-0.6, 1.6, 2201)
    two = np.exp(-(t - 0.05) ** 2 / 0.02) + 0.9 * np.exp(-(t - 1.0) ** 2
                                                         / 0.02)
    assert half_peak_separation(t, two, 1000.0) == pytest.approx(950.0,
                                                                 abs=2.0)
    # a texture bump next to the first electrode, taller than the second
    # electrode's peak, does not steal the second maximum
    bumpy = two + 0.95 * np.exp(-(t - 0.25) ** 2 / 0.002)
    assert half_peak_separation(t, bumpy, 1000.0) > 700.0
    # one merged peak has no second maximum
    merged = np.exp(-(t - 0.5) ** 2 / 0.2)
    assert half_peak_separation(t, merged, 1000.0) is None
    # distances come out in microns on axis_profile's own parameterisation
    xs = np.arange(0.0, 6000.0, 20.0)
    ys = np.arange(-1000.0, 1000.0, 20.0)
    X, Y = np.meshgrid(xs, ys)
    f = _blob(X, Y, 2000.0) + _blob(X, Y, 3500.0)
    tt, vv = axis_profile(f, {"xs": xs, "ys": ys}, 2000.0, 0.0, 3500.0, 0.0)
    assert half_peak_separation(tt, vv, 1500.0) == pytest.approx(1500.0,
                                                                 abs=10.0)


def test_linear_fit_fidelity_separates_reachable_from_unreachable():
    from prosthesis.v2_paper import linear_fit_fidelity, shape_targets
    n = 16
    y, x = np.mgrid[0:n, 0:n]
    rng = np.random.default_rng(3)
    centres = rng.uniform(2, n - 2, size=(12, 2))
    singles = np.array([np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 4.0)
                        for cx, cy in centres])
    # any sum of the singles is reached exactly
    target = singles[[0, 4, 7]].sum(0)
    assert linear_fit_fidelity(singles, target) == pytest.approx(1.0,
                                                                abs=1e-9)
    # a thin bar is not, with a dozen blurred phosphenes
    for shape in shape_targets(n).values():
        assert linear_fit_fidelity(singles, shape) < 0.6
    assert linear_fit_fidelity(singles, np.zeros((n, n))) == 0.0


def test_crossing_finds_the_first_turn_to_positive():
    """The sign change is interpolated, and only negative-to-positive counts."""
    assert crossing([1.0, 2.0, 3.0], [-1.0, 1.0, 3.0]) == pytest.approx(1.5)
    assert crossing([1.0, 2.0, 3.0], [1.0, -1.0, 3.0]) == pytest.approx(2.25)
    assert crossing([1.0, 2.0], [-1.0, -0.5]) is None
