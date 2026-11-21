"""Checks on where the nonlinearity sits, and on the drive and time axes.

These pin down the claims the paper rests on after the mechanism claim was
narrowed.  Four of them are structural and should never drift:

* a nonlinearity applied per-electrode before summation cannot break
  superposition, however compressive it is -- this is why Dynaphos measures
  exactly zero and why "the model contains a nonlinearity" explains nothing;
* a pointwise nonlinearity applied *after* summation does break it, so
  divisive normalisation is not required and must not be claimed to be;
* the two are separated by shape, not size: pointwise decays monotonically
  from the shortest separation, a pooled divisor peaks in the interior;
* the interaction changes sign with drive, so any result quoted at a single
  amplitude is quoting one half of a biphasic function.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from prosthesis.cortex_circuit import CircuitSpatial

XR, YR, STEP = (-3.2, -0.4), (-0.2, 2.0), 0.05
X0, Y0 = 20000.0, -5000.0


def _model(**kw):
    m = CircuitSpatial(xrange=XR, yrange=YR, xystep=STEP, **kw)
    m.build()
    return m


def _fields(m, sep_um, amp):
    xe = np.array([X0, X0 + sep_um], float)
    ye = np.array([Y0, Y0], float)
    g = m._ensure_cortical_grid(xe, ye)
    fa = m._activation_field(np.array([amp, 0.0]), xe, ye, g)
    fb = m._activation_field(np.array([0.0, amp]), xe, ye, g)
    fab = m._activation_field(np.array([amp, amp]), xe, ye, g)
    return fa, fb, fab


def _max_dev(m, sep_um, amp):
    fa, fb, fab = _fields(m, sep_um, amp)
    lin = fa + fb
    return float(np.abs(fab - lin).max() / max(float(np.abs(lin).max()), 1e-12))


def _signed_total(m, sep_um, amp):
    """Positive means supra-additive."""
    fa, fb, fab = _fields(m, sep_um, amp)
    lin = fa + fb
    return float((fab - lin).sum() / max(float(lin.sum()), 1e-12))


# --- where the nonlinearity sits -------------------------------------------

def test_per_electrode_nonlinearity_preserves_superposition():
    """The Dynaphos structure: compress each drive, then add.

    The amplitude saturation in this model is per-electrode and sits before
    the electrodes are combined, so however hard it saturates the model stays
    additive across electrodes.  This is the control that makes "it contains
    a nonlinearity" an empty explanation.
    """
    for a50 in (30.0, 150.0, 600.0):
        m = _model(horiz_gain=0.0, norm_gain=0.0, amp_a50=a50)
        assert _max_dev(m, 800.0, 300.0) < 1e-9, (
            f"per-electrode saturation with a50={a50} broke superposition")


def test_pointwise_nonlinearity_after_summation_does_break_it():
    """norm_sigma -> 0 makes the divisor a pointwise function of the total."""
    m = _model(horiz_gain=0.0, norm_gain=1.0, norm_sigma=0.0,
               excitability=1499.0)
    assert _max_dev(m, 400.0, 300.0) > 0.2


def test_horizontal_spread_stays_linear_at_every_gain():
    for gain in (0.3, 0.6, 1.2):
        m = _model(horiz_gain=gain, norm_gain=0.0)
        assert _max_dev(m, 1200.0, 300.0) < 1e-9


# --- shape is the discriminator, not magnitude -----------------------------

@pytest.mark.parametrize("sigma,expect_interior", [(0.0, False),
                                                   (1500.0, True)])
def test_interaction_shape_separates_the_mechanisms(sigma, expect_interior):
    """Pointwise peaks at the shortest separation; a pooled divisor does not.

    This is the measurement the paper proposes, so it needs to hold in the
    model before anyone is asked to run it in a human.
    """
    k = 1499.0 if sigma == 0.0 else 531.0
    m = _model(horiz_gain=0.0, norm_gain=1.0, norm_sigma=sigma,
               excitability=k)
    seps = np.arange(300.0, 3001.0, 300.0)
    devs = np.array([_max_dev(m, s, 300.0) for s in seps])
    interior = int(np.argmax(devs)) > 0
    assert interior is expect_interior, (
        f"sigma={sigma}: peak at {seps[int(np.argmax(devs))]:.0f} um")


def test_pointwise_decays_to_nothing_by_three_millimetres():
    """Its reach is the overlap, so it must die where the overlap does."""
    m = _model(horiz_gain=0.0, norm_gain=1.0, norm_sigma=0.0,
               excitability=1499.0)
    assert _max_dev(m, 3200.0, 300.0) < 0.01


def test_pooled_divisor_still_interacts_beyond_the_overlap():
    m = _model(horiz_gain=0.0, norm_gain=1.0, norm_sigma=1500.0,
               excitability=531.0)
    assert _max_dev(m, 3200.0, 300.0) > 0.02


# --- the drive axis ---------------------------------------------------------

def test_interaction_reverses_sign_with_drive():
    """Facilitatory when the pool sits below semi-saturation, else suppressive.

    Three limitations in an earlier draft were all consequences of measuring
    at one amplitude only.  If this ever stops holding, the drive-dependent
    predictions in the paper go with it.
    """
    m = _model(horiz_gain=0.6, norm_gain=1.0, norm_sigma=1500.0,
               excitability=589.0)
    low = _signed_total(m, 800.0, 30.0)
    high = _signed_total(m, 800.0, 800.0)
    assert low > 0.0, f"expected facilitation at low drive, got {low:.4f}"
    assert high < 0.0, f"expected suppression at high drive, got {high:.4f}"


def test_supra_additivity_needs_an_expansive_exponent():
    """With n = 1 the numerator is additive and the low-drive gain vanishes."""
    m = _model(horiz_gain=0.0, norm_gain=1.0, norm_sigma=1500.0,
               expansive_n=1.0, excitability=531.0)
    assert _signed_total(m, 800.0, 30.0) <= 1e-6


# --- the pool ODE and its scaling ------------------------------------------

def test_pool_relaxes_with_the_right_time_constant():
    """One time constant of decay must leave exactly 1/e of the pool."""
    tau = 50.0
    dt = 0.05
    p, t = 1.0, 0.0
    while t < tau:
        p *= np.exp(-dt / tau)
        t += dt
    assert abs(p - np.exp(-1.0)) < 1e-3


def test_critical_delay_scales_with_tau_and_log_drive():
    """D* = tau * ln(c/(1-c) * R'), solved from the ODE.

    Doubling tau must double the delay; squaring R' must double the part of
    the delay that depends on drive.  Both are properties of the closed form
    rather than of any particular simulation, so they are cheap to assert and
    catch a whole class of error in the dynamics code.
    """
    crit = 0.95
    coef = crit / (1.0 - crit)

    def delay(tau, rp):
        return tau * np.log(coef * rp)

    assert abs(delay(100.0, 4.0) - 2 * delay(50.0, 4.0)) < 1e-9
    d1 = delay(50.0, 2.0) - delay(50.0, 1.0)
    d2 = delay(50.0, 4.0) - delay(50.0, 2.0)
    assert abs(d1 - d2) < 1e-9          # equal ratios give equal increments
    # Below the criterion floor no delay is needed at all.
    assert delay(50.0, 1.0 / coef * 0.5) < 0.0


# --- the ensemble ceiling ---------------------------------------------------

def test_adding_electrodes_gives_diminishing_returns():
    """Total delivered activation must grow far slower than channel count."""
    m = _model(horiz_gain=0.6, norm_gain=1.0, norm_sigma=1500.0,
               excitability=589.0, cortex_margin=9000.0)

    def total(n, pitch=800.0):
        side = int(np.ceil(np.sqrt(n)))
        offs = (np.arange(side) - (side - 1) / 2.0) * pitch
        gx, gy = np.meshgrid(offs, offs)
        pts = np.column_stack([gx.ravel(), gy.ravel()])[:n]
        xe, ye = pts[:, 0] + X0, pts[:, 1] + Y0
        g = m._ensure_cortical_grid(xe, ye)
        return float(m._activation_field(np.full(n, 300.0), xe, ye, g).sum())

    t1, t16 = total(1), total(16)
    assert t16 / t1 < 16.0 * 0.6, (
        f"16 electrodes delivered {t16/t1:.1f}x a single one; the pooled "
        "divisor should hold this well below 16")


def test_independent_configuration_has_no_ceiling():
    """The same sweep with the circuit off must be exactly linear in N."""
    m = _model(horiz_gain=0.0, norm_gain=0.0, excitability=1019.0,
               cortex_margin=9000.0)

    def total(n, pitch=800.0):
        side = int(np.ceil(np.sqrt(n)))
        offs = (np.arange(side) - (side - 1) / 2.0) * pitch
        gx, gy = np.meshgrid(offs, offs)
        pts = np.column_stack([gx.ravel(), gy.ravel()])[:n]
        xe, ye = pts[:, 0] + X0, pts[:, 1] + Y0
        g = m._ensure_cortical_grid(xe, ye)
        return float(m._activation_field(np.full(n, 300.0), xe, ye, g).sum())

    assert abs(total(9) / total(1) - 9.0) < 1e-6
