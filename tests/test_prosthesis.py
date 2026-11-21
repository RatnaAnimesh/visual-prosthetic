"""Checks on the cortical circuit model.

The important one is ``test_linear_mechanisms_cannot_break_independence``.
That result is counterintuitive enough that it should be pinned down: it is
the reason the project's original hypothesis (horizontal connections, by
analogy with retinal axon bundles) was wrong, and if a future change to the
horizontal machinery ever appears to produce electrode interaction, it is a
bug rather than a discovery.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pulse2percept.implants.cortex import Cortivis

from prosthesis.cortex_circuit import CircuitSpatial, orientation_map

XR, YR, STEP = (-3.2, -0.4), (-0.2, 2.0), 0.05


def _model(**kw):
    m = CircuitSpatial(xrange=XR, yrange=YR, xystep=STEP, **kw)
    m.build()
    return m


def _pair(implant, target_um):
    ref = implant.electrode_names[0]
    x0, y0 = implant[ref].x, implant[ref].y
    cand = sorted(((np.hypot(implant[n].x - x0, implant[n].y - y0), n)
                   for n in implant.electrode_names if n != ref))
    d, other = min(cand, key=lambda s: abs(s[0] - target_um))
    return (np.array([x0, implant[other].x], float),
            np.array([y0, implant[other].y], float), d)


def _deviation(model, xe, ye, amp=300.0):
    g = model._ensure_cortical_grid(xe, ye)
    fa = model._activation_field(np.array([amp, 0.0]), xe, ye, g)
    fb = model._activation_field(np.array([0.0, amp]), xe, ye, g)
    fab = model._activation_field(np.array([amp, amp]), xe, ye, g)
    lin = fa + fb
    return float(np.abs(fab - lin).max() / max(np.abs(lin).max(), 1e-12))


# --- orientation map -------------------------------------------------------

def test_orientation_map_covers_full_cycle():
    t = orientation_map((120, 120), 50.0, 750.0, seed=0)
    assert t.min() >= 0 and t.max() < 180
    # All orientations should be represented, not clustered.
    hist, _ = np.histogram(t, bins=12, range=(0, 180))
    assert hist.min() > 0.4 * hist.mean()


def test_orientation_map_has_the_right_column_spacing():
    """Iso-orientation domains should repeat near the specified spacing."""
    step, spacing = 25.0, 750.0
    t = orientation_map((256, 256), step, spacing, seed=1)
    z = np.exp(2j * np.deg2rad(t))
    power = np.abs(np.fft.fftshift(np.fft.fft2(z))) ** 2
    ny, nx = power.shape
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=step))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=step))[None, :]
    f = np.sqrt(fx ** 2 + fy ** 2)
    peak_f = float(f.ravel()[int(np.argmax(power))])
    assert peak_f == pytest.approx(1.0 / spacing, rel=0.35)


# --- the central results ---------------------------------------------------

def test_circuit_off_is_exactly_linear():
    implant = Cortivis()
    xe, ye, _ = _pair(implant, 1200)
    m = _model(horiz_gain=0.0, norm_gain=0.0)
    assert _deviation(m, xe, ye) < 1e-9


def test_linear_mechanisms_cannot_break_independence():
    """Horizontal spread is three Gaussian blurs -- a linear operator.

    However orientation-selective it is, L(A + B) = L(A) + L(B).  This was the
    project's original hypothesis and it is provably incapable of producing
    the effect it was hypothesised to produce.
    """
    implant = Cortivis()
    for target in (400, 1200, 2400):
        xe, ye, _ = _pair(implant, target)
        for gain in (0.3, 0.6, 1.2):
            m = _model(horiz_gain=gain, norm_gain=0.0)
            assert _deviation(m, xe, ye) < 1e-9, (
                f"horizontal spread produced interaction at {target} um, "
                f"gain {gain} -- it is linear and cannot"
            )


def test_normalisation_does_break_independence():
    implant = Cortivis()
    xe, ye, _ = _pair(implant, 1200)
    m = _model(horiz_gain=0.0, norm_gain=1.0)
    assert _deviation(m, xe, ye) > 0.05


def test_interaction_is_suppressive():
    """Normalisation must make the pair dimmer than the sum, never brighter."""
    implant = Cortivis()
    xe, ye, _ = _pair(implant, 1200)
    m = _model(horiz_gain=0.6, norm_gain=1.0)
    g = m._ensure_cortical_grid(xe, ye)
    fa = m._activation_field(np.array([300.0, 0.0]), xe, ye, g)
    fb = m._activation_field(np.array([0.0, 300.0]), xe, ye, g)
    fab = m._activation_field(np.array([300.0, 300.0]), xe, ye, g)
    lin = fa + fb
    i = np.unravel_index(int(np.argmax(lin)), lin.shape)
    assert fab[i] < lin[i]


def test_interaction_decays_with_separation():
    """Distant electrodes share no pool, so they must stop interacting."""
    implant = Cortivis()
    m = _model(horiz_gain=0.6, norm_gain=1.0)
    near = _deviation(m, *_pair(implant, 1600)[:2])
    far = _deviation(m, *_pair(implant, 4500)[:2])
    assert near > 5 * far
    assert far < 0.05


# --- model plumbing --------------------------------------------------------

def test_phosphene_grows_with_current():
    implant = Cortivis()
    m = _model(horiz_gain=0.6, norm_gain=1.0)
    ref = implant.electrode_names[0]
    xe = np.array([implant[ref].x], float)
    ye = np.array([implant[ref].y], float)
    g = m._ensure_cortical_grid(xe, ye)
    areas = []
    for amp in (100.0, 300.0, 600.0):
        f = m._activation_field(np.array([amp]), xe, ye, g)
        areas.append(int((f >= f.max() / 2).sum()))
    assert areas[0] < areas[1] < areas[2]


def test_predict_percept_runs_through_pulse2percept():
    """The model must be usable through the standard API, not just internally."""
    from pulse2percept.stimuli import BiphasicPulseTrain
    implant = Cortivis()
    m = _model(horiz_gain=0.6, norm_gain=1.0)
    implant.stim = {implant.electrode_names[0]:
                    BiphasicPulseTrain(300, 300.0, 0.17, stim_dur=20.0)}
    p = m.predict_percept(implant)
    assert p.data.ndim == 3
    assert np.isfinite(p.data).all()
    assert p.data.max() > 0


# --- percept decoder -------------------------------------------------------

def _code():
    from prosthesis.percept_decoder import CorticalCode
    return CorticalCode((17000, 23000), (-8000, -2000), step_um=200.0)


def test_encoder_and_inverse_share_one_operator():
    """The encoder must be defined through the same matrix the inverse solves.

    Maintaining two implementations is how the convolution/correlation flip
    got in: a convolution flips its kernel, a dot product does not, and Gabor
    filters are not symmetric.  Pinning the identity prevents a silent
    regression.
    """
    code = _code()
    rng = np.random.default_rng(0)
    img = rng.normal(size=(code.n_px, code.n_px))
    direct = np.maximum(code.M @ img.ravel(), 0.0)
    assert np.allclose(code.encode(img)[0], direct, atol=1e-12)


def test_inverse_recovers_held_out_images():
    from prosthesis.percept_decoder import (InversePercept, pink_noise,
                                            reconstruction_r2)
    code = _code()
    inv = InversePercept(code, lam=0.02)
    rng = np.random.default_rng(5)
    test = pink_noise(6, code.n_px, rng)
    assert reconstruction_r2(test, inv.round_trip(test)) > 0.3


def test_inverse_is_linear_in_activity():
    """The readout must not itself introduce electrode interaction.

    If it did, a departure from independence measured after decoding could
    come from the decoder rather than from the cortex, and Phase 1's result
    would be uninterpretable.
    """
    from prosthesis.percept_decoder import InversePercept
    code = _code()
    inv = InversePercept(code, lam=0.05, iters=1)
    rng = np.random.default_rng(2)
    a = np.abs(rng.normal(size=code.n_sites))
    b = np.abs(rng.normal(size=code.n_sites))
    lhs = inv.decode(a + b)[0]
    rhs = inv.decode(a)[0] + inv.decode(b)[0]
    assert np.allclose(lhs, rhs, atol=1e-8)


def test_sites_carry_orientation_and_phase():
    code = _code()
    assert code.n_sites > 100
    assert code.theta.min() >= 0 and code.theta.max() < 180
    assert code.phase.min() >= 0 and code.phase.max() < 2 * np.pi
    # Phase must be scattered, not constant -- an orientation-only code
    # discards the sign of local contrast and cannot be inverted.
    assert code.phase.std() > 1.0


# --- Phase 4 metrics -------------------------------------------------------

def test_correlation_fidelity_is_blind_to_brightness():
    """The trap Phase 4 turns on.

    Squared correlation scores a dim percept identically to a bright one.
    That is why brightness has to be reported separately, and pinning it here
    stops anyone later treating the correlation number as a completeness
    check.
    """
    from prosthesis.phase4_encoder import contrast, fidelity
    rng = np.random.default_rng(0)
    target = rng.normal(size=(16, 16))
    dim = 0.25 * target
    assert fidelity(dim, target) == pytest.approx(fidelity(target, target),
                                                  abs=1e-9)
    assert contrast(dim) == pytest.approx(0.25 * contrast(target), rel=1e-9)


def test_linear_reference_separates_deliverable_from_undeliverable():
    """The diagnostic that caught the undeliverable targets.

    It is deliberately *not* asserted to be an upper bound: the model is
    nonlinear in amplitude, so optimisation can exceed it slightly.  What it
    must do is separate a target the electrodes can produce from one they
    cannot.
    """
    from prosthesis.phase4_encoder import (Pipeline, configs,
                                           linear_reference, pick_electrodes)
    from prosthesis.percept_decoder import CorticalCode, InversePercept
    implant = Cortivis()
    n_el = 8
    names = pick_electrodes(implant, n_el)
    xe = np.array([implant[k].x for k in names], float)
    ye = np.array([implant[k].y for k in names], float)
    code = CorticalCode((16200, 23800), (-8800, -1200), step_um=200.0)
    dec = InversePercept(code, lam=0.02, precompute=True)
    pipe = Pipeline(configs()["independent"], code, dec, xe, ye)

    rng = np.random.default_rng(1)
    amps = np.zeros(n_el)
    amps[rng.choice(n_el, 3, replace=False)] = 450.0
    target = pipe.percept(amps)

    deliverable = linear_reference(pipe, target, n_el)
    # A thin geometric bar is the kind of thing a handful of phosphenes
    # cannot draw.
    bar = np.zeros((code.n_px, code.n_px))
    bar[code.n_px // 2, 2:-2] = 1.0
    undeliverable = linear_reference(pipe, bar, n_el)

    assert deliverable > 0.9
    assert undeliverable < 0.5
    assert deliverable > 2 * undeliverable


def test_far_pool_is_off_by_default_and_broadens_suppression():
    """The second, feedback-scale normalisation pool.

    It exists to test whether long-range phosphene suppression needs it, not
    as part of the baseline claim, so it must be inert unless requested.
    """
    implant = Cortivis()
    xe, ye, _ = _pair(implant, 2400)
    base = _model(horiz_gain=0.6, norm_gain=1.0)
    far = _model(horiz_gain=0.6, norm_gain=1.0, norm_far_gain=3.0,
                 norm_far_sigma=8000.0)
    assert base.norm_far_gain == 0.0
    # Enabling it must increase interaction at a long separation.
    assert _deviation(far, xe, ye, amp=2000.0) > _deviation(base, xe, ye,
                                                            amp=2000.0)


def test_pool_width_effect_is_not_monotonic():
    """Widening a divisive pool does not simply extend its reach.

    A first version of this test asserted that broadening dilutes the pool and
    weakens it everywhere.  That is false and the test caught it: at short
    separations a broader pool *increases* the departure from superposition
    (20% to 90% going from 1.5 mm to 12 mm at 800 um), while at longer
    separations it decreases it.  The paper's claim was corrected to match.
    """
    implant = Cortivis()
    near_xe, near_ye, _ = _pair(implant, 800)
    far_xe, far_ye, _ = _pair(implant, 2400)

    def dev(sigma, xe, ye):
        m = _model(horiz_gain=0.0, norm_gain=0.0, norm_far_gain=2.0,
                   norm_far_sigma=sigma)
        return _deviation(m, xe, ye)

    assert dev(12000.0, near_xe, near_ye) > dev(1500.0, near_xe, near_ye)
    assert dev(12000.0, far_xe, far_ye) < dev(1500.0, far_xe, far_ye)


def test_physiological_far_pool_cannot_reach_13mm():
    """The §3.5 result: an ordered far pool leaves long-range suppression weak.

    'Ordered' means far surround suppression weaker than near, as reported in
    V1.  Reproducing the ~13.5 mm single-phosphene suppression requires
    inverting that, which is why the paper declines to fit it.
    """
    m = _model(horiz_gain=0.6, norm_gain=1.0, norm_far_gain=0.5,
               norm_far_sigma=8000.0, cortex_margin=20000.0)
    xe = np.array([20000.0, 20000.0 + 13470.0])
    ye = np.array([-5000.0, -5000.0])
    assert _deviation(m, xe, ye, amp=4000.0) < 0.10


def test_later_phases_share_phase1s_calibration():
    """Phases 2, 4 and 5 once typed the fitted constant in by hand, and three
    of them fell out of step with the fit.  They now read it from Phase 1."""
    import json
    from prosthesis.phase1_circuit import calibrated
    from prosthesis.phase2_percepts import configs as p2
    from prosthesis.phase4_encoder import configs as p4
    cal = json.loads((pathlib.Path(__file__).resolve().parents[1] / "results"
                      / "phase1_circuit.json").read_text())["calibration"]
    for label, k in (("circuit off", "independent"),
                     ("full circuit", "circuit")):
        want = cal[label]["excitability"]
        assert calibrated(label)["excitability"] == want
        assert p4()[k]["excitability"] == want
    assert p2()["circuit"]["excitability"] == cal["full circuit"]["excitability"]
