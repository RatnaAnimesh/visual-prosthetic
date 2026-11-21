"""Checks on the parts of the model that are easy to get quietly wrong.

These are not tests of whether the cortex is realistic -- that is what the
experiments are for.  They test the things that would silently corrupt an
experiment: sign conventions, circular statistics, receptor kinetics,
delay bookkeeping, and whether the geniculate wiring is actually oriented.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from column import ColumnConfig, CorticalColumn, DriftingGrating, RetinalGrid
from column.analysis import (
    circular_mean_orientation,
    f0_f1,
    orientation_selectivity_index,
    tuning_width_hwhh,
)
from column.connectivity import (
    circular_orientation_difference,
    gabor,
    geniculate_to_simple_cell,
    ocular_dominance_index,
)
from column.lgn import LGN
from column.neurons import BASKET_FS, PYR_RS, NeuronArray
from column.stimuli import Blank, FullFieldFlash
from column.synapses import E_GABA_A, ReceptorBank, ShortTermPlasticity, \
    THALAMOCORTICAL, mg_block


# --- circular statistics ---------------------------------------------------

def test_osi_flat_curve_is_zero():
    o = np.linspace(0, 180, 8, endpoint=False)
    assert orientation_selectivity_index(np.ones(8), o) == pytest.approx(0.0,
                                                                        abs=1e-9)


def test_osi_cosine_curve_is_half():
    o = np.linspace(0, 180, 8, endpoint=False)
    r = 1 + np.cos(2 * np.deg2rad(o - 45.0))
    assert orientation_selectivity_index(r, o) == pytest.approx(0.5, abs=1e-6)


@pytest.mark.parametrize("pref", [0.0, 30.0, 90.0, 175.0])
def test_circular_mean_recovers_preferred_orientation(pref):
    o = np.linspace(0, 180, 12, endpoint=False)
    r = 1 + np.cos(2 * np.deg2rad(o - pref))
    est, _ = circular_mean_orientation(r, o)
    d = abs(est - pref) % 180.0
    assert min(d, 180.0 - d) < 1e-6


def test_orientation_difference_wraps_at_180():
    # 175 and 5 degrees are 10 degrees apart, not 170.
    assert circular_orientation_difference(175.0, 5.0) == pytest.approx(10.0)
    assert circular_orientation_difference(0.0, 90.0) == pytest.approx(90.0)


def test_tuning_width_of_known_curve():
    o = np.linspace(0, 180, 36, endpoint=False)
    # A curve that is above half-height over +-30 degrees around the peak.
    d = circular_orientation_difference(o, 90.0)
    r = np.where(d <= 30.0, 1.0, 0.0)
    assert tuning_width_hwhh(r, o) == pytest.approx(30.0, abs=5.0)


def test_f1_f0_of_a_perfectly_modulated_train():
    # Spikes locked to one phase of a 4 Hz cycle give F1/F0 close to 2.
    freq = 4.0
    t = np.arange(0, 2000, 250.0) + 5.0
    f0, f1 = f0_f1(t, 0.0, 2000.0, freq)
    assert f0 == pytest.approx(len(t) / 2.0, rel=0.01)
    assert f1 / f0 > 1.8


def test_f1_is_zero_for_a_uniform_train():
    freq = 4.0
    t = np.linspace(0, 2000, 400, endpoint=False)
    f0, f1 = f0_f1(t, 0.0, 2000.0, freq)
    assert f0 > 0
    assert f1 / f0 < 0.1


# --- receptors and plasticity ---------------------------------------------

def test_mg_block_is_monotonic_and_bounded():
    V = np.linspace(-90, 40, 50)
    b = mg_block(V)
    assert np.all(np.diff(b) > 0)
    assert b[0] < 0.05 and b[-1] > 0.9


def test_inhibition_reverses_at_its_equilibrium_potential():
    """GABA-A current must vanish exactly at E_Cl and change sign across it."""
    r = ReceptorBank(3)
    r.g_gaba_a[:] = 1.0
    V = np.array([E_GABA_A - 10.0, E_GABA_A, E_GABA_A + 10.0])
    i = r.current(V)
    assert i[1] == pytest.approx(0.0, abs=1e-9)
    assert i[0] > 0 > i[2]


def test_ampa_decays_with_its_time_constant():
    r = ReceptorBank(1)
    r.g_ampa[:] = 1.0
    for _ in range(20):  # 2 ms at dt = 0.1
        r.decay(0.1)
    assert r.g_ampa[0] == pytest.approx(np.exp(-1.0), rel=0.02)


def test_depression_reduces_release_on_repeated_spikes():
    stp = ShortTermPlasticity(1, THALAMOCORTICAL)
    idx = np.array([0])
    first = stp.release(idx, 0.0)[0]
    second = stp.release(idx, 5.0)[0]
    assert second < first
    # And it recovers given enough time.
    recovered = stp.release(idx, 5000.0)[0]
    assert recovered > second


def test_facilitating_class_increases_release():
    from column.synapses import EXC_TO_SST
    stp = ShortTermPlasticity(1, EXC_TO_SST)
    idx = np.array([0])
    first = stp.release(idx, 0.0)[0]
    second = stp.release(idx, 20.0)[0]
    assert second > first


# --- neurons ---------------------------------------------------------------

def test_neuron_is_silent_below_rheobase_and_fires_above():
    rng = np.random.default_rng(0)
    for ct in (PYR_RS, BASKET_FS):
        na = NeuronArray(1, rng)
        na.assign(np.array([0]), ct, 0, jitter=0.0)
        na.initialise_state()
        na.V[:] = ct.E_L
        counts = []
        for I in (0.0, 1000.0):
            na.V[:] = ct.E_L
            na.w[:] = 0.0
            na.ref_until[:] = -1.0
            n, t = 0, 0.0
            for _ in range(4000):
                n += na.step(np.zeros(1), I, t, 0.05).size
                na.clamp()
                t += 0.05
            counts.append(n)
        assert counts[0] == 0
        assert counts[1] > 10


def test_regular_spiking_adapts_and_fast_spiking_does_not():
    rng = np.random.default_rng(0)
    isis = {}
    for ct in (PYR_RS, BASKET_FS):
        na = NeuronArray(1, rng)
        na.assign(np.array([0]), ct, 0, jitter=0.0)
        na.initialise_state()
        na.V[:] = ct.E_L
        na.w[:] = 0.0
        spikes, t = [], 0.0
        for _ in range(10000):
            if na.step(np.zeros(1), 400.0, t, 0.05).size:
                spikes.append(t)
            na.clamp()
            t += 0.05
        isis[ct.name] = np.diff(spikes)
    rs = isis["pyr_rs"]
    fs = isis["basket_fs"]
    assert rs[-1] > 1.5 * rs[0]        # interval lengthens: adaptation
    assert fs[-1] == pytest.approx(fs[0], rel=0.15)  # no adaptation


def test_refractory_period_is_respected():
    rng = np.random.default_rng(0)
    na = NeuronArray(1, rng)
    na.assign(np.array([0]), BASKET_FS, 0, jitter=0.0)
    na.initialise_state()
    na.V[:] = BASKET_FS.E_L
    spikes, t = [], 0.0
    for _ in range(20000):
        if na.step(np.zeros(1), 4000.0, t, 0.05).size:
            spikes.append(t)
        na.clamp()
        t += 0.05
    assert np.min(np.diff(spikes)) >= BASKET_FS.t_ref - 1e-9


# --- geniculate front end --------------------------------------------------

def test_dog_cancels_a_uniform_field():
    """Centre-surround antagonism: a full-field flash must drive the relay
    cells far less than a grating of the same contrast."""
    g = RetinalGrid(2.0, 40)
    lgn = LGN(g, spacing_deg=0.11, seed=1)

    def mean_rate(stim, T=400.0):
        lgn.reset()
        cnt = np.zeros(lgn.n)
        t = 0.0
        for _ in range(int(T / 0.1)):
            f = lgn.step(stim, t, 0.1)
            if t >= 150 and f.size:
                np.add.at(cnt, f, 1)
            t += 0.1
        return (cnt * 1000.0 / (T - 150)).mean()

    grating = mean_rate(DriftingGrating(orientation_deg=0.0, spatial_freq=2.0,
                                        temporal_freq=4.0, contrast=1.0))
    flash = mean_rate(FullFieldFlash(contrast=1.0))
    blank = mean_rate(Blank())
    assert grating > 1.8 * flash
    assert flash > blank


def test_magnocellular_has_higher_contrast_gain():
    g = RetinalGrid(2.0, 40)
    lgn = LGN(g, spacing_deg=0.11, seed=1)
    lgn.reset()
    stim = DriftingGrating(orientation_deg=0.0, spatial_freq=1.0,
                           temporal_freq=4.0, contrast=0.15)
    cnt = np.zeros(lgn.n)
    t = 0.0
    for _ in range(4000):
        f = lgn.step(stim, t, 0.1)
        if t >= 150 and f.size:
            np.add.at(cnt, f, 1)
        t += 0.1
    r = cnt * 1000.0 / 250.0
    m = lgn.stream == "M"
    p = lgn.stream == "P"
    assert r[m].mean() > r[p].mean()


# --- geniculate-to-cortex wiring ------------------------------------------

def test_geniculate_convergence_is_oriented():
    """The Hubel-Wiesel arrangement must actually be elongated: the recruited
    afferents should spread further along the preferred axis than across it."""
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-0.5, 0.5, n)
    sign = rng.choice([-1, 1], n)
    theta = 0.0
    w = geniculate_to_simple_cell(
        x, y, sign,
        dict(x0=0.0, y0=0.0, theta=theta, sf=2.6, phase=0.0,
             sigma_across=0.128, sigma_along=0.128 * 2.4),
        1.0, rng, min_frac=0.05,
    )
    live = w > 0
    th = np.deg2rad(theta)
    across = x[live] * np.sin(th) + y[live] * np.cos(th)
    along = x[live] * np.cos(th) - y[live] * np.sin(th)
    assert along.std() > 1.5 * across.std()


def test_on_and_off_afferents_occupy_opposite_subregions():
    """ON-centre cells must be recruited where the Gabor is positive and
    OFF-centre cells where it is negative."""
    rng = np.random.default_rng(1)
    n = 6000
    x = rng.uniform(-0.4, 0.4, n)
    y = rng.uniform(-0.4, 0.4, n)
    params = dict(x0=0.0, y0=0.0, theta=0.0, sf=2.6, phase=0.0,
                  sigma_across=0.128, sigma_along=0.3)
    g = gabor(x, y, params["x0"], params["y0"], params["theta"],
              params["sf"], params["phase"], params["sigma_across"],
              params["sigma_along"])
    for s in (+1, -1):
        w = geniculate_to_simple_cell(
            x, y, np.full(n, s), params, 1.0, rng, min_frac=0.05)
        live = w > 0
        assert live.any()
        # Every recruited afferent sits where sign * gabor is positive.
        assert np.all(s * g[live] > 0)


def test_ocular_dominance_alternates_with_position():
    x = np.array([0.0, 187.5, 375.0, 562.5, 750.0])
    od = ocular_dominance_index(x, period_um=750.0)
    assert od[0] == pytest.approx(0.0, abs=1e-9)
    assert od[1] > 0.5           # one eye
    assert od[3] < -0.5          # the other, half a cycle later
    assert od[4] == pytest.approx(0.0, abs=1e-6)


# --- network bookkeeping ---------------------------------------------------

def test_column_builds_with_expected_sizes():
    col = CorticalColumn(ColumnConfig(n_columns=4))
    assert col.net.allocated() == col.net.n
    assert col.net.n == 4 * 508
    # Every neuron got assigned a real cell type.
    assert np.all(col.net.neurons.C > 0)
    assert np.all(col.net.neurons.tau_w > 0)
    # Preferred orientations span the full cycle.
    assert col.col_theta.min() == 0.0 and col.col_theta.max() < 180.0


def test_synaptic_delays_are_respected():
    """A spike must not affect its target before its conduction delay."""
    import scipy.sparse as sp
    from column.network import Network
    from column.neurons import PYR_RS
    from column.synapses import EXC_TO_EXC

    net = Network(2, dt=0.1, seed=0)
    a = net.add_population("a", 1, PYR_RS, "II/III")
    b = net.add_population("b", 1, PYR_RS, "II/III")
    W = sp.coo_matrix(([5.0], ([0], [0])), shape=(1, 1))
    net.connect(a, b, EXC_TO_EXC, W, delay_ms=3.0)
    net.compile()
    net.reset()
    net.bg_rate[:] = 0.0
    net.bg_inh_rate[:] = 0.0

    # Force cell a to spike once, then watch cell b's conductance.
    net.neurons.V[a.idx] = 100.0
    trace = []
    for _ in range(60):
        net.step()
        trace.append(net.receptors.g_ampa[b.idx][0])
    trace = np.array(trace)
    first = int(np.argmax(trace > 0))
    # 3 ms delay at dt=0.1 ms, allowing a step of slack for buffer ordering.
    assert 28 <= first <= 32


def test_thalamic_input_is_delivered_and_drives_layer_four():
    col = CorticalColumn(ColumnConfig(n_columns=2))
    col.reset()
    col.present(Blank(), 150.0)
    quiet = col.net.receptors.g_ampa[col.group("L4Cb_ss")].mean()
    col.reset()
    col.present(DriftingGrating(orientation_deg=float(col.col_theta[0]),
                                spatial_freq=2.6, temporal_freq=4.0), 250.0)
    driven = col.net.receptors.g_ampa[col.group("L4Cb_ss")].mean()
    assert driven > quiet


def test_reset_clears_state():
    col = CorticalColumn(ColumnConfig(n_columns=2))
    col.present(DriftingGrating(orientation_deg=0.0, spatial_freq=2.6,
                                temporal_freq=4.0), 100.0)
    assert col.net.spikes.arrays()[0].size >= 0
    col.reset()
    t, i = col.net.spikes.arrays()
    assert t.size == 0 and i.size == 0
    assert col.net.t == 0.0
    assert np.all(col.net.receptors.g_ampa == 0)


def test_normalisation_pool_is_off_by_default_and_wires_when_enabled():
    """The pool must change nothing unless asked for.

    It ships disabled because it trades away more selectivity than it buys
    back in contrast invariance, so the default build has to be identical to
    one with the feature absent.
    """
    base = CorticalColumn(ColumnConfig(n_columns=2))
    n_base = sum(M.nnz for M in base.net.matrices.values())

    pooled = CorticalColumn(ColumnConfig(n_columns=2, norm_p=0.2,
                                         norm_inh_p=0.3))
    n_pooled = sum(M.nnz for M in pooled.net.matrices.values())
    assert n_pooled > n_base

    same = CorticalColumn(ColumnConfig(n_columns=2))
    assert sum(M.nnz for M in same.net.matrices.values()) == n_base


def test_normalisation_pool_is_orientation_blind():
    """The pool's whole purpose is to ignore orientation.

    If it inherited a like-to-like preference it would subtract hardest at the
    preferred orientation and flatten the tuning curve instead of scaling it.
    """
    col = CorticalColumn(ColumnConfig(n_columns=8, norm_p=0.35,
                                      norm_inh_p=0.0))
    src = col.group("L4Cb_ss")
    dst = col.group("L4_pv")

    # Sum every exc_to_pv matrix, not just the first: the class spans several
    # conduction delays, and the within-column projection lives in a different
    # one from the pool.
    rows, cols, vals = [], [], []
    for (cname, _), M in col.net.matrices.items():
        if cname != "exc_to_pv":
            continue
        sub = M.tocsr()[src][:, dst].tocoo()
        rows.append(sub.row)
        cols.append(sub.col)
        vals.append(sub.data)
    row = np.concatenate(rows)
    colx = np.concatenate(cols)
    val = np.concatenate(vals)

    # Only cross-column pairs: the within-column projection is orientation
    # correlated by construction, and is not what the pool is meant to be.
    cross = col.column_of[src][row] != col.column_of[dst][colx]
    row, colx, val = row[cross], colx[cross], val[cross]
    assert val.size > 100

    dtheta = circular_orientation_difference(
        col.theta_pref[src][row], col.theta_pref[dst][colx]
    )
    near = val[dtheta < 30]
    far = val[dtheta > 60]
    assert near.size > 20 and far.size > 20
    # A like-to-like rule would put far more total weight on similar
    # orientations; the pool must spread it roughly evenly.
    assert abs(near.mean() - far.mean()) / max(near.mean(), far.mean()) < 0.25
    assert 0.6 < (near.size / far.size) * (
        (dtheta > 60).sum() / max((dtheta < 30).sum(), 1)) < 1.7


def test_presynaptic_gain_is_inert_by_default():
    col = CorticalColumn(ColumnConfig(n_columns=2))
    assert col.thalamic.gain == 1.0
    col.present(DriftingGrating(orientation_deg=0.0, spatial_freq=2.6,
                                temporal_freq=4.0), 150.0)
    assert col.thalamic.gain == 1.0


def test_presynaptic_gain_falls_with_drive():
    """The normaliser must divide harder when the cortex is driven harder.

    It is a feedback loop, so the check is on the direction of the effect:
    a full-contrast grating has to produce more division than a blank field.
    """
    cfg = ColumnConfig(n_columns=4, presyn_norm_k=0.05, w_thal=22.0)
    cfg.tonic_gain["ss"] = -5.0
    col = CorticalColumn(cfg)

    col.reset()
    col.present(Blank(), 400.0)
    gain_blank = col.thalamic.gain

    col.reset()
    col.present(DriftingGrating(orientation_deg=float(col.col_theta[0]),
                                spatial_freq=2.6, temporal_freq=4.0,
                                contrast=1.0), 400.0)
    gain_driven = col.thalamic.gain

    assert gain_driven < gain_blank
    assert gain_blank == pytest.approx(1.0, abs=0.15)
    assert col.cfg.presyn_norm_floor <= gain_driven <= 1.0


def test_presynaptic_gain_scales_release_not_soma():
    """The gain must act on transmitter release, leaving the target's own
    inhibitory conductance untouched -- that is the whole point of putting it
    presynaptically rather than on the soma."""
    import scipy.sparse as sp
    from column.network import ExternalInput, Network
    from column.neurons import PYR_RS
    from column.synapses import THALAMOCORTICAL

    def delivered(gain):
        net = Network(1, dt=0.1, seed=0)
        pop = net.add_population("t", 1, PYR_RS, "IVCbeta")
        ext = ExternalInput(net, 1, THALAMOCORTICAL)
        W = sp.coo_matrix(([10.0], ([0], [0])), shape=(1, net.n))
        ext.connect(W, 1.0)
        net.compile()
        ext.compile()
        net.reset()
        net.bg_rate[:] = 0.0
        net.bg_inh_rate[:] = 0.0
        net.I_tonic[:] = 0.0
        ext.gain = gain
        ext.deliver(np.array([0]), 0.0)
        for _ in range(3):
            net.step()
        return net.receptors.g_ampa[pop.idx][0], net.receptors.g_gaba_a[pop.idx][0]

    full_e, full_i = delivered(1.0)
    half_e, half_i = delivered(0.5)
    assert half_e == pytest.approx(full_e * 0.5, rel=1e-6)
    assert full_i == 0.0 and half_i == 0.0
