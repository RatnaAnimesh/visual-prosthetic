"""Point neurons for the cortical column.

Every cell in the network is an adaptive exponential integrate-and-fire (AdEx)
unit.  The AdEx model is the smallest one that still reproduces the firing
signatures Kandel describes for the distinct cortical cell classes -- regular
spiking pyramidal cells with strong spike-frequency adaptation, intrinsically
bursting layer V pyramids, and non-adapting fast-spiking GABAergic
interneurons:

    C dV/dt = -g_L (V - E_L) + g_L D_T exp((V - V_T)/D_T) - w + I
    tau_w dw/dt = a (V - E_L) - w
    if V > V_peak:  V <- V_reset,  w <- w + b

The exponential term is the regenerative Na+ current of Kandel ch. 7 in
reduced form: it is negligible at rest and diverges once V passes V_T, which
is what makes threshold an emergent property rather than a hard-coded
comparison.  ``w`` lumps together the slow K+ currents (I_M, I_AHP) that give
spike-frequency adaptation.

Units are physiological throughout: mV, ms, nS, pA, pF.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class CellType:
    """Intrinsic parameters of one electrophysiological cell class."""

    name: str
    C: float  # membrane capacitance, pF
    g_L: float  # leak conductance, nS
    E_L: float  # leak reversal / resting potential, mV
    V_T: float  # spike-initiation threshold of the exponential term, mV
    Delta_T: float  # sharpness of spike initiation, mV
    a: float  # subthreshold adaptation coupling, nS
    tau_w: float  # adaptation time constant, ms
    b: float  # spike-triggered adaptation increment, pA
    V_reset: float  # reset potential, mV
    t_ref: float  # absolute refractory period, ms
    V_peak: float = 0.0  # spike cutoff, mV
    excitatory: bool = True

    @property
    def tau_m(self) -> float:
        """Passive membrane time constant, ms (Kandel ch. 6)."""
        return self.C / self.g_L

    def scaled(self, **kw) -> "CellType":
        return replace(self, **kw)


# --- Excitatory classes ----------------------------------------------------

# Layer II/III and VI pyramidal cells: regular spiking with pronounced
# spike-frequency adaptation.
PYR_RS = CellType(
    name="pyr_rs", C=200.0, g_L=10.0, E_L=-70.0, V_T=-50.0, Delta_T=2.0,
    a=2.0, tau_w=120.0, b=60.0, V_reset=-58.0, t_ref=2.0,
)

# Layer IVC spiny stellate cells: the excitatory recipients of thalamic input.
# Smaller, more compact, and less adapting than pyramids -- they act as a
# faithful relay of the geniculate drive.
SPINY_STELLATE = CellType(
    name="spiny_stellate", C=150.0, g_L=10.0, E_L=-70.0, V_T=-50.0, Delta_T=2.0,
    a=1.0, tau_w=100.0, b=25.0, V_reset=-60.0, t_ref=2.0,
)

# Layer V intrinsically bursting pyramids -- the cells projecting to superior
# colliculus and pons.  A depolarised reset drives the 2-4 spike bursts.
PYR_IB = CellType(
    name="pyr_ib", C=200.0, g_L=10.0, E_L=-70.0, V_T=-50.0, Delta_T=2.0,
    a=2.0, tau_w=130.0, b=25.0, V_reset=-46.0, t_ref=2.0,
)

# Layer VI corticothalamic pyramids: slow, sparse firing.
PYR_CT = PYR_RS.scaled(name="pyr_ct", b=90.0, tau_w=150.0)

# --- Inhibitory classes ----------------------------------------------------

# Parvalbumin+ basket cells: fast-spiking, non-adapting, perisomatic targeting.
# Short membrane time constant and sharp spike initiation let them follow
# high-frequency input, which is what makes feedforward inhibition fast enough
# to sculpt the cortical response window.
BASKET_FS = CellType(
    name="basket_fs", C=100.0, g_L=10.0, E_L=-65.0, V_T=-48.0, Delta_T=0.5,
    a=0.0, tau_w=30.0, b=0.0, V_reset=-65.0, t_ref=1.0, excitatory=False,
)

# Chandelier (axo-axonic) cells: fast-spiking like baskets but targeting the
# axon initial segment, so they veto spike initiation itself.
CHANDELIER = BASKET_FS.scaled(name="chandelier", b=5.0)

# Somatostatin+ Martinotti cells: low-threshold spiking, adapting, and they
# target distal apical dendrites in layer I.
MARTINOTTI_LTS = CellType(
    name="martinotti_lts", C=100.0, g_L=5.0, E_L=-65.0, V_T=-52.0, Delta_T=2.0,
    a=4.0, tau_w=150.0, b=20.0, V_reset=-60.0, t_ref=2.0, excitatory=False,
)

CELL_TYPES = {
    ct.name: ct
    for ct in (
        PYR_RS, SPINY_STELLATE, PYR_IB, PYR_CT,
        BASKET_FS, CHANDELIER, MARTINOTTI_LTS,
    )
}


class NeuronArray:
    """All neurons of the network as flat, vectorised parameter arrays.

    Populations are contiguous index ranges into these arrays, so one
    integration step updates the entire cortex with a handful of numpy
    operations.
    """

    def __init__(self, n: int, rng: np.random.Generator):
        self.n = n
        self.rng = rng
        z = np.zeros(n)
        self.C = z.copy()
        self.g_L = z.copy()
        self.E_L = z.copy()
        self.V_T = z.copy()
        self.Delta_T = z.copy()
        self.a = z.copy()
        self.tau_w = np.ones(n)
        self.b = z.copy()
        self.V_reset = z.copy()
        self.t_ref = z.copy()
        self.V_peak = z.copy()
        self.excitatory = np.zeros(n, dtype=bool)
        self.type_id = np.zeros(n, dtype=np.int16)

        # State
        self.V = z.copy()
        self.w = z.copy()
        self.ref_until = np.full(n, -1.0)

    def assign(self, idx: np.ndarray, ct: CellType, type_id: int,
               jitter: float = 0.08) -> None:
        """Give ``idx`` the parameters of cell type ``ct``.

        Real cortical neurons of one class are not identical.  A modest
        lognormal spread of the intrinsic parameters keeps the population from
        firing as one synchronised block -- heterogeneity is itself a
        computational feature, not noise to be removed.
        """
        k = len(idx)
        jit = lambda: self.rng.lognormal(0.0, jitter, k) if jitter > 0 else 1.0

        self.C[idx] = ct.C * jit()
        self.g_L[idx] = ct.g_L * jit()
        self.E_L[idx] = ct.E_L + self.rng.normal(0.0, 1.0, k)
        self.V_T[idx] = ct.V_T + self.rng.normal(0.0, 1.0, k)
        self.Delta_T[idx] = ct.Delta_T
        self.a[idx] = ct.a
        self.tau_w[idx] = ct.tau_w * jit()
        self.b[idx] = ct.b * jit()
        self.V_reset[idx] = ct.V_reset
        self.t_ref[idx] = ct.t_ref
        self.V_peak[idx] = ct.V_peak
        self.excitatory[idx] = ct.excitatory
        self.type_id[idx] = type_id

    def initialise_state(self) -> None:
        """Start each cell at a random point around rest, not all identical."""
        self.V[:] = self.E_L + self.rng.normal(0.0, 3.0, self.n)
        self.w[:] = 0.0
        self.ref_until[:] = -1.0

    def step(self, I_syn: np.ndarray, I_ext: np.ndarray, t: float,
             dt: float) -> np.ndarray:
        """Advance one timestep; return indices of neurons that spiked.

        ``I_syn`` and ``I_ext`` are in pA.  Integration is forward Euler, which
        is adequate at dt <= 0.1 ms provided the exponential term is clipped
        before it overflows.
        """
        free = t >= self.ref_until

        exp_arg = np.clip((self.V - self.V_T) / self.Delta_T, -50.0, 20.0)
        I_exp = self.g_L * self.Delta_T * np.exp(exp_arg)
        I_leak = -self.g_L * (self.V - self.E_L)

        # Both derivatives are evaluated at the *same* old state, so the
        # diverging V of a spike cannot contaminate the adaptation variable.
        dV = (I_leak + I_exp - self.w + I_syn + I_ext) / self.C
        dw = (self.a * (self.V - self.E_L) - self.w) / self.tau_w

        self.V += np.where(free, dV * dt, 0.0)
        self.w += dw * dt

        # Runaway of the exponential term *is* the spike.
        spiked = np.flatnonzero(self.V >= self.V_peak)
        if spiked.size:
            self.V[spiked] = self.V_reset[spiked]
            self.w[spiked] += self.b[spiked]
            self.ref_until[spiked] = t + self.t_ref[spiked]
        return spiked

    def clamp(self, lo: float = -90.0) -> None:
        """Guard against numerical excursions below any physiological floor."""
        np.clip(self.V, lo, None, out=self.V)
