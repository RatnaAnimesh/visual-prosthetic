"""Conductance-based synapses, receptor kinetics, and short-term plasticity.

Kandel ch. 10-13: cortical transmission is glutamatergic (AMPA and NMDA
receptor-channels, reversal near 0 mV) and GABAergic (GABA_A
receptor-channels, Cl- permeable, reversal near the resting potential).  The
synaptic current is therefore *conductance* driven, not current driven:

    I_syn = -g_AMPA (V - E_glu)
            - g_NMDA B(V) (V - E_glu)
            - g_GABA (V - E_Cl)

which matters, because inhibition near E_Cl works largely by shunting -- it
raises the membrane conductance and divides the excitatory drive rather than
simply subtracting from it.  A current-based model cannot show that.

The NMDA channel carries the voltage-dependent Mg2+ block: it passes current
only when glutamate binding *coincides* with depolarisation, which makes it a
coincidence detector and the substrate of recurrent amplification.

Its ratio to AMPA has to be chosen with care.  NMDA decays about fifty times
more slowly than AMPA, so in the steady state it contributes roughly
nmda_ratio * 50 * B(V) times the AMPA conductance -- and that long time
constant low-pass filters the input, passing a stimulus's mean drive in full
while attenuating modulation faster than a few hertz.  For a visual cortex
driven by drifting gratings that is exactly the wrong way round: the
orientation information lives in the modulated component of the geniculate
input, while the orientation-*independent* component lives in the mean.  In
this model, ratios much above 0.3 measurably wash out orientation tuning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- Reversal potentials (mV) ---------------------------------------------
E_GLU = 0.0  # AMPA / NMDA: mixed Na+-K+ conductance, reverses near zero
E_GABA_A = -70.0  # Cl- equilibrium potential in adult cortical neurons
E_GABA_B = -90.0  # K+ equilibrium potential

# --- Receptor kinetics (ms) -----------------------------------------------
TAU_AMPA = 2.0
TAU_NMDA_RISE = 2.0
TAU_NMDA_DECAY = 100.0
TAU_GABA_A = 6.0

MG_CONC = 1.0  # mM


def mg_block(V: np.ndarray) -> np.ndarray:
    """Fraction of NMDA channels relieved of the Mg2+ block at voltage V.

    Jahr & Stevens form.  Near rest (-70 mV) only ~5% of channels conduct; the
    curve passes through half-maximum around -20 mV.
    """
    return 1.0 / (1.0 + np.exp(-0.062 * V) * (MG_CONC / 3.57))


@dataclass(frozen=True)
class SynapseClass:
    """A physiologically distinct family of synapses.

    Short-term plasticity follows the Tsodyks-Markram scheme: each
    presynaptic terminal holds a fraction ``x`` of available vesicles and a
    release probability ``u``.  A spike releases ``u * x``, depletes ``x``, and
    (for facilitating classes) increments ``u``; both relax back with their
    own time constants.  The class-specific values below are the reason the
    same spike train has opposite effects on different targets.
    """

    name: str
    inhibitory: bool = False
    U: float = 0.5  # baseline release probability
    tau_rec: float = 800.0  # vesicle recovery from depression, ms
    tau_fac: float = 0.0  # facilitation decay, ms (0 = no facilitation)
    nmda_ratio: float = 0.0  # g_NMDA peak / g_AMPA peak


# Thalamocortical afferents onto layer IVC.  Strong, reliable, and markedly
# depressing -- which is exactly what makes the cortex respond to *changes* in
# geniculate drive rather than to its absolute level.
#
# The recovery time constant is worth stating explicitly, because a depressing
# synapse driven well above 1/(U*tau_rec) transmits a steady-state conductance
# of about N*w*tau_syn/tau_rec *regardless of presynaptic rate*.  Set tau_rec
# too long and the pathway saturates so early that no visual drive reaches the
# cortex at all; the value here keeps the synapse clearly depressing (about
# three-fold at 40 spikes/s) while leaving the pathway able to drive its
# target.
THALAMOCORTICAL = SynapseClass(
    name="thalamocortical", U=0.40, tau_rec=50.0, nmda_ratio=0.08,
)

# Recurrent excitation between pyramidal / spiny stellate cells: moderately
# depressing, with a substantial NMDA component that supports the sustained
# recurrent amplification of a weak feedforward signal.
EXC_TO_EXC = SynapseClass(
    name="exc_to_exc", U=0.45, tau_rec=300.0, nmda_ratio=0.22,
)

# Excitation onto fast-spiking baskets: depressing, and AMPA dominated.  Fast,
# faithful, and short-lived -- this is the arm of feedforward inhibition.
EXC_TO_PV = SynapseClass(
    name="exc_to_pv", U=0.5, tau_rec=150.0, tau_fac=0.0, nmda_ratio=0.04,
)

# Excitation onto somatostatin+ Martinotti cells: strongly *facilitating*.  A
# single spike barely engages them; a burst recruits them powerfully, so this
# inhibitory arm switches on only when local activity is sustained.
EXC_TO_SST = SynapseClass(
    name="exc_to_sst", U=0.08, tau_rec=200.0, tau_fac=500.0, nmda_ratio=0.18,
)

# GABAergic output.  Perisomatic (basket / chandelier) inhibition is the
# depressing, powerful kind; dendritic (Martinotti) inhibition is weaker per
# spike but sustained.
PV_TO_ANY = SynapseClass(
    name="pv_to_any", inhibitory=True, U=0.35, tau_rec=150.0,
)
SST_TO_ANY = SynapseClass(
    name="sst_to_any", inhibitory=True, U=0.25, tau_rec=200.0, tau_fac=200.0,
)

SYNAPSE_CLASSES = {
    s.name: s
    for s in (
        THALAMOCORTICAL, EXC_TO_EXC, EXC_TO_PV, EXC_TO_SST,
        PV_TO_ANY, SST_TO_ANY,
    )
}


class ShortTermPlasticity:
    """Tsodyks-Markram state for one synapse class, held per presynaptic cell.

    Holding the state per axon rather than per synapse is an approximation --
    real depression is specific to the release site -- but it captures the
    class-specific dynamics at a fraction of the memory cost.
    """

    def __init__(self, n_pre: int, sc: SynapseClass):
        self.sc = sc
        self.x = np.ones(n_pre)  # available resources
        self.u = np.full(n_pre, sc.U)  # release probability
        self.last_spike = np.full(n_pre, -1e9)

    def release(self, pre_idx: np.ndarray, t: float) -> np.ndarray:
        """Return the released fraction for the spiking cells and update state.

        State is advanced lazily -- only for the axons that just fired -- so
        the cost scales with spikes, not with network size.
        """
        if pre_idx.size == 0:
            return np.empty(0)
        sc = self.sc
        dt_since = t - self.last_spike[pre_idx]

        x = 1.0 - (1.0 - self.x[pre_idx]) * np.exp(-dt_since / sc.tau_rec)
        if sc.tau_fac > 0.0:
            u = sc.U + (self.u[pre_idx] - sc.U) * np.exp(-dt_since / sc.tau_fac)
            u = u + sc.U * (1.0 - u)  # facilitation on this spike
        else:
            u = np.full(pre_idx.size, sc.U)

        released = u * x
        self.x[pre_idx] = x - released
        self.u[pre_idx] = u
        self.last_spike[pre_idx] = t
        return released

    def reset(self) -> None:
        self.x[:] = 1.0
        self.u[:] = self.sc.U
        self.last_spike[:] = -1e9


class ReceptorBank:
    """Per-neuron synaptic conductances and the current they drive.

    AMPA and GABA_A are single-exponential decays; NMDA uses a two-state
    rise/decay pair because its slow onset is what lets it integrate over the
    tens of milliseconds that recurrent cortical activity occupies.
    """

    def __init__(self, n: int):
        self.g_ampa = np.zeros(n)
        self.g_nmda_rise = np.zeros(n)
        self.g_nmda = np.zeros(n)
        self.g_gaba_a = np.zeros(n)
        # Normalisation so a unit event gives a unit peak conductance.
        tr, td = TAU_NMDA_RISE, TAU_NMDA_DECAY
        t_peak = (tr * td) / (td - tr) * np.log(td / tr)
        self._nmda_norm = 1.0 / (np.exp(-t_peak / td) - np.exp(-t_peak / tr))

    def decay(self, dt: float) -> None:
        self.g_ampa *= np.exp(-dt / TAU_AMPA)
        self.g_gaba_a *= np.exp(-dt / TAU_GABA_A)
        self.g_nmda *= np.exp(-dt / TAU_NMDA_DECAY)
        self.g_nmda_rise *= np.exp(-dt / TAU_NMDA_RISE)

    def current(self, V: np.ndarray) -> np.ndarray:
        """Total synaptic current in pA (positive = depolarising)."""
        g_nmda_open = (self.g_nmda - self.g_nmda_rise) * self._nmda_norm
        i_glu = (self.g_ampa + g_nmda_open * mg_block(V)) * (E_GLU - V)
        i_gaba = self.g_gaba_a * (E_GABA_A - V)
        return i_glu + i_gaba

    def conductance(self) -> np.ndarray:
        """Total synaptic conductance -- the substrate of shunting inhibition."""
        return self.g_ampa + self.g_gaba_a

    def reset(self) -> None:
        self.g_ampa[:] = 0.0
        self.g_nmda[:] = 0.0
        self.g_nmda_rise[:] = 0.0
        self.g_gaba_a[:] = 0.0
