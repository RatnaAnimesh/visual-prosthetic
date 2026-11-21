"""The simulation engine: projections, axonal delays, and the update loop.

A ``Projection`` is a sparse set of synapses from one population to another,
carrying a synapse class and an axonal conduction delay.  Projections that
share a class and a delay are fused into a single sparse matrix so that one
timestep costs a handful of sparse operations regardless of how many
anatomical pathways the model contains.

Delivery uses a ring buffer indexed by delay: a spike at step k with delay d
deposits its conductance into slot (k + d) mod D, which is read out and
cleared when the clock reaches it.  Conduction delay is not a detail --
Kandel's intrinsic circuitry (Fig. 25-15) is a loop, and a loop without delay
is an algebraic constraint rather than a dynamical system.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .neurons import NeuronArray
from .synapses import ReceptorBank, ShortTermPlasticity, SynapseClass


@dataclass
class Population:
    """A named, contiguous block of neurons."""

    name: str
    start: int
    size: int
    cell_type: str
    layer: str
    column: int = -1

    @property
    def idx(self) -> np.ndarray:
        return np.arange(self.start, self.start + self.size)

    @property
    def stop(self) -> int:
        return self.start + self.size

    def __len__(self) -> int:
        return self.size


@dataclass
class SpikeRecorder:
    """Collects spike times as a growing (time, neuron) event list."""

    times: list = field(default_factory=list)
    ids: list = field(default_factory=list)

    def record(self, t: float, idx: np.ndarray) -> None:
        if idx.size:
            self.times.append(np.full(idx.size, t))
            self.ids.append(idx)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.times:
            return np.empty(0), np.empty(0, dtype=int)
        return np.concatenate(self.times), np.concatenate(self.ids)

    def clear(self) -> None:
        self.times.clear()
        self.ids.clear()

    def counts(self, n: int, t0: float = -np.inf, t1: float = np.inf
               ) -> np.ndarray:
        """Spike count per neuron within a time window."""
        t, i = self.arrays()
        if t.size == 0:
            return np.zeros(n)
        m = (t >= t0) & (t < t1)
        return np.bincount(i[m], minlength=n).astype(float)

    def rates(self, n: int, t0: float, t1: float) -> np.ndarray:
        """Firing rate per neuron in spikes/s over [t0, t1) given in ms."""
        return self.counts(n, t0, t1) * 1000.0 / max(t1 - t0, 1e-9)


class Network:
    """Holds the neurons, the wiring, and the clock."""

    def __init__(self, n: int, dt: float = 0.1, seed: int = 0):
        self.dt = dt
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.neurons = NeuronArray(n, self.rng)
        self.receptors = ReceptorBank(n)
        self.populations: dict[str, Population] = {}
        self._pending: list[tuple] = []  # (pre, post, W, class, delay_ms)
        self._compiled = False
        self.spikes = SpikeRecorder()
        self.t = 0.0
        self.step_count = 0

        # Background drive standing in for the rest of cortex.  Its mean and
        # its variance are supplied separately: a tonic current sets where the
        # cell sits relative to threshold, while sparse, large conductance
        # events supply the membrane-potential fluctuations.  Folding both into
        # one Poisson conductance would force a large mean conductance in order
        # to get any depolarisation, and that shunt would swamp the visually
        # driven input the column is supposed to be responding to.
        self.bg_rate = np.zeros(n)  # spikes/s of Poisson bombardment
        self.bg_weight = np.zeros(n)  # nS per background event
        self.bg_inh_rate = np.zeros(n)
        self.bg_inh_weight = np.zeros(n)
        self.I_tonic = np.zeros(n)  # pA

        self._trace_idx: np.ndarray | None = None
        self.traces: dict[str, list] = {}
        self._reserved_delay_ms = 0.0

    # -- construction -------------------------------------------------------

    def add_population(self, name: str, size: int, cell_type, layer: str,
                       column: int = -1) -> Population:
        from .neurons import CELL_TYPES

        start = sum(len(p) for p in self.populations.values())
        if start + size > self.n:
            raise ValueError(
                f"population '{name}' exceeds the {self.n} allocated neurons"
            )
        ct = cell_type if hasattr(cell_type, "name") else CELL_TYPES[cell_type]
        pop = Population(name, start, size, ct.name, layer, column)
        self.populations[name] = pop
        type_id = len(
            {p.cell_type for p in self.populations.values()}
        )
        self.neurons.assign(pop.idx, ct, type_id)
        return pop

    def allocated(self) -> int:
        return sum(len(p) for p in self.populations.values())

    def connect(self, pre: Population, post: Population, sc: SynapseClass,
                W: sp.spmatrix, delay_ms: float) -> None:
        """Register synapses; ``W`` is (len(pre), len(post)) in nS."""
        if W.nnz == 0:
            return
        self._pending.append((pre, post, W.tocoo(), sc, delay_ms))
        self._compiled = False

    def compile(self) -> None:
        """Fuse pending projections into per-(class, delay) sparse matrices."""
        groups: dict[tuple[str, int], list] = defaultdict(list)
        self.syn_classes: dict[str, SynapseClass] = {}
        max_delay_steps = max(
            1, int(round(self._reserved_delay_ms / self.dt))
        )

        for pre, post, W, sc, delay_ms in self._pending:
            d = max(1, int(round(delay_ms / self.dt)))
            max_delay_steps = max(max_delay_steps, d)
            groups[(sc.name, d)].append((pre, post, W))
            self.syn_classes[sc.name] = sc

        self.matrices: dict[tuple[str, int], sp.csr_matrix] = {}
        for key, items in groups.items():
            rows, cols, vals = [], [], []
            for pre, post, W in items:
                rows.append(W.row + pre.start)
                cols.append(W.col + post.start)
                vals.append(W.data)
            M = sp.coo_matrix(
                (np.concatenate(vals),
                 (np.concatenate(rows), np.concatenate(cols))),
                shape=(self.n, self.n),
            ).tocsr()
            M.sum_duplicates()
            self.matrices[key] = M

        self.stp = {
            name: ShortTermPlasticity(self.n, sc)
            for name, sc in self.syn_classes.items()
        }
        # Which presynaptic neurons belong to which synapse class.
        self.class_members: dict[str, np.ndarray] = {}
        for name in self.syn_classes:
            mask = np.zeros(self.n, dtype=bool)
            for (cname, _), M in self.matrices.items():
                if cname == name:
                    mask |= np.diff(M.indptr) > 0
            self.class_members[name] = mask

        self.D = max_delay_steps + 1
        self.buf_ampa = np.zeros((self.D, self.n))
        self.buf_nmda = np.zeros((self.D, self.n))
        self.buf_gaba = np.zeros((self.D, self.n))
        self._compiled = True
        self._pending_kept = self._pending
        self._pending = []

    # -- runtime ------------------------------------------------------------

    def reserve_delay(self, delay_ms: float) -> None:
        """Guarantee the delay buffer is long enough for an external source."""
        self._reserved_delay_ms = max(self._reserved_delay_ms, delay_ms)

    def push_conductance(self, contrib: np.ndarray, delay_steps: int,
                         inhibitory: bool = False,
                         nmda_ratio: float = 0.0) -> None:
        """Deposit conductance into the delay buffer ``delay_steps`` ahead.

        Used by projections whose presynaptic cells live outside the
        NeuronArray -- notably the geniculate afferents.
        """
        slot = (self.step_count + delay_steps) % self.D
        if inhibitory:
            self.buf_gaba[slot] += contrib
        else:
            self.buf_ampa[slot] += contrib
            if nmda_ratio:
                self.buf_nmda[slot] += contrib * nmda_ratio

    def set_background(self, pop: Population, exc_rate: float,
                       exc_weight: float, inh_rate: float = 0.0,
                       inh_weight: float = 0.0, tonic: float = 0.0) -> None:
        """Poisson bombardment representing afferents outside the model.

        Cortical neurons in vivo sit in a high-conductance state driven by
        thousands of weak, largely uncorrelated synapses.  Without it the model
        is far quieter and far more deterministic than real cortex.
        """
        self.bg_rate[pop.idx] = exc_rate
        self.bg_weight[pop.idx] = exc_weight
        self.bg_inh_rate[pop.idx] = inh_rate
        self.bg_inh_weight[pop.idx] = inh_weight
        self.I_tonic[pop.idx] = tonic

    def record_traces(self, idx: np.ndarray) -> None:
        self._trace_idx = np.asarray(idx)
        self.traces = {"t": [], "V": [], "g_e": [], "g_i": []}

    def reset(self) -> None:
        self.neurons.initialise_state()
        self.receptors.reset()
        for s in self.stp.values():
            s.reset()
        self.buf_ampa[:] = 0.0
        self.buf_nmda[:] = 0.0
        self.buf_gaba[:] = 0.0
        self.spikes.clear()
        self.t = 0.0
        self.step_count = 0
        if self._trace_idx is not None:
            self.traces = {"t": [], "V": [], "g_e": [], "g_i": []}

    def inject_spikes(self, idx: np.ndarray, weight: np.ndarray | float,
                      receptor: str = "ampa", nmda_ratio: float = 0.0) -> None:
        """Deliver externally generated spikes (e.g. from the LGN) at once."""
        if idx.size == 0:
            return
        if receptor == "ampa":
            np.add.at(self.receptors.g_ampa, idx, weight)
            if nmda_ratio:
                np.add.at(self.receptors.g_nmda, idx, weight * nmda_ratio)
                np.add.at(self.receptors.g_nmda_rise, idx, weight * nmda_ratio)
        else:
            np.add.at(self.receptors.g_gaba_a, idx, weight)

    def step(self, I_ext: np.ndarray | float = 0.0) -> np.ndarray:
        """Advance the whole network by one dt; return the spiking indices."""
        if not self._compiled:
            self.compile()

        slot = self.step_count % self.D
        self.receptors.g_ampa += self.buf_ampa[slot]
        self.receptors.g_nmda += self.buf_nmda[slot]
        self.receptors.g_nmda_rise += self.buf_nmda[slot]
        self.receptors.g_gaba_a += self.buf_gaba[slot]
        self.buf_ampa[slot] = 0.0
        self.buf_nmda[slot] = 0.0
        self.buf_gaba[slot] = 0.0

        # Background bombardment: Poisson counts folded straight into g.
        if self.bg_rate.any():
            lam = self.bg_rate * self.dt / 1000.0
            self.receptors.g_ampa += self.rng.poisson(lam) * self.bg_weight
        if self.bg_inh_rate.any():
            lam = self.bg_inh_rate * self.dt / 1000.0
            self.receptors.g_gaba_a += (
                self.rng.poisson(lam) * self.bg_inh_weight
            )

        I_syn = self.receptors.current(self.neurons.V)
        fired = self.neurons.step(I_syn, I_ext + self.I_tonic, self.t, self.dt)
        self.neurons.clamp()
        self.spikes.record(self.t, fired)

        if fired.size:
            self._deliver(fired)

        self.receptors.decay(self.dt)

        if self._trace_idx is not None:
            k = self._trace_idx
            self.traces["t"].append(self.t)
            self.traces["V"].append(self.neurons.V[k].copy())
            self.traces["g_e"].append(self.receptors.g_ampa[k].copy())
            self.traces["g_i"].append(self.receptors.g_gaba_a[k].copy())

        self.t += self.dt
        self.step_count += 1
        return fired

    def _deliver(self, fired: np.ndarray) -> None:
        """Push each spike's conductance into the delay buffer."""
        released: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, sc in self.syn_classes.items():
            sub = fired[self.class_members[name][fired]]
            if sub.size:
                released[name] = (sub, self.stp[name].release(sub, self.t))

        for (cname, d), M in self.matrices.items():
            if cname not in released:
                continue
            sub, rel = released[cname]
            contrib = rel @ M[sub]  # (n,) dense row of delivered conductance
            if not np.any(contrib):
                continue
            slot = (self.step_count + d) % self.D
            sc = self.syn_classes[cname]
            if sc.inhibitory:
                self.buf_gaba[slot] += contrib
            else:
                self.buf_ampa[slot] += contrib
                if sc.nmda_ratio:
                    self.buf_nmda[slot] += contrib * sc.nmda_ratio

    def run(self, duration_ms: float, on_step=None) -> None:
        """Integrate for ``duration_ms``; ``on_step(net, t)`` runs each step."""
        n_steps = int(round(duration_ms / self.dt))
        for _ in range(n_steps):
            if on_step is not None:
                on_step(self, self.t)
            self.step()

    # -- analysis helpers ---------------------------------------------------

    def pop_rates(self, pop: Population, t0: float, t1: float) -> np.ndarray:
        return self.spikes.rates(self.n, t0, t1)[pop.idx]

    def pop_mean_rate(self, pop: Population, t0: float, t1: float) -> float:
        return float(self.pop_rates(pop, t0, t1).mean())

    def trace_arrays(self) -> dict:
        return {k: np.asarray(v) for k, v in self.traces.items()}


class ExternalInput:
    """A projection from a spike source that is not part of the NeuronArray.

    The geniculate afferents are the case that matters: they carry their own
    short-term depression (Kandel's thalamocortical synapses are the strongly
    depressing kind) and their own conduction delay, but they deliver into the
    same receptor bank as every cortical synapse.
    """

    def __init__(self, net: Network, n_src: int, sc: SynapseClass):
        self.net = net
        self.n_src = n_src
        self.sc = sc
        self.stp = ShortTermPlasticity(n_src, sc)
        self._pending: list[tuple[sp.coo_matrix, float]] = []
        self.matrices: dict[int, sp.csr_matrix] = {}
        # Divisive gain on release, in [0, 1].  This is where presynaptic
        # inhibition acts: it scales the transmitter released by the afferent
        # rather than adding conductance at the target's soma, so it divides
        # the drive *before* it meets the spike threshold instead of competing
        # with it.
        self.gain = 1.0

    def connect(self, W: sp.spmatrix, delay_ms: float) -> None:
        if W.nnz == 0:
            return
        self._pending.append((W.tocoo(), delay_ms))
        self.net.reserve_delay(delay_ms)

    def compile(self) -> None:
        groups: dict[int, list] = defaultdict(list)
        for W, delay_ms in self._pending:
            d = max(1, int(round(delay_ms / self.net.dt)))
            groups[d].append(W)
        for d, ws in groups.items():
            rows = np.concatenate([w.row for w in ws])
            cols = np.concatenate([w.col for w in ws])
            vals = np.concatenate([w.data for w in ws])
            M = sp.coo_matrix(
                (vals, (rows, cols)), shape=(self.n_src, self.net.n)
            ).tocsr()
            M.sum_duplicates()
            self.matrices[d] = M
        self._pending = []

    def deliver(self, fired: np.ndarray, t: float) -> None:
        if fired.size == 0 or not self.matrices:
            return
        rel = self.stp.release(fired, t)
        if self.gain != 1.0:
            rel = rel * self.gain
        for d, M in self.matrices.items():
            contrib = rel @ M[fired]
            if np.any(contrib):
                self.net.push_conductance(
                    contrib, d, self.sc.inhibitory, self.sc.nmda_ratio
                )

    def reset(self) -> None:
        self.stp.reset()

    def fan_in(self) -> np.ndarray:
        """Number of afferents converging on each cortical neuron."""
        total = np.zeros(self.net.n)
        for M in self.matrices.values():
            total += np.diff(M.tocsc().indptr)
        return total


