"""The cortical column: layers, cell classes, and the intrinsic circuitry.

This assembles a hypercolumn of primary visual cortex -- Kandel Fig. 25-13's
"cortical computational module", the chunk of tissue roughly a millimetre
across that contains a full cycle of orientation columns, one cycle of left-
and right-eye ocular-dominance columns, and all of the functional and
anatomical cell types of V1.

Laminar scheme follows Fig. 25-12 and 25-15.  Layer IVC is split into the
magnocellular-recipient IVC-alpha and the parvocellular-recipient IVC-beta;
layer IVB is the magnocellular output sublayer.  The intrinsic excitatory
connections are those of Fig. 25-15B:

    LGN (M) -> IVC-alpha        LGN (P) -> IVC-beta
    IVC-alpha -> IVB            IVC-beta -> II/III
    IVB -> II/III, V            II/III -> V
    V -> II/III, V -> VI        VI -> IV, VI -> II/III

with layer II/III projecting to other cortical areas, layer V to superior
colliculus and pons, and layer VI back to the lateral geniculate nucleus.
Inhibition is carried by the GABAergic smooth cells the figure legend
mentions: perisomatic fast-spiking baskets and dendrite-targeting Martinotti
cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from . import neurons as nrn
from . import synapses as syn
from .connectivity import (
    column_positions,
    distance_connect,
    geniculate_to_simple_cell,
    ocular_dominance_index,
    orientation_tuned_connect,
    phase_opposed_connect,
    random_connect,
)
from .lgn import LGN
from .network import ExternalInput, Network, Population
from .stimuli import Blank, RetinalGrid, Stimulus

# --- Laminar composition ---------------------------------------------------
# Cell counts per orientation column at scale 1.0.  Proportions follow the
# laminar cell densities of macaque V1, where layer IV is exceptionally thick
# and layer VI is large -- V1 devotes far more of its depth to the input and
# feedback layers than any other cortical area does.
LAYER_PLAN = [
    # (name,        layer,      cell type,            count)
    ("L1_inh",      "I",        nrn.MARTINOTTI_LTS,   6),
    ("L23_pyr",     "II/III",   nrn.PYR_RS,           120),
    ("L23_pv",      "II/III",   nrn.BASKET_FS,        20),
    ("L23_sst",     "II/III",   nrn.MARTINOTTI_LTS,   12),
    ("L4B_pyr",     "IVB",      nrn.PYR_RS,           34),
    ("L4Ca_ss",     "IVCalpha", nrn.SPINY_STELLATE,   50),
    ("L4Cb_ss",     "IVCbeta",  nrn.SPINY_STELLATE,   70),
    ("L4_pv",       "IVC",      nrn.BASKET_FS,        30),
    ("L5_pyr",      "V",        nrn.PYR_RS,           34),
    ("L5_ib",       "V",        nrn.PYR_IB,           14),
    ("L5_pv",       "V",        nrn.BASKET_FS,        8),
    ("L5_sst",      "V",        nrn.MARTINOTTI_LTS,   5),
    ("L6_pyr",      "VI",       nrn.PYR_RS,           70),
    ("L6_ct",       "VI",       nrn.PYR_CT,           20),
    ("L6_pv",       "VI",       nrn.BASKET_FS,        15),
]

# Cortical depth of each layer in micrometres below the pia, used only for
# geometry and plotting.  V1 is about 2 mm thick.
LAYER_DEPTH = {
    "I": 75, "II/III": 450, "IVB": 800, "IVCalpha": 950,
    "IVCbeta": 1150, "IVC": 1050, "V": 1450, "VI": 1800,
}

# --- Intrinsic excitatory circuitry ---------------------------------------
# (pre, post, connection probability, peak conductance nS, delay ms)
EXC_INTRINSIC = [
    # Recurrent excitation inside the thalamorecipient layer.  Most synapses
    # on a layer IV cell come from other layer IV cells, not from the thalamus.
    ("L4Ca_ss", "L4Ca_ss", 0.10, 0.32, 0.8),
    ("L4Cb_ss", "L4Cb_ss", 0.10, 0.32, 0.8),
    # The two streams leaving layer IVC (Fig. 25-15B).
    ("L4Ca_ss", "L4B_pyr", 0.18, 0.55, 1.0),
    ("L4Cb_ss", "L23_pyr", 0.15, 0.50, 1.2),
    ("L4Ca_ss", "L23_pyr", 0.05, 0.30, 1.2),
    ("L4Cb_ss", "L4B_pyr", 0.04, 0.25, 1.0),
    # Layer IVB pyramid projecting to layers II, III and V.
    ("L4B_pyr", "L23_pyr", 0.12, 0.45, 0.8),
    ("L4B_pyr", "L5_pyr", 0.09, 0.40, 1.0),
    ("L4B_pyr", "L5_ib", 0.09, 0.40, 1.0),
    # Superficial recurrent excitation and the descending limb to layer V.
    ("L23_pyr", "L23_pyr", 0.12, 0.32, 0.8),
    ("L23_pyr", "L5_pyr", 0.12, 0.45, 1.0),
    ("L23_pyr", "L5_ib", 0.12, 0.45, 1.0),
    # Layer V pyramid projecting back up to layers II, III and down to VI.
    ("L5_pyr", "L23_pyr", 0.06, 0.28, 1.2),
    ("L5_ib", "L23_pyr", 0.04, 0.28, 1.2),
    ("L5_pyr", "L5_pyr", 0.10, 0.32, 0.8),
    ("L5_pyr", "L6_pyr", 0.10, 0.40, 1.0),
    ("L5_ib", "L6_ct", 0.09, 0.40, 1.0),
    ("L4Cb_ss", "L6_pyr", 0.10, 0.40, 1.4),
    ("L4Cb_ss", "L6_ct", 0.07, 0.35, 1.4),
    ("L4Ca_ss", "L6_pyr", 0.06, 0.35, 1.4),
    # Layer VI pyramid projecting to layer IV and to layers II, III.
    ("L6_pyr", "L4Ca_ss", 0.07, 0.22, 1.4),
    ("L6_pyr", "L4Cb_ss", 0.10, 0.22, 1.4),
    ("L6_ct", "L4Cb_ss", 0.06, 0.18, 1.4),
    ("L6_pyr", "L23_pyr", 0.04, 0.18, 1.4),
    ("L6_pyr", "L6_pyr", 0.08, 0.28, 0.8),
]

# Excitation onto fast-spiking baskets.  Note the probabilities are higher
# than onto pyramidal targets: local interneurons are densely and
# promiscuously innervated, which is why inhibition tracks excitation so
# closely and arrives only a millisecond or two behind it.
EXC_TO_PV_INTRINSIC = [
    ("L4Ca_ss", "L4_pv", 0.28, 0.55, 0.6),
    ("L4Cb_ss", "L4_pv", 0.28, 0.55, 0.6),
    ("L6_pyr", "L4_pv", 0.08, 0.30, 1.2),
    ("L23_pyr", "L23_pv", 0.28, 0.50, 0.6),
    ("L4Cb_ss", "L23_pv", 0.12, 0.40, 1.0),
    ("L4B_pyr", "L23_pv", 0.10, 0.40, 0.8),
    ("L5_pyr", "L5_pv", 0.28, 0.50, 0.6),
    ("L5_ib", "L5_pv", 0.22, 0.50, 0.6),
    ("L23_pyr", "L5_pv", 0.10, 0.35, 1.0),
    ("L6_pyr", "L6_pv", 0.28, 0.45, 0.6),
    ("L5_pyr", "L6_pv", 0.10, 0.35, 1.0),
]

# Excitation onto Martinotti cells: sparse and individually weak, but the
# synapse facilitates steeply, so these cells are recruited by sustained
# local firing rather than by single spikes.
EXC_TO_SST_INTRINSIC = [
    ("L23_pyr", "L23_sst", 0.20, 0.45, 0.8),
    ("L5_pyr", "L5_sst", 0.20, 0.45, 0.8),
    ("L5_ib", "L5_sst", 0.18, 0.45, 0.8),
    ("L23_pyr", "L1_inh", 0.10, 0.30, 1.0),
]

# Push-pull inhibition inside layer IVC, wired by spatial phase rather than
# at random.  Handled separately from the rule table below.
PUSH_PULL = [
    ("L4_pv", "L4Ca_ss", 0.55, 1.7, 0.6),
    ("L4_pv", "L4Cb_ss", 0.55, 1.7, 0.6),
]

# Perisomatic inhibition: strong, fast, and locally dense.
PV_INTRINSIC = [
    ("L4_pv", "L4B_pyr", 0.20, 1.2, 0.8),
    ("L4_pv", "L4_pv", 0.30, 1.1, 0.6),
    ("L23_pv", "L23_pyr", 0.38, 1.8, 0.6),
    ("L23_pv", "L23_pv", 0.30, 1.1, 0.6),
    ("L23_pv", "L23_sst", 0.12, 0.7, 0.8),
    ("L5_pv", "L5_pyr", 0.38, 1.8, 0.6),
    ("L5_pv", "L5_ib", 0.38, 1.8, 0.6),
    ("L5_pv", "L5_pv", 0.30, 1.1, 0.6),
    ("L6_pv", "L6_pyr", 0.38, 1.6, 0.6),
    ("L6_pv", "L6_ct", 0.38, 1.6, 0.6),
    ("L6_pv", "L6_pv", 0.30, 1.1, 0.6),
]

# Dendrite-targeting inhibition: weaker per spike, slower, and aimed at the
# apical tuft, where it gates the contextual and feedback input arriving in
# layer I rather than vetoing the cell outright.
SST_INTRINSIC = [
    ("L23_sst", "L23_pyr", 0.35, 0.75, 0.8),
    ("L23_sst", "L23_pv", 0.20, 0.45, 0.8),
    ("L5_sst", "L5_pyr", 0.35, 0.75, 0.8),
    ("L5_sst", "L5_ib", 0.35, 0.75, 0.8),
    ("L1_inh", "L23_pyr", 0.25, 0.55, 1.0),
    ("L1_inh", "L5_pyr", 0.15, 0.45, 1.2),
    ("L1_inh", "L5_ib", 0.15, 0.45, 1.2),
]


@dataclass
class ColumnConfig:
    """Everything that sets the scale and geometry of the model."""

    n_columns: int = 8  # orientation columns spanning 180 degrees
    scale: float = 1.0  # multiplier on every population size
    dt: float = 0.1  # ms
    seed: int = 0
    pinwheel_radius_um: float = 250.0
    column_jitter_um: float = 55.0
    rf_size_deg: float = 2.0
    rf_pixels: int = 40
    lgn_spacing_deg: float = 0.05
    # Feedforward receptive-field geometry
    sf_parvo: float = 2.6  # cycles/deg, layer IVC-beta
    sf_magno: float = 1.4  # cycles/deg, layer IVC-alpha
    rf_aspect: float = 2.4  # envelope elongation along the preferred axis
    theta_jitter_deg: float = 9.0  # scatter of preferred orientation in IV
    rf_scatter_deg: float = 0.045  # scatter of RF centres within a column
    # Peak thalamocortical conductance.  This is a *connection* weight, not a
    # single release site: one geniculate axon makes several contacts on a
    # layer IVC cell, and the model lumps them, so a few nanosiemens is the
    # right order of magnitude rather than the ~0.2 nS of one synapse.
    w_thal: float = 14.0  # nS
    w_thal_pv: float = 3.2  # thalamic drive onto layer IV baskets
    thal_min_frac: float = 0.03  # keep afferents above this fraction of peak
    # Fraction of the layer IV basket cells' thalamic drive that is an
    # oriented, phase-specific Gabor rather than untuned circular pooling.
    pv_tuned_frac: float = 0.6
    push_pull_opposition: float = 0.85
    push_pull_sigma_theta: float = 32.0
    thal_delay_ms: float = 1.6
    # Global gains on each synapse class.  The rule tables below hold the
    # *relative* strengths of the pathways, which is what the anatomy
    # constrains; these multipliers set the absolute scale.  They are needed
    # because the model has roughly 25 times fewer neurons than a real
    # hypercolumn, so each modelled connection has to stand in for many real
    # ones -- the standard consequence of down-scaling a network while trying
    # to preserve its operating point.
    gain_exc: float = 26.0  # excitatory -> excitatory
    gain_e2pv: float = 22.0  # excitatory -> basket
    gain_e2sst: float = 22.0  # excitatory -> Martinotti
    gain_pv: float = 24.0  # basket -> anything
    gain_sst: float = 18.0  # Martinotti -> anything
    # Horizontal (inter-columnar) connectivity
    horiz_sigma_um: float = 700.0
    horiz_sigma_theta_deg: float = 26.0
    horiz_p_max: float = 0.09
    horiz_w: float = 0.26
    horiz_p_max_pv: float = 0.10
    horiz_w_pv: float = 0.34
    local_p_max: float = 0.03
    local_sigma_um: float = 180.0
    local_w: float = 0.22
    # Normalisation pool: excitation onto the layer IV baskets gathered from
    # every orientation column, with no like-to-like preference at all.
    #
    # This is what divisive normalisation needs and what the like-to-like
    # horizontal connections cannot supply.  A basket cell driven only by its
    # own column inherits that column's orientation tuning, so strengthening
    # it subtracts hardest at the preferred orientation and *flattens* the
    # tuning curve.  A basket cell that sums over all orientations instead
    # reports the total drive in the hypercolumn -- a quantity that grows with
    # contrast but is almost independent of orientation -- which is exactly
    # the signal needed to make the effective threshold track contrast.
    norm_p: float = 0.0  # set > 0 to enable the pool
    norm_w: float = 0.30
    norm_sigma_um: float = 2000.0  # effectively hypercolumn-wide
    # The return arm.  Basket output onto layer IVC is otherwise wired by
    # spatial phase (push-pull), which is structured inhibition and cannot
    # deliver an unstructured, drive-proportional suppression however hard the
    # baskets are driven.  This adds a phase- and orientation-blind component.
    norm_inh_p: float = 0.0
    norm_inh_w: float = 1.7
    # Presynaptic normalisation of the geniculate drive.
    #
    # Somatic inhibition cannot regulate a spike threshold cleanly, because it
    # is competing with the same threshold it is trying to set: raising it
    # suppresses the preferred response as much as the null one.  Presynaptic
    # inhibition of the thalamocortical terminals divides the drive *at the
    # synapse*, so the cortex sees a rescaled input rather than a shifted
    # operating point.  GABA-B receptors on thalamocortical terminals are the
    # real counterpart.
    #
    # The controlling signal is the mean firing rate of the layer IV basket
    # population summed over *every* orientation column.  Because the columns
    # span the full 180 degrees at one retinotopic location, that sum measures
    # total drive and is nearly blind to orientation -- which is precisely the
    # quantity a divisive normaliser needs.
    # gain = 1 / (1 + k * max(pool - baseline, 0)).  Rectifying at the pool's
    # spontaneous rate means the gain is 1 when nothing is being shown, so
    # w_thal keeps its meaning and only the *contrast-dependent* part of the
    # pool does any dividing.
    presyn_norm_k: float = 0.0  # set > 0 to enable
    presyn_norm_baseline: float = 14.0  # Hz, spontaneous pool rate
    presyn_norm_tau: float = 60.0  # ms, the slow kinetics of GABA-B
    presyn_norm_floor: float = 0.12  # never divide the drive away entirely
    # Background bombardment representing the rest of cortex.  The rate and
    # the per-event weight are set separately on purpose: their product fixes
    # the mean conductance, but the weight alone fixes the size of the
    # membrane-potential fluctuations.  Cortical neurons in vivo sit a couple
    # of millivolts below threshold and are pushed over it by those
    # fluctuations, so getting the variance right matters as much as the mean.
    bg_rate_exc: float = 1200.0  # events/s summed over all background inputs
    bg_weight_exc: float = 0.95  # nS per event
    bg_rate_inh: float = 700.0
    bg_weight_inh: float = 0.70
    bg_tonic_pyr: float = 118.0  # pA of tonic depolarisation
    # Interneurons carry smaller dendritic trees and therefore fewer
    # background synapses than pyramidal cells.
    bg_gain: dict = field(default_factory=lambda: {
        "pyr": 1.0, "ss": 1.0, "pv": 0.72, "sst": 0.60, "inh": 0.60,
    })
    # Per-class scaling of the tonic drive.  Layer IVC is the one population
    # with a *negative* value, and the reason is central to how the model
    # works.  A drifting grating raises the mean geniculate conductance by the
    # same amount at every orientation; only the modulated component carries
    # orientation.  A layer IVC cell must therefore sit far enough below
    # threshold that the untuned mean alone does not fire it, leaving the
    # modulation to do the work -- Hubel and Wiesel's iceberg, made
    # quantitative.  The offset stands for the tonic inhibition and adaptation
    # that hold the thalamorecipient layer near threshold in vivo.
    tonic_gain: dict = field(default_factory=lambda: {
        "pyr": 1.0, "ss": -7.3, "L4Ca_ss": -11.5,
        "pv": 0.60, "sst": 0.42, "inh": 0.42,
    })


class CorticalColumn:
    """A V1 hypercolumn wired to a geniculate input array."""

    def __init__(self, cfg: ColumnConfig | None = None):
        self.cfg = cfg or ColumnConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        self.grid = RetinalGrid(self.cfg.rf_size_deg, self.cfg.rf_pixels)
        self.lgn = LGN(
            self.grid, spacing_deg=self.cfg.lgn_spacing_deg,
            rng=np.random.default_rng(self.cfg.seed + 101),
        )

        self.col_xy, self.col_theta = column_positions(
            self.cfg.n_columns, self.cfg.pinwheel_radius_um
        )
        self.col_od = ocular_dominance_index(self.col_xy[:, 0])

        n_total = self._count_neurons()
        self.net = Network(n_total, dt=self.cfg.dt, seed=self.cfg.seed + 7)

        self.pops: dict[str, Population] = {}
        self.pos = np.zeros((n_total, 2))  # cortical surface position, um
        self.depth = np.zeros(n_total)  # depth below pia, um
        self.theta_pref = np.full(n_total, np.nan)  # degrees
        self.rf_phase = np.zeros(n_total)  # radians
        self.rf_x = np.zeros(n_total)
        self.rf_y = np.zeros(n_total)
        self.od_index = np.zeros(n_total)
        self.column_of = np.full(n_total, -1)
        self.pop_of = np.empty(n_total, dtype=object)

        self._build_populations()

        # The normalisation pool: every layer IV basket, in every orientation
        # column.  Spanning all orientations at one retinotopic position is
        # what makes its summed activity a measure of drive rather than of
        # orientation.
        self._norm_pool_mask = np.zeros(n_total, dtype=bool)
        self._norm_pool_mask[self.group("L4_pv")] = True
        self._n_norm_pool = int(self._norm_pool_mask.sum())
        self.norm_pool_rate = 0.0

        self._wire_intrinsic()
        self._wire_horizontal()
        self._wire_thalamus()
        self._set_background()
        self.net.compile()
        self.thalamic.compile()
        self.net.reset()
        self.lgn.reset()

    # -- construction -------------------------------------------------------

    def _count_neurons(self) -> int:
        per_col = sum(self._size(c) for _, _, _, c in LAYER_PLAN)
        return per_col * self.cfg.n_columns

    def _size(self, base: int) -> int:
        return max(1, int(round(base * self.cfg.scale)))

    def key(self, base: str, col: int) -> str:
        return f"{base}__c{col}"

    def _build_populations(self) -> None:
        cfg = self.cfg
        for c in range(cfg.n_columns):
            cx, cy = self.col_xy[c]
            for base, layer, ct, count in LAYER_PLAN:
                n = self._size(count)
                pop = self.net.add_population(
                    self.key(base, c), n, ct, layer, column=c
                )
                idx = pop.idx
                # Cells scatter around the column centre; the scatter is what
                # makes distance-dependent connectivity meaningful within a
                # column as well as between columns.
                self.pos[idx, 0] = cx + self.rng.normal(
                    0, cfg.column_jitter_um, n
                )
                self.pos[idx, 1] = cy + self.rng.normal(
                    0, cfg.column_jitter_um, n
                )
                self.depth[idx] = LAYER_DEPTH[layer] + self.rng.normal(0, 30, n)
                self.column_of[idx] = c
                self.od_index[idx] = self.col_od[c]
                self.pop_of[idx] = base

                # Functional labels.  Orientation preference is inherited from
                # the column with scatter; spatial phase is drawn independently
                # for every cell, which is the crucial degree of freedom -- it
                # is what lets layer II/III build a phase-invariant complex
                # cell out of phase-sensitive layer IV inputs.
                self.theta_pref[idx] = (
                    self.col_theta[c]
                    + self.rng.normal(0, cfg.theta_jitter_deg, n)
                ) % 180.0
                self.rf_phase[idx] = self.rng.uniform(0, 2 * np.pi, n)
                self.rf_x[idx] = self.rng.normal(0, cfg.rf_scatter_deg, n)
                self.rf_y[idx] = self.rng.normal(0, cfg.rf_scatter_deg, n)
                self.pops[self.key(base, c)] = pop

    def _connect_rules(self, rules, sc: syn.SynapseClass,
                       gain: float = 1.0) -> None:
        """Apply an intrinsic rule table within every column."""
        for c in range(self.cfg.n_columns):
            for pre_b, post_b, p, w, delay in rules:
                pre = self.pops[self.key(pre_b, c)]
                post = self.pops[self.key(post_b, c)]
                W = random_connect(
                    len(pre), len(post), p, w * gain, self.rng,
                    allow_self=pre is not post,
                )
                self.net.connect(pre, post, sc, W, delay)

    def _wire_intrinsic(self) -> None:
        cfg = self.cfg
        self._connect_rules(EXC_INTRINSIC, syn.EXC_TO_EXC, cfg.gain_exc)
        self._connect_rules(EXC_TO_PV_INTRINSIC, syn.EXC_TO_PV, cfg.gain_e2pv)
        self._connect_rules(EXC_TO_SST_INTRINSIC, syn.EXC_TO_SST, cfg.gain_e2sst)
        self._connect_rules(PV_INTRINSIC, syn.PV_TO_ANY, cfg.gain_pv)
        self._connect_rules(SST_INTRINSIC, syn.SST_TO_ANY, cfg.gain_sst)
        self._wire_push_pull()

    def _wire_push_pull(self) -> None:
        """Layer IV inhibition wired in antiphase with the excitation."""
        cfg = self.cfg
        for c in range(cfg.n_columns):
            for pre_b, post_b, p, w, delay in PUSH_PULL:
                pre = self.pops[self.key(pre_b, c)]
                post = self.pops[self.key(post_b, c)]
                W = phase_opposed_connect(
                    self.rf_phase[pre.idx], self.rf_phase[post.idx],
                    self.theta_pref[pre.idx], self.theta_pref[post.idx],
                    p, cfg.push_pull_sigma_theta, w * cfg.gain_pv,
                    self.rng, opposition=cfg.push_pull_opposition,
                )
                self.net.connect(pre, post, syn.PV_TO_ANY, W, delay)

                if cfg.norm_inh_p > 0:
                    Wn = random_connect(
                        len(pre), len(post), cfg.norm_inh_p,
                        cfg.norm_inh_w * cfg.gain_pv, self.rng,
                    )
                    self.net.connect(pre, post, syn.PV_TO_ANY, Wn, delay)

    def _wire_horizontal(self) -> None:
        """Long-range horizontal connections between orientation columns.

        Kandel Fig. 25-16: superficial pyramidal axons run for millimetres
        parallel to the surface and terminate in clusters on columns of
        similar orientation preference, contacting both pyramidal cells and
        inhibitory interneurons.  Because they also recruit inhibition, the
        net effect of the surround is usually suppressive even though the
        connection itself is excitatory.
        """
        cfg = self.cfg
        for a in range(cfg.n_columns):
            for b in range(cfg.n_columns):
                if a == b:
                    continue
                d = np.linalg.norm(self.col_xy[a] - self.col_xy[b])
                # Horizontal axons conduct slowly: about 0.2 m/s, so a
                # millimetre costs several milliseconds.  That delay is why
                # contextual modulation always lags the feedforward response.
                delay = 1.2 + d / 200.0

                for src in ("L23_pyr", "L5_pyr"):
                    pre = self.pops[self.key(src, a)]
                    pre_idx = pre.idx
                    for dst, p_max, w, s_theta in (
                        ("L23_pyr", cfg.horiz_p_max, cfg.horiz_w,
                         cfg.horiz_sigma_theta_deg),
                        ("L23_pv", cfg.horiz_p_max_pv, cfg.horiz_w_pv,
                         cfg.horiz_sigma_theta_deg * 1.6),
                    ):
                        post = self.pops[self.key(dst, b)]
                        sc = (
                            syn.EXC_TO_PV if dst.endswith("pv")
                            else syn.EXC_TO_EXC
                        )
                        W = orientation_tuned_connect(
                            self.theta_pref[pre_idx],
                            self.theta_pref[post.idx],
                            self.pos[pre_idx], self.pos[post.idx],
                            p_max, cfg.horiz_sigma_um, s_theta, w, self.rng,
                        )
                        self.net.connect(pre, post, sc, W, delay)

                # The normalisation pool: every column's layer IV excitatory
                # cells drive every column's baskets, orientation-blind.
                if cfg.norm_p > 0:
                    for src in ("L4Cb_ss", "L4Ca_ss"):
                        pre = self.pops[self.key(src, a)]
                        post = self.pops[self.key("L4_pv", b)]
                        W = distance_connect(
                            self.pos[pre.idx], self.pos[post.idx],
                            cfg.norm_p, cfg.norm_sigma_um, cfg.norm_w,
                            self.rng,
                        )
                        self.net.connect(pre, post, syn.EXC_TO_PV, W, delay)

                # Short-range, orientation-independent mixing between
                # neighbouring columns, including in the input layer.
                for src, dst in (
                    ("L4Cb_ss", "L4Cb_ss"), ("L4Ca_ss", "L4Ca_ss"),
                    ("L4Cb_ss", "L4_pv"), ("L23_pyr", "L23_pyr"),
                ):
                    pre = self.pops[self.key(src, a)]
                    post = self.pops[self.key(dst, b)]
                    sc = syn.EXC_TO_PV if dst.endswith("pv") else syn.EXC_TO_EXC
                    W = distance_connect(
                        self.pos[pre.idx], self.pos[post.idx],
                        cfg.local_p_max, cfg.local_sigma_um, cfg.local_w,
                        self.rng,
                    )
                    self.net.connect(pre, post, sc, W, delay)

    def _wire_thalamus(self) -> None:
        """Geniculate afferents onto layer IVC and the layer IV baskets.

        Two things are happening here.  The excitatory cells get the
        Hubel-Wiesel arrangement -- a row of ON-centre afferents flanked by
        OFF-centre ones, aligned with the column's preferred orientation --
        while the fast-spiking baskets pool their thalamic input over a
        *circular* region with no orientation alignment at all.  That
        asymmetry is deliberate: it produces broadly tuned inhibition against
        sharply tuned excitation, which is the arrangement that sharpens the
        cortical tuning curve beyond its feedforward input.
        """
        cfg = self.cfg
        self.thalamic = ExternalInput(
            self.net, self.lgn.n, syn.THALAMOCORTICAL
        )
        lx, ly, lsign = self.lgn.x, self.lgn.y, self.lgn.sign

        for c in range(cfg.n_columns):
            od = self.col_od[c]
            # Ocular dominance: the relative strength of the two eyes' input,
            # set by where this column sits in the ocular-dominance stripe.
            eye_gain = {0: 0.5 * (1.0 - od), 1: 0.5 * (1.0 + od)}

            for base, stream, sf in (
                ("L4Ca_ss", "M", cfg.sf_magno),
                ("L4Cb_ss", "P", cfg.sf_parvo),
            ):
                pop = self.pops[self.key(base, c)]
                sel = self.lgn.select(stream=stream)
                rows, cols, vals = [], [], []
                sigma_across = 1.0 / (3.0 * sf)
                for j, gid in enumerate(pop.idx):
                    params = dict(
                        x0=self.rf_x[gid], y0=self.rf_y[gid],
                        theta=self.theta_pref[gid], sf=sf,
                        phase=self.rf_phase[gid],
                        sigma_across=sigma_across,
                        sigma_along=sigma_across * cfg.rf_aspect,
                    )
                    w = geniculate_to_simple_cell(
                        lx[sel], ly[sel], lsign[sel], params,
                        cfg.w_thal, self.rng, min_frac=cfg.thal_min_frac,
                    )
                    gains = np.array([eye_gain[e] for e in self.lgn.eye[sel]])
                    w = w * gains
                    live = np.flatnonzero(w > 1e-6)
                    if live.size:
                        rows.append(sel[live])
                        cols.append(np.full(live.size, j))
                        vals.append(w[live])
                if rows:
                    W = sp.coo_matrix(
                        (np.concatenate(vals),
                         (np.concatenate(rows),
                          np.concatenate(cols) + pop.start)),
                        shape=(self.lgn.n, self.net.n),
                    )
                    self.thalamic.connect(W, cfg.thal_delay_ms)

            # Thalamic drive onto the layer IV fast-spiking cells.  These get
            # a mixture: an oriented, phase-specific Gabor component that makes
            # them simple cells in their own right, and a broad, circular
            # pooling component that is orientation-blind.  Real layer IV
            # interneurons come in both flavours, and the model needs both --
            # the tuned part supplies push-pull, the untuned part supplies the
            # broadly tuned inhibition that sharpens the excitatory tuning.
            pop = self.pops[self.key("L4_pv", c)]
            sel_p = self.lgn.select(stream="P")
            rows, cols, vals = [], [], []
            for j, gid in enumerate(pop.idx):
                params = dict(
                    x0=self.rf_x[gid], y0=self.rf_y[gid],
                    theta=self.theta_pref[gid], sf=cfg.sf_parvo,
                    phase=self.rf_phase[gid],
                    sigma_across=1.0 / (3.0 * cfg.sf_parvo),
                    sigma_along=cfg.rf_aspect / (3.0 * cfg.sf_parvo),
                )
                w = np.zeros(self.lgn.n)
                w[sel_p] = geniculate_to_simple_cell(
                    lx[sel_p], ly[sel_p], lsign[sel_p], params,
                    cfg.w_thal_pv * cfg.pv_tuned_frac, self.rng,
                    min_frac=cfg.thal_min_frac,
                )
                d2 = (lx - self.rf_x[gid]) ** 2 + (ly - self.rf_y[gid]) ** 2
                pool = np.exp(-d2 / (2 * 0.16 ** 2)) * self.rng.random(self.lgn.n)
                w += pool * cfg.w_thal_pv * (1.0 - cfg.pv_tuned_frac)
                w *= np.array([eye_gain[e] for e in self.lgn.eye])
                w[w < 0.03 * cfg.w_thal_pv] = 0.0
                live = np.flatnonzero(w > 0)
                if live.size:
                    rows.append(live)
                    cols.append(np.full(live.size, j + pop.start))
                    vals.append(w[live])
            if rows:
                W = sp.coo_matrix(
                    (np.concatenate(vals),
                     (np.concatenate(rows), np.concatenate(cols))),
                    shape=(self.lgn.n, self.net.n),
                )
                # Thalamic input reaches the basket cells slightly *before* it
                # reaches the pyramidal cells, so inhibition can clamp the
                # response window from the first millisecond.
                self.thalamic.connect(W, cfg.thal_delay_ms - 0.3)

    def _set_background(self) -> None:
        cfg = self.cfg
        for name, pop in self.net.populations.items():
            base = name.split("__")[0]
            kind = base.rsplit("_", 1)[-1]
            gain = cfg.bg_gain.get(base, cfg.bg_gain.get(kind, 1.0))
            # A population-specific entry wins over the cell-class default.
            # Layer IVC-alpha needs its own: the magnocellular afferents that
            # drive it have higher contrast gain and higher firing rates than
            # the parvocellular ones driving IVC-beta, so it takes a larger
            # standing hyperpolarisation to hold it at the same distance below
            # threshold.
            tgain = cfg.tonic_gain.get(base, cfg.tonic_gain.get(kind, 1.0))
            self.net.set_background(
                pop,
                cfg.bg_rate_exc * gain, cfg.bg_weight_exc,
                cfg.bg_rate_inh * gain, cfg.bg_weight_inh,
                tonic=cfg.bg_tonic_pyr * tgain,
            )

    # -- running ------------------------------------------------------------

    def reset(self) -> None:
        self.net.reset()
        self.lgn.reset()
        self.thalamic.reset()
        self.norm_pool_rate = 0.0
        self.thalamic.gain = 1.0

    def _update_presyn_norm(self, n_pool_spikes: int) -> None:
        """Advance the presynaptic gain from the pooled basket activity."""
        cfg = self.cfg
        inst = n_pool_spikes * 1000.0 / (self._n_norm_pool * cfg.dt)
        self.norm_pool_rate += (
            (inst - self.norm_pool_rate) * cfg.dt / cfg.presyn_norm_tau
        )
        drive = max(self.norm_pool_rate - cfg.presyn_norm_baseline, 0.0)
        g = 1.0 / (1.0 + cfg.presyn_norm_k * drive)
        self.thalamic.gain = max(g, cfg.presyn_norm_floor)

    def present(self, stim: Stimulus, duration_ms: float,
                record_lgn: bool = False) -> dict:
        """Run the model while showing ``stim``; return timing information."""
        n_steps = int(round(duration_ms / self.cfg.dt))
        t0 = self.net.t
        lgn_counts = np.zeros(self.lgn.n) if record_lgn else None

        norm_on = self.cfg.presyn_norm_k > 0
        pool = self._norm_pool_mask

        for _ in range(n_steps):
            fired = self.lgn.step(stim, self.net.t, self.cfg.dt)
            self.thalamic.deliver(fired, self.net.t)
            if record_lgn and fired.size:
                np.add.at(lgn_counts, fired, 1)
            cortical = self.net.step()
            if norm_on:
                # The pool is read from the spikes just emitted, so the gain
                # applied to the next timestep's afferents reflects the
                # cortex's current activity -- a feedback loop with one step
                # of delay, as any real one has.
                n_pool = int(pool[cortical].sum()) if cortical.size else 0
                self._update_presyn_norm(n_pool)

        out = {"t0": t0, "t1": self.net.t, "duration": duration_ms}
        if record_lgn:
            out["lgn_rates"] = lgn_counts * 1000.0 / duration_ms
        return out

    def settle(self, duration_ms: float = 200.0) -> None:
        """Run with a blank field so transients from the initial conditions
        die out before any measurement is taken."""
        self.present(Blank(), duration_ms)

    # -- introspection ------------------------------------------------------

    def group(self, base: str) -> np.ndarray:
        """All neuron indices of a population type across every column."""
        return np.concatenate(
            [self.pops[self.key(base, c)].idx
             for c in range(self.cfg.n_columns)]
        )

    def rates(self, base: str, t0: float, t1: float) -> np.ndarray:
        return self.net.spikes.rates(self.net.n, t0, t1)[self.group(base)]

    def population_names(self) -> list[str]:
        return [p[0] for p in LAYER_PLAN]

    def summary(self) -> str:
        cfg = self.cfg
        lines = [
            f"V1 hypercolumn: {cfg.n_columns} orientation columns, "
            f"{self.net.n} neurons",
            f"  pinwheel radius {cfg.pinwheel_radius_um:.0f} um, "
            f"dt {cfg.dt} ms",
            self.lgn.summary(),
            "  preferred orientations: "
            + ", ".join(f"{t:.0f}" for t in self.col_theta) + " deg",
            "  ocular dominance index: "
            + ", ".join(f"{o:+.2f}" for o in self.col_od),
            "",
            "  layer          cell type        n/column   total",
        ]
        for base, layer, ct, count in LAYER_PLAN:
            n = self._size(count)
            lines.append(
                f"  {layer:<12s} {base:<15s} {n:6d} {n * cfg.n_columns:8d}"
            )
        n_syn = sum(M.nnz for M in self.net.matrices.values())
        n_thal = sum(M.nnz for M in self.thalamic.matrices.values())
        lines += [
            "",
            f"  intracortical synapses: {n_syn:,}",
            f"  thalamocortical synapses: {n_thal:,}",
        ]
        return "\n".join(lines)
