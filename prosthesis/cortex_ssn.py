"""A forward model of cortical microstimulation built from measured parts.

The earlier model (``cortex_circuit.CircuitSpatial``) made two structural
assumptions that the literature does not support, and this module replaces
both.  It is deliberately a separate class: the first model is what the
existing manuscript reports, and it stays reproducible.

What was wrong, and what replaces it
------------------------------------

1.  *Direct activation was a Gaussian ball of tissue.*  Two-photon imaging
    (Histed, Bonin & Reid 2009) shows instead a sparse set of cells scattered
    over hundreds of microns, which fills in with current rather than
    expanding, because the electrode excites *axons* and the cell bodies those
    axons belong to lie elsewhere.  Kumaravelu et al. (2022) reconcile this
    with the classical current-distance law: the volume in which spikes are
    initiated does grow with current, but the somata recruited are those
    whose axons have excitable sites (terminals, branch points) inside it.

    Here: spikes are initiated within a ball of radius R = sqrt(I / K_ax).  A
    cell at lateral offset d from the tip has an expected number of excitable
    sites inside the ball of

        hits = s * (4/3) pi R^3 * F(d),

    where F is the cell's axonal footprint (per unit area, integrating to 1)
    and s the density of excitable sites per unit depth.  The cell fires with
    probability p = 1 - exp(-hits).  Contributions from several electrodes add
    in ``hits`` -- a cell can be recruited only once -- so occupancy is a
    saturating nonlinearity with a physical origin rather than an assumed one.

    The footprint is not a free function: an excitatory cell's axon terminals
    are distributed according to the same lateral connectivity kernel that
    wires the network below, so one piece of anatomy fixes both.

2.  *Interaction was a divisive-normalisation formula.*  That formula
    describes what cortex computes without saying how.  The stabilized
    supralinear network (Ahmadian, Rubin & Miller 2013; Rubin, Van Hooser &
    Miller 2015) is the current mechanistic account: excitatory and
    inhibitory populations with a supralinear input-output function, in which
    normalisation, surround suppression and the switch from supralinear to
    sublinear summation all emerge from recurrent dynamics.  The parameters
    are those of Obeid & Miller's model of V1 layer 2/3 (eNeuro 2025), taken
    as published and not retuned.

One piece of physics couples the two and matters for every prediction: a
spike initiated in the axon cannot be vetoed by inhibition arriving at the
soma.  Directly driven cells therefore fire whatever the network does, and
only *transsynaptic* activity is subject to suppression.

Their synaptic output is not what the pulse rate would suggest, however.
Cortical synapses depress, and driven at hundreds of hertz they transmit a
steady-state drive that saturates near 1/(U tau_rec).  The network's weights
describe synapses at visually driven rates, so a directly driven cell is
given the equivalent rate at which its depressed output matches (see
``effective_rate``).

Provenance of every number is in ``BIOPHYSICAL_MODEL.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft as sfft

__all__ = ["SSNParams", "RecruitmentParams", "SSNSheet", "BiophysicalCortex",
           "effective_rate", "orientation_map_periodic"]


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SSNParams:
    """Obeid & Miller (eNeuro 2025), rate model of V1 layer 2/3.

    Lengths are in microns and are quoted on the 107 um grid they were
    published for; weights are sums over that grid and are rescaled by area
    when the model is run at another resolution.
    """

    k: float = 0.01
    n: float = 2.2
    tau_e: float = 10.0          # ms
    tau_i: float = 6.67          # ms
    ref_step_um: float = 107.0
    # Excitatory projections: dense within L_o, Gaussian skirt beyond it.
    L_o: float = 321.0
    sigma_ee: float = 321.0
    sigma_ie: float = 642.0
    J_ee_near: float = 0.072
    J_ie_near: float = 0.060
    J_ee_far: float = 0.036
    J_ie_far: float = 0.036
    # Inhibitory projections: local.
    sigma_i: float = 214.0
    J_ei: float = 0.0528
    J_ii: float = 0.0288
    # Orientation dependence q = A + B exp(-dtheta^2 / 2 sigma^2).
    A_near: float = 0.20
    B_near: float = 0.80
    sori_near: float = 55.0
    A_far: float = 0.14
    B_far: float = 0.86
    sori_far: float = 25.0
    # Orientation map: eight periods across the published 8 mm sheet.
    map_period_um: float = 1000.0
    # Rate ceilings.  Not part of the published model, which is never driven
    # hard enough to need them; microstimulation is.  Set far above the
    # visually driven range so the validated regime is untouched.
    r_max_e: float = 200.0
    r_max_i: float = 400.0
    # Fraction of inhibition carried by a slow (GABA-B-like) component.
    # Microstimulation is followed by inhibition lasting on the order of
    # 100-150 ms that is reduced by GABA-B blockade (Butovas & Schwarz 2003;
    # Butovas et al. 2006).  Steady states do not depend on this split.
    slow_inh_frac: float = 0.0
    tau_slow: float = 100.0      # ms


@dataclass(frozen=True)
class RecruitmentParams:
    """Direct, axonal recruitment by the electrode."""

    # Current-distance constant for spike *initiation* (uA / mm^2).  Derived
    # from Kumaravelu et al. 2022: median initiation radius 26, 56 and 105 um
    # at 5, 20 and 50 uA.  A median over a uniform ball sits at 0.794 R, so
    # R = 33, 71, 132 um and K = I / R^2 = 4700, 4000, 2900; geometric mean
    # below.  Compare 1292 (Stoney 1968, pyramidal-tract somata) and 675
    # (Tehovnik), both of which describe a ball of cell bodies.
    K_ax: float = 3800.0
    # Excitable axonal sites per neuron per mm of cortical depth.  The
    # excitatory value is calibrated so that roughly five cells in a
    # two-photon plane respond at 8.5 uA (Histed et al. 2009).  The
    # inhibitory value is calibrated to the measured ratio of recruited
    # inhibitory to excitatory fractions, about 1.1 up to 50 uA in awake mouse
    # V1 (Dadarlat, Sun & Stryker 2024), within 300 um of the tip.  Equal
    # densities would give 1.7, because interneuron axons are more compact
    # and so place more of their sites near the electrode.  Both are swept in
    # the sensitivity analysis (``v2_validate.py`` checks them).
    site_density_e: float = 9.0
    site_density_i: float = 6.0
    # Tsodyks-Markram parameters used only to convert a pulse rate into the
    # equivalent undepressed presynaptic rate.
    U_e: float = 0.45
    tau_rec_e: float = 300.0     # ms
    U_i: float = 0.35
    tau_rec_i: float = 150.0     # ms
    ref_rate_e: float = 20.0     # Hz, rate regime the network weights describe
    ref_rate_i: float = 30.0


def effective_rate(pulse_hz: float, U: float, tau_rec_ms: float,
                   ref_hz: float) -> float:
    """Undepressed-equivalent presynaptic rate of a cell driven at ``pulse_hz``.

    A depressing synapse in steady state transmits per spike a fraction
    x = 1 / (1 + U tau f) of its resting efficacy.  The network's weights are
    effective weights at visually driven rates, so efficacy is expressed
    relative to x at ``ref_hz``.  The result rises linearly at low rates and
    saturates at ref + 1/(U tau): driving harder than a few tens of hertz buys
    almost no additional synaptic output.
    """
    if pulse_hz <= 0:
        return 0.0
    r_d = 1000.0 / (U * tau_rec_ms)
    return float(pulse_hz * (ref_hz + r_d) / (pulse_hz + r_d))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def orientation_map_periodic(n: int, step_um: float, period_um: float,
                             seed: int = 0, bandwidth: float = 0.10
                             ) -> np.ndarray:
    """Orientation preference map (radians, modulo pi) on a periodic sheet.

    Half the phase of a complex field whose power sits on a ring at the column
    frequency -- equivalent to Kaschube et al.'s superposition of plane waves
    of one wavelength, and exactly periodic so the sheet can be convolved by
    FFT without an edge.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    f = np.hypot(*np.meshgrid(sfft.fftfreq(n, d=step_um),
                              sfft.fftfreq(n, d=step_um)))
    f0 = 1.0 / period_um
    ring = np.exp(-((f - f0) ** 2) / (2 * (bandwidth * f0) ** 2))
    zf = sfft.ifft2(sfft.fft2(z) * ring)
    return (np.angle(zf) / 2.0) % np.pi


def _ori_harmonics(A: float, B: float, sigma_deg: float, M: int) -> np.ndarray:
    """Cosine-series coefficients of q(d) = A + B exp(-d^2 / 2 sigma^2).

    q(d) = sum_m c_m cos(2 m d) on the orientation circle.  Expanding the
    kernel this way turns an orientation-dependent sum over the whole sheet
    into a handful of plain convolutions.
    """
    d = np.linspace(-np.pi / 2, np.pi / 2, 7200, endpoint=False)
    q = A + B * np.exp(-np.rad2deg(d) ** 2 / (2 * sigma_deg ** 2))
    c = [q.mean()]
    for m in range(1, M + 1):
        c.append(2.0 * (q * np.cos(2 * m * d)).mean())
    return np.array(c)


def _periodic_distance(n: int, step_um: float) -> np.ndarray:
    i = np.arange(n)
    i = np.minimum(i, n - i) * step_um
    return np.hypot(i[:, None], i[None, :])


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------

class SSNSheet:
    """Stabilized supralinear network on a periodic 2-D sheet of cortex."""

    def __init__(self, n: int = 128, step_um: float = 107.0, seed: int = 0,
                 params: SSNParams | None = None, harmonics: int = 4):
        self.p = p = params or SSNParams()
        self.n, self.step = n, float(step_um)
        self.theta = orientation_map_periodic(n, step_um, p.map_period_um,
                                              seed)
        d = self.dist = _periodic_distance(n, step_um)
        # The published weights are per connection on a 107 um grid.  On a
        # finer grid each cell stands for less cortex and there are more of
        # them, so each connection carries proportionally less and the total
        # input from a region of cortex is unchanged.
        area = (step_um / p.ref_step_um) ** 2

        near = (d <= p.L_o + 1e-6).astype(float)

        def far(sigma):
            return np.where(d > p.L_o + 1e-6,
                            np.exp(-(d - p.L_o) ** 2 / (2 * sigma ** 2)), 0.0)

        self.k_near = near
        self.k_far_ee, self.k_far_ie = far(p.sigma_ee), far(p.sigma_ie)
        self.k_inh = np.exp(-d ** 2 / (2 * p.sigma_i ** 2))

        cn = _ori_harmonics(p.A_near, p.B_near, p.sori_near, harmonics)
        cf = _ori_harmonics(p.A_far, p.B_far, p.sori_far, harmonics)
        self.c_near, self.c_far = cn, cf

        # Basis [1, cos2t, sin2t, cos4t, sin4t, ...]; harmonic index of each.
        basis, order = [np.ones((n, n))], [0]
        for m in range(1, harmonics + 1):
            basis += [np.cos(2 * m * self.theta), np.sin(2 * m * self.theta)]
            order += [m, m]
        self.order = np.array(order)
        self.basis = np.array(basis)

        # Harmonics the inhibitory (near-type) kernel actually needs.
        keep = np.abs(cn[self.order]) > 1e-4
        self.basis_i = self.basis[keep]
        order_i = self.order[keep]

        F = sfft.rfft2
        fn, fee, fie, fi = F(near), F(self.k_far_ee), F(self.k_far_ie), \
            F(self.k_inh)
        self.KE_E = np.array([area * (p.J_ee_near * cn[m] * fn +
                                      p.J_ee_far * cf[m] * fee)
                              for m in self.order])
        self.KE_I = np.array([area * (p.J_ie_near * cn[m] * fn +
                                      p.J_ie_far * cf[m] * fie)
                              for m in self.order])
        self.KI = np.array([area * cn[m] * fi for m in order_i])

        # Axonal footprints for the recruitment stage, normalised to unit
        # integral (per um^2) using the orientation-averaged kernel so the
        # normalisation does not depend on where the electrode sits.
        self._fp_norm_e = ((p.J_ee_near * cn[0] * near +
                            p.J_ee_far * cf[0] * self.k_far_ee).sum()
                           * step_um ** 2)
        self._fp_norm_i = self.k_inh.sum() * cn[0] * step_um ** 2

    # -- recurrent input ---------------------------------------------------

    def recurrent(self, RE: np.ndarray, RI: np.ndarray):
        """Net recurrent input to E and to I cells for presynaptic rates."""
        p, shape = self.p, (self.n, self.n)
        FE = sfft.rfft2(RE[None] * self.basis)
        FI = sfft.rfft2(RI[None] * self.basis_i)
        e_to_e = (self.basis * sfft.irfft2(FE * self.KE_E, s=shape)).sum(0)
        e_to_i = (self.basis * sfft.irfft2(FE * self.KE_I, s=shape)).sum(0)
        inh = (self.basis_i * sfft.irfft2(FI * self.KI, s=shape)).sum(0)
        return e_to_e - p.J_ei * inh, e_to_i - p.J_ii * inh

    def transfer(self, I: np.ndarray, r_max: float) -> np.ndarray:
        """Rectified power law, with a ceiling far above the visual range."""
        f = self.p.k * np.maximum(I, 0.0) ** self.p.n
        return r_max * (1.0 - np.exp(-f / r_max))

    # -- dynamics ----------------------------------------------------------

    def run(self, h_e=0.0, h_i=0.0, p_e=0.0, p_i=0.0, rho_e=0.0, rho_i=0.0,
            t_ms: float = 200.0, dt: float = 0.5, state=None,
            tol: float | None = 1e-4, return_state: bool = False):
        """Integrate to ``t_ms``, stopping early once the rates settle.

        ``h`` are feedforward inputs.  ``p`` are the fractions of each
        population driven directly by the electrode.  Those cells transmit at
        least the equivalent rate ``rho`` whatever inhibition they receive;
        direct drive is a floor rather than a ceiling, so a cell already
        firing faster for synaptic reasons is not slowed by being stimulated.
        The remaining fraction follows the network.  Returns the presynaptic
        (output) rate fields of E and I.
        """
        P = self.p
        z = np.zeros((self.n, self.n))
        rE, rI, S = (z.copy(), z.copy(), z.copy()) if state is None else \
            (state[0].copy(), state[1].copy(), state[2].copy())
        p_e, p_i = np.broadcast_to(p_e, z.shape), np.broadcast_to(p_i, z.shape)
        phi = P.slow_inh_frac
        steps = max(1, int(round(t_ms / dt)))
        for it in range(steps):
            RE = p_e * np.maximum(rho_e, rE) + (1.0 - p_e) * rE
            RI = p_i * np.maximum(rho_i, rI) + (1.0 - p_i) * rI
            if phi > 0:
                S += (RI - S) * dt / P.tau_slow
                RI_eff = (1.0 - phi) * RI + phi * S
            else:
                RI_eff = RI
            rec_e, rec_i = self.recurrent(RE, RI_eff)
            dE = (self.transfer(rec_e + h_e, P.r_max_e) - rE) * dt / P.tau_e
            dI = (self.transfer(rec_i + h_i, P.r_max_i) - rI) * dt / P.tau_i
            rE += dE
            rI += dI
            if tol is not None and phi == 0 and it > 40 and it % 10 == 0:
                if max(np.abs(dE).max(), np.abs(dI).max()) < tol * dt:
                    break
        RE = p_e * np.maximum(rho_e, rE) + (1.0 - p_e) * rE
        RI = p_i * np.maximum(rho_i, rI) + (1.0 - p_i) * rI
        if return_state:
            return RE, RI, (rE, rI, S)
        return RE, RI


# ---------------------------------------------------------------------------
# Recruitment + network = the forward model
# ---------------------------------------------------------------------------

class BiophysicalCortex:
    """Electrodes in, excitatory output field out.

    Mirrors the two methods the experiment scripts call on the earlier model
    (``_ensure_cortical_grid`` and ``_activation_field``) so the same
    experiments can be run on either.
    """

    def __init__(self, step_um: float = 107.0, margin_um: float = 3200.0,
                 pulse_hz: float = 300.0, seed: int = 0,
                 ssn: SSNParams | None = None,
                 recruit: RecruitmentParams | None = None,
                 network: bool = True, min_n: int = 96,
                 background: float = 10.0):
        self.step, self.margin = float(step_um), float(margin_um)
        self.pulse_hz, self.seed = float(pulse_hz), seed
        self.ssn_params = ssn or SSNParams()
        self.rp = recruit or RecruitmentParams()
        # ``network=False`` leaves only direct recruitment, the control that
        # isolates occupancy from recurrent interaction.
        self.network = network
        self.min_n = min_n
        # Ongoing activity.  A supralinear network at zero rate is inert:
        # its gain vanishes, weak inputs do nothing, and neither the
        # facilitation of weak inputs nor the suppression of ongoing firing
        # around an electrode can appear.  Cortex is not silent, and is
        # inhibition-stabilized even at spontaneous rates (Sanzeni et al.
        # 2020), so a uniform background input sets rates of about 1.4 Hz
        # (E) and 5 Hz (I), in line with the spontaneous rates of Obeid and
        # Miller's spiking version.  Everything the model reports is the
        # *change* from this baseline.
        self.background = float(background)
        self._baseline: dict[int, tuple] = {}
        self._sheets: dict[int, SSNSheet] = {}
        self._cgrid = None
        self.last = {}

    # -- grid --------------------------------------------------------------

    def _ensure_cortical_grid(self, x_el, y_el):
        x_el, y_el = np.asarray(x_el, float), np.asarray(y_el, float)
        span = max(np.ptp(x_el), np.ptp(y_el)) + 2 * self.margin
        n = max(self.min_n, int(np.ceil(span / self.step)))
        n = int(sfft.next_fast_len(n, real=True))
        cx = 0.5 * (x_el.min() + x_el.max())
        cy = 0.5 * (y_el.min() + y_el.max())
        key = (n, round(cx, 3), round(cy, 3))
        if self._cgrid is not None and self._cgrid["key"] == key:
            return self._cgrid
        if n not in self._sheets:
            self._sheets[n] = SSNSheet(n, self.step, self.seed,
                                       self.ssn_params)
        # Centred on a node, so a lone electrode sits on a grid point rather
        # than equidistant from four of them.
        xs = cx + (np.arange(n) - n // 2) * self.step
        ys = cy + (np.arange(n) - n // 2) * self.step
        self._cgrid = {"key": key, "xs": xs, "ys": ys, "n": n,
                       "sheet": self._sheets[n]}
        return self._cgrid

    def site_index(self, g, xe: float, ye: float) -> tuple[int, int]:
        """(row, column) of the grid node nearest to a point on cortex."""
        n = g["n"]
        i = int(np.clip(np.rint((ye - g["ys"][0]) / self.step), 0, n - 1))
        j = int(np.clip(np.rint((xe - g["xs"][0]) / self.step), 0, n - 1))
        return i, j

    # -- direct recruitment --------------------------------------------------

    def init_radius_um(self, amp_ua: float) -> float:
        return 1000.0 * np.sqrt(abs(amp_ua) / self.rp.K_ax)

    def _footprint(self, sheet: SSNSheet, g, xe, ye, kind: str) -> np.ndarray:
        """Axon-site density of cells across the sheet, seen from the tip.

        A cell at x has terminals at the tip in proportion to its outgoing
        connection kernel evaluated between x and the tip -- distance and
        orientation similarity both.  Returned per um^2, unit integral.
        """
        P = sheet.p
        X, Y = np.meshgrid(g["xs"], g["ys"])
        d = np.hypot(X - xe, Y - ye)
        # Orientation at the tip, read from the nearest site.
        i, j = self.site_index(g, xe, ye)
        dth = np.abs(sheet.theta - sheet.theta[i, j])
        dth = np.rad2deg(np.minimum(dth, np.pi - dth))

        def q(A, B, s):
            return A + B * np.exp(-dth ** 2 / (2 * s ** 2))

        if kind == "i":
            f = np.exp(-d ** 2 / (2 * P.sigma_i ** 2)) * \
                q(P.A_near, P.B_near, P.sori_near)
            return f / sheet._fp_norm_i
        near = (d <= P.L_o) * q(P.A_near, P.B_near, P.sori_near)
        far = np.where(d > P.L_o,
                       np.exp(-(d - P.L_o) ** 2 / (2 * P.sigma_ee ** 2)), 0.0) \
            * q(P.A_far, P.B_far, P.sori_far)
        return (P.J_ee_near * near + P.J_ee_far * far) / sheet._fp_norm_e

    def hits(self, amps, x_el, y_el, g):
        """Expected excitable sites inside the initiation ball, per cell."""
        sheet = g["sheet"]
        he = np.zeros((g["n"], g["n"]))
        hi = np.zeros_like(he)
        for a, xe, ye in zip(amps, x_el, y_el):
            a = abs(float(a))
            if a <= 0:
                continue
            R = self.init_radius_um(a)
            vol_mm3 = 4.0 / 3.0 * np.pi * (R / 1000.0) ** 3
            fe = self._footprint(sheet, g, xe, ye, "e") * 1e6   # per mm^2
            fi = self._footprint(sheet, g, xe, ye, "i") * 1e6
            # The ball has lateral extent of its own; each lateral coordinate
            # of a uniform ball has variance R^2 / 5.
            sb = R / np.sqrt(5.0) / self.step
            if sb > 0.3:
                from scipy.ndimage import gaussian_filter
                fe = gaussian_filter(fe, sb, mode="wrap")
                fi = gaussian_filter(fi, sb, mode="wrap")
            he += self.rp.site_density_e * vol_mm3 * fe
            hi += self.rp.site_density_i * vol_mm3 * fi
        return he, hi

    def recruited(self, amps, x_el, y_el, g):
        """Fraction of E and of I cells driven directly at each site."""
        he, hi = self.hits(amps, x_el, y_el, g)
        return 1.0 - np.exp(-he), 1.0 - np.exp(-hi)

    # -- forward model -------------------------------------------------------

    def _rho(self):
        r = self.rp
        return (effective_rate(self.pulse_hz, r.U_e, r.tau_rec_e, r.ref_rate_e),
                effective_rate(self.pulse_hz, r.U_i, r.tau_rec_i, r.ref_rate_i))

    def baseline(self, g):
        """Spontaneous state of the sheet: (rE, rI, slow-inhibition trace)."""
        n = g["n"]
        if n not in self._baseline:
            sheet, b = g["sheet"], self.background
            if not self.network or b <= 0:
                z = np.zeros((n, n))
                self._baseline[n] = (z, z.copy(), z.copy())
            else:
                # Converged far past the evoked tolerance: any residual here
                # appears in every evoked field, and summed over a large
                # sheet it is not small next to the response to 10 uA.
                rE, rI = sheet.run(b, b, t_ms=5000.0, tol=1e-9)
                self._baseline[n] = (rE, rI, rI.copy())
        return self._baseline[n]

    def _activation_field(self, amps, x_el, y_el, g, h_e=0.0, h_i=0.0):
        """Evoked excitatory output across the sheet (Hz-equivalent).

        The change from spontaneous activity, so it is negative wherever the
        electrode suppresses ongoing firing.  Directly driven cells stop
        firing spontaneously and fire at the pulse rate instead.
        """
        sheet = g["sheet"]
        pe, pi = self.recruited(amps, x_el, y_el, g)
        rho_e, rho_i = self._rho()
        r0e, r0i, s0 = self.baseline(g)
        if not self.network:
            rE, RE, RI = r0e, pe * rho_e, pi * rho_i
        else:
            b = self.background
            # Tolerance matters more than it looks.  A residual of 1e-3 Hz
            # per site, which is what tol = 1e-4 leaves, sums over ~10^4
            # sites to a spurious 1% "interaction" between electrodes 4.5 mm
            # apart at 10 uA.  At 1e-7 the same pair reads 0.02%.
            RE, RI, st = sheet.run(b + h_e, b + h_i, pe, pi, rho_e, rho_i,
                                   state=(r0e, r0i, s0), t_ms=3000.0,
                                   tol=1e-7, return_state=True)
            rE = st[0]
        self.last = {"p_e": pe, "p_i": pi, "R_e": RE, "R_i": RI,
                     "direct_e": pe * (rho_e - r0e),
                     "indirect_e": (1.0 - pe) * (rE - r0e),
                     "baseline_e": r0e, "baseline_i": r0i}
        return RE - r0e

    def simulate(self, schedule, x_el, y_el, g, t_end: float, dt: float = 0.5,
                 record_every: float = 1.0):
        """Time course for a stimulation schedule.

        ``schedule`` is a list of (t_on, t_off, amplitudes).  Returns times and
        the evoked excitatory output at each electrode's own site, which is
        what a report of that phosphene's brightness would track.
        """
        sheet = g["sheet"]
        rho_e, rho_i = self._rho()
        b = self.background if self.network else 0.0
        r0e = self.baseline(g)[0]
        idx = [self.site_index(g, xe, ye) for xe, ye in zip(x_el, y_el)]
        cache = {}
        state, t, ts, out = tuple(a.copy() for a in self.baseline(g)), 0.0, \
            [], []
        z = np.zeros((g["n"], g["n"]))
        while t < t_end - 1e-9:
            amps = None
            for t0, t1, a in schedule:
                if t0 <= t < t1:
                    amps = tuple(float(v) for v in a)
                    break
            if amps is None:
                pe, pi = z, z
            else:
                if amps not in cache:
                    cache[amps] = self.recruited(amps, x_el, y_el, g)
                pe, pi = cache[amps]
            RE, RI, state = sheet.run(b, b, pe, pi, rho_e, rho_i,
                                      t_ms=record_every, dt=dt, state=state,
                                      tol=None, return_state=True)
            t += record_every
            ts.append(t)
            out.append([RE[i, j] - r0e[i, j] for i, j in idx])
        return np.array(ts), np.array(out)
