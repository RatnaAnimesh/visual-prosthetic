"""Rules for building the synaptic weight matrices.

Cortical wiring is not random, but neither is it a wiring diagram: it is a set
of probabilistic rules whose parameters depend on the pre- and postsynaptic
cell class, the cortical distance between the cells, and -- for the horizontal
connections of Kandel Fig. 25-16 -- on how similar the two cells' functional
preferences are.  Each function here returns a sparse (n_pre, n_post) matrix
of peak conductances in nS.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _lognormal_weights(rng: np.random.Generator, n: int, mean: float,
                       cv: float = 0.7) -> np.ndarray:
    """Synaptic strengths are lognormally distributed, not Gaussian.

    Cortical unitary EPSP amplitudes span two orders of magnitude with a long
    right tail: most connections are weak and a few are very strong.  The
    heavy tail matters -- it is what lets a small number of strong inputs
    dominate a cell's response.
    """
    if mean <= 0:
        return np.zeros(n)
    sigma = np.sqrt(np.log(1.0 + cv ** 2))
    mu = np.log(mean) - sigma ** 2 / 2
    return rng.lognormal(mu, sigma, n)


def random_connect(n_pre: int, n_post: int, p: float, weight: float,
                   rng: np.random.Generator, cv: float = 0.7,
                   allow_self: bool = True) -> sp.coo_matrix:
    """Bernoulli connectivity with lognormal weights."""
    if p <= 0 or weight <= 0:
        return sp.coo_matrix((n_pre, n_post))
    n_syn = rng.binomial(n_pre * n_post, p)
    if n_syn == 0:
        return sp.coo_matrix((n_pre, n_post))
    flat = rng.choice(n_pre * n_post, size=n_syn, replace=False)
    rows, cols = np.divmod(flat, n_post)
    if not allow_self:
        keep = rows != cols
        rows, cols = rows[keep], cols[keep]
    w = _lognormal_weights(rng, rows.size, weight, cv)
    return sp.coo_matrix((w, (rows, cols)), shape=(n_pre, n_post))


def distance_connect(pre_xy: np.ndarray, post_xy: np.ndarray, p_max: float,
                     sigma_um: float, weight: float,
                     rng: np.random.Generator, cv: float = 0.7,
                     allow_self: bool = False) -> sp.coo_matrix:
    """Gaussian falloff of connection probability with cortical distance."""
    if p_max <= 0 or weight <= 0:
        return sp.coo_matrix((len(pre_xy), len(post_xy)))
    d2 = (
        (pre_xy[:, None, 0] - post_xy[None, :, 0]) ** 2
        + (pre_xy[:, None, 1] - post_xy[None, :, 1]) ** 2
    )
    p = p_max * np.exp(-d2 / (2 * sigma_um ** 2))
    if not allow_self and len(pre_xy) == len(post_xy):
        np.fill_diagonal(p, 0.0)
    mask = rng.random(p.shape) < p
    rows, cols = np.nonzero(mask)
    w = _lognormal_weights(rng, rows.size, weight, cv)
    return sp.coo_matrix((w, (rows, cols)), shape=p.shape)


def orientation_tuned_connect(pre_theta: np.ndarray, post_theta: np.ndarray,
                              pre_xy: np.ndarray, post_xy: np.ndarray,
                              p_max: float, sigma_um: float,
                              sigma_theta_deg: float, weight: float,
                              rng: np.random.Generator, cv: float = 0.7,
                              allow_self: bool = False) -> sp.coo_matrix:
    """Long-range horizontal connections that link like to like.

    Gilbert and Wiesel's result (Kandel Fig. 25-16): the axon collaterals of
    superficial pyramidal cells run for millimetres parallel to the cortical
    surface and form *clustered* terminations in columns of similar
    orientation preference.  Probability here is the product of a distance
    term and an orientation-similarity term, with orientation treated modulo
    180 degrees because a bar and its 180-degree rotation are the same bar.
    """
    if p_max <= 0 or weight <= 0:
        return sp.coo_matrix((len(pre_theta), len(post_theta)))
    d2 = (
        (pre_xy[:, None, 0] - post_xy[None, :, 0]) ** 2
        + (pre_xy[:, None, 1] - post_xy[None, :, 1]) ** 2
    )
    dtheta = circular_orientation_difference(
        pre_theta[:, None], post_theta[None, :]
    )
    p = (
        p_max
        * np.exp(-d2 / (2 * sigma_um ** 2))
        * np.exp(-dtheta ** 2 / (2 * sigma_theta_deg ** 2))
    )
    if not allow_self and len(pre_theta) == len(post_theta):
        np.fill_diagonal(p, 0.0)
    mask = rng.random(p.shape) < p
    rows, cols = np.nonzero(mask)
    w = _lognormal_weights(rng, rows.size, weight, cv)
    return sp.coo_matrix((w, (rows, cols)), shape=p.shape)


def phase_opposed_connect(pre_phase: np.ndarray, post_phase: np.ndarray,
                          pre_theta: np.ndarray, post_theta: np.ndarray,
                          p_max: float, sigma_theta_deg: float,
                          weight: float, rng: np.random.Generator,
                          opposition: float = 0.85,
                          cv: float = 0.6) -> sp.coo_matrix:
    """Push-pull wiring: inhibition arrives in antiphase with excitation.

    A layer IV simple cell has an ON subregion that is excited by light and
    *inhibited* by dark, and an OFF subregion with the opposite arrangement.
    Anatomically that means the inhibitory interneurons contacting a given
    spiny stellate cell tend to have receptive fields of the same orientation
    but opposite spatial phase.

    This matters more than it might appear.  A drifting grating raises the
    mean geniculate drive by the same amount at every orientation -- the
    tuning lives entirely in the *modulated* component.  Push-pull inhibition
    cancels that untuned mean and doubles the modulation, because excitation
    and inhibition arrive half a cycle apart.  Without it, a feedforward model
    inherits a large orientation-independent drive that swamps its own
    selectivity.

    ``opposition`` sets how strongly the wiring prefers antiphase partners;
    0 gives phase-independent connectivity, 1 makes it strictly antiphase.
    """
    if p_max <= 0 or weight <= 0:
        return sp.coo_matrix((len(pre_phase), len(post_phase)))
    dphi = pre_phase[:, None] - post_phase[None, :]
    # (1 - cos) / 2 is 1 at antiphase and 0 at identical phase.
    phase_term = (1.0 - opposition * np.cos(dphi)) / (1.0 + opposition)
    dtheta = circular_orientation_difference(
        pre_theta[:, None], post_theta[None, :]
    )
    ori_term = np.exp(-dtheta ** 2 / (2 * sigma_theta_deg ** 2))
    p = p_max * phase_term * ori_term
    mask = rng.random(p.shape) < p
    rows, cols = np.nonzero(mask)
    w = _lognormal_weights(rng, rows.size, weight, cv)
    return sp.coo_matrix((w, (rows, cols)), shape=p.shape)


def circular_orientation_difference(a_deg, b_deg):
    """Smallest angle between two orientations, in degrees, modulo 180."""
    d = np.abs(a_deg - b_deg) % 180.0
    return np.minimum(d, 180.0 - d)


def gabor(x: np.ndarray, y: np.ndarray, x0: float, y0: float,
          theta_deg: float, sf_cpd: float, phase_rad: float,
          sigma_across: float, sigma_along: float) -> np.ndarray:
    """A Gabor receptive field sampled at retinal positions (x, y).

    ``theta_deg`` is the orientation of the bars, so the modulation runs along
    the perpendicular axis and the envelope is elongated parallel to them.
    This is the shape a row of geniculate centres has to tile in order to
    build the Hubel-Wiesel simple cell.
    """
    th = np.deg2rad(theta_deg)
    dx, dy = x - x0, y - y0
    across = dx * np.sin(th) + dy * np.cos(th)
    along = dx * np.cos(th) - dy * np.sin(th)
    envelope = np.exp(
        -(across ** 2) / (2 * sigma_across ** 2)
        - (along ** 2) / (2 * sigma_along ** 2)
    )
    return envelope * np.cos(2 * np.pi * sf_cpd * across + phase_rad)


def geniculate_to_simple_cell(lgn_x: np.ndarray, lgn_y: np.ndarray,
                              lgn_sign: np.ndarray, cell_params: dict,
                              weight: float, rng: np.random.Generator,
                              min_frac: float = 0.12,
                              cv: float = 0.5) -> np.ndarray:
    """Weights from a labelled set of relay cells onto one simple cell.

    The Hubel-Wiesel construction: an ON-centre relay cell is recruited where
    the target's receptive field has an ON subregion, an OFF-centre cell where
    it has an OFF subregion.  Taking ``sign * gabor`` and half-wave rectifying
    does exactly that, and the elongation of the Gabor envelope along the
    preferred orientation is what converts a set of concentric,
    orientation-blind inputs into an orientation-selective output.

    Weak contacts below ``min_frac`` of the maximum are dropped, so each
    cortical cell is driven by a countable set of relay cells rather than by
    a smear over the whole array.
    """
    g = gabor(
        lgn_x, lgn_y, cell_params["x0"], cell_params["y0"],
        cell_params["theta"], cell_params["sf"], cell_params["phase"],
        cell_params["sigma_across"], cell_params["sigma_along"],
    )
    drive = np.maximum(lgn_sign * g, 0.0)
    if drive.max() <= 0:
        return np.zeros_like(drive)
    drive = drive / drive.max()
    drive[drive < min_frac] = 0.0
    w = drive * weight
    live = w > 0
    if live.any():
        w[live] *= _lognormal_weights(rng, int(live.sum()), 1.0, cv)
    return w


def column_positions(n_columns: int, radius_um: float = 250.0
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Lay orientation columns out as a pinwheel on the cortical surface.

    Kandel Fig. 25-11C: orientation preference cycles through a full 180
    degrees around a singularity, and one complete cycle -- a hypercolumn --
    repeats roughly every 750 micrometres.  Going once around the pinwheel in
    cortical space therefore advances the preferred orientation by 180
    degrees, not 360.
    """
    k = np.arange(n_columns)
    angle = 2 * np.pi * k / n_columns
    x = radius_um * np.cos(angle)
    y = radius_um * np.sin(angle)
    theta = 180.0 * k / n_columns
    return np.stack([x, y], axis=1), theta


def ocular_dominance_index(x_um: np.ndarray, period_um: float = 750.0,
                           sharpness: float = 2.2) -> np.ndarray:
    """Ocular-dominance index in [-1, 1] as a function of cortical position.

    Ocular-dominance columns are stripes roughly 750 micrometres wide that
    crisscross the orientation columns (Kandel Fig. 25-11B), so a single
    hypercolumn spans about one full left/right cycle.  ``sharpness`` controls
    how squared-off the stripes are; real bands are closer to square than to
    sinusoidal in layer IVC and blur out in the superficial layers.
    """
    s = np.sin(2 * np.pi * x_um / period_um)
    return np.tanh(sharpness * s)
