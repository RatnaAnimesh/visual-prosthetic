"""Turning cortical activity into a predicted percept.

Every model in this project so far has stopped at cortical activation and then
*assumed* the percept is that activation painted back through the retinotopic
map.  That assumption is doing real work, and it is not obviously right: it
says a cortical site's contribution to what you see depends only on where it
sits in the map, and not at all on what it is tuned for.

This module replaces the assumption with something fitted.  The idea:

  1. Run the *normal vision* pathway forwards.  A V1 site is an oriented
     filter -- it has a receptive-field position from the retinotopy, an
     orientation preference from the orientation map, and a spatial phase.
     Show it images and record what the population does.
  2. Fit the inverse: given a population response, which image produced it?
     That is a well-posed linear problem and ridge regression solves it.
  3. Apply that same decoder to *stimulation-evoked* activity.  The predicted
     percept is then "the image that would have produced this cortical
     activity", which is a statement the model can actually support.

The interesting part is step 3.  Visual input drives a site only if the
stimulus matches its orientation and phase; an electrode drives every site
near it regardless of tuning.  A decoder trained on natural vision has never
seen the second kind of pattern, and what it makes of it is a prediction about
what electrical stimulation looks like.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from pulse2percept.topography import Polimeni2006Map

from prosthesis.cortex_circuit import COLUMN_SPACING_UM, orientation_map

# V1 at ~2 degrees eccentricity: receptive fields a few tenths of a degree
# across.  The preferred spatial frequency is set at the low end of the
# physiological range for a self-consistent reason: a full set of orientations
# and phases occupies a whole hypercolumn (~750 um, ~0.18 deg here), so the
# population code cannot resolve the image more finely than that, and cannot
# faithfully represent spatial frequencies above the corresponding Nyquist
# limit either.  Asking it to carry 3 cycles/deg measurably degrades the
# reconstruction (R^2 0.57 against 0.65).
PREF_SF_CPD = 1.5
RF_SIGMA_DEG = 0.12
N_ORIENT = 8
N_PHASE = 4


def pink_noise(n, size, rng, beta=1.2):
    """Images with natural spatial statistics (1/f^beta amplitude spectrum).

    A stand-in for a natural image set.  What matters for fitting a decoder is
    that the training images have roughly the right correlation structure --
    natural scenes are dominated by low frequencies, and a decoder trained on
    white noise would waste its capacity on detail that never occurs.
    """
    f = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(f, f)
    r = np.sqrt(fx ** 2 + fy ** 2)
    r[0, 0] = 1e-6
    amp = 1.0 / r ** (beta / 2.0)
    amp[0, 0] = 0.0

    out = np.empty((n, size, size))
    for i in range(n):
        ph = rng.uniform(0, 2 * np.pi, (size, size))
        img = np.real(np.fft.ifft2(amp * np.exp(1j * ph)))
        s = img.std()
        out[i] = img / s if s > 0 else img
    return out


def gabor_bank(size, deg_per_px, sf=PREF_SF_CPD, sigma=RF_SIGMA_DEG,
               n_orient=N_ORIENT, n_phase=N_PHASE):
    """Quadrature-ish Gabor filters: n_orient orientations x n_phase phases."""
    half = int(np.ceil(3 * sigma / deg_per_px))
    ax = np.arange(-half, half + 1) * deg_per_px
    X, Y = np.meshgrid(ax, ax)

    thetas = np.linspace(0, 180, n_orient, endpoint=False)
    phases = np.linspace(0, 2 * np.pi, n_phase, endpoint=False)
    filters, labels = [], []
    for t in thetas:
        th = np.deg2rad(t)
        across = X * np.sin(th) + Y * np.cos(th)
        along = X * np.cos(th) - Y * np.sin(th)
        env = np.exp(-(across ** 2 + along ** 2) / (2 * sigma ** 2))
        for p in phases:
            g = env * np.cos(2 * np.pi * sf * across + p)
            g = g - g.mean()
            n = np.linalg.norm(g)
            filters.append(g / n if n > 0 else g)
            labels.append((t, p))
    return np.array(filters), thetas, phases, labels


class CorticalCode:
    """A sheet of V1 sites, each with a position, an orientation and a phase.

    The sheet is the bridge between the two halves of the project: it is
    sampled on cortex (where the circuit model and the electrodes live) but
    every site also knows its receptive field in the visual field (where
    images and percepts live).
    """

    def __init__(self, x_range_um, y_range_um, step_um=100.0,
                 patch_deg=(-3.0, -0.8, 0.0, 2.2), n_px=16, seed=0,
                 theta_at=None):
        """``theta_at``, if given, maps cortical (X, Y) in um to preferred
        orientation in degrees.  The decoder has to read activity with the
        orientation map of the model that produced it, so a model with its
        own map passes it here.  Left as None, the sites take the circuit
        model's map, as before.
        """
        self.step_um = step_um
        self.xs = np.arange(x_range_um[0], x_range_um[1] + 1, step_um)
        self.ys = np.arange(y_range_um[0], y_range_um[1] + 1, step_um)
        X, Y = np.meshgrid(self.xs, self.ys)
        self.shape_ctx = X.shape

        if theta_at is None:
            # Orientation preference, from the same generator the circuit uses.
            theta = orientation_map(X.shape, step_um, COLUMN_SPACING_UM,
                                    seed=seed)
        else:
            theta = np.asarray(theta_at(X, Y), dtype=float) % 180.0
        rng = np.random.default_rng(seed + 1)
        # Spatial phase is scattered within a column, as in real V1 -- and it
        # is what makes the code invertible: an orientation-only code discards
        # phase and no decoder could recover the image from it.
        phase = rng.uniform(0, 2 * np.pi, X.shape)

        # Receptive-field centres, via the same retinotopy the implant uses.
        vfmap = Polimeni2006Map()
        vx, vy = vfmap.to_dva()["v1"](X.ravel(), Y.ravel())

        self.x0, self.x1, self.y0, self.y1 = patch_deg
        self.n_px = n_px
        self.deg_per_px = (self.x1 - self.x0) / (n_px - 1)

        col = (vx - self.x0) / self.deg_per_px
        row = (vy - self.y0) / ((self.y1 - self.y0) / (n_px - 1))
        keep = (np.isfinite(col) & np.isfinite(row) &
                (col >= 0) & (col <= n_px - 1) &
                (row >= 0) & (row <= n_px - 1))

        self.ctx_x = X.ravel()[keep]
        self.ctx_y = Y.ravel()[keep]
        self.theta = theta.ravel()[keep]
        self.phase = phase.ravel()[keep]
        self.rf_col = col[keep]
        self.rf_row = row[keep]
        self.n_sites = int(keep.sum())

        filters, thetas, phases, _ = gabor_bank(n_px, self.deg_per_px)
        self.filters = filters
        self._thetas, self._phases = thetas, phases
        # Assign each site to its nearest filter channel.
        dt = np.abs(self.theta[:, None] - thetas[None, :]) % 180.0
        dt = np.minimum(dt, 180.0 - dt)
        ti = np.argmin(dt, axis=1)
        dp = np.abs(self.phase[:, None] - phases[None, :]) % (2 * np.pi)
        dp = np.minimum(dp, 2 * np.pi - dp)
        pi_ = np.argmin(dp, axis=1)
        self.channel = ti * len(phases) + pi_

        self._build_forward()

    # -- forward: images -> cortical activity ------------------------------

    def _build_forward(self):
        """The forward operator M: one row per site, giving that site's filter
        translated to its receptive field.

        Encoding is defined *through* this matrix rather than by convolving
        separately.  Keeping one operator removes a whole class of bug: a
        convolution flips its kernel where a dot product does not, and an
        encoder and an inverse built independently will silently disagree by
        exactly that flip.
        """
        n_px = self.n_px
        M = np.zeros((self.n_sites, n_px * n_px))
        for i in range(self.n_sites):
            f = self.filters[self.channel[i]]
            h = f.shape[0] // 2
            r = int(round(self.rf_row[i]))
            c = int(round(self.rf_col[i]))
            r0, r1 = max(0, r - h), min(n_px, r + h + 1)
            c0, c1 = max(0, c - h), min(n_px, c + h + 1)
            if r1 <= r0 or c1 <= c0:
                continue
            patch = np.zeros((n_px, n_px))
            patch[r0:r1, c0:c1] = f[h - (r - r0):h + (r1 - r),
                                    h - (c - c0):h + (c1 - c)]
            M[i] = patch.ravel()
        self.M = M

    def encode(self, images):
        """Population response: half-wave rectified filter responses.

        Rectification matters -- real neurons cannot fire negatively, and it
        is the rectified response an electrode's activation must be compared
        against.
        """
        images = np.asarray(images, dtype=float)
        if images.ndim == 2:
            images = images[None]
        flat = images.reshape(len(images), -1)
        return np.maximum(flat @ self.M.T, 0.0)

    # -- sampling a cortical field onto the sites --------------------------

    def sample_field(self, field, xs, ys):
        """Read a circuit-model activation field out at each site."""
        step = xs[1] - xs[0]
        col = (self.ctx_x - xs[0]) / step
        row = (self.ctx_y - ys[0]) / (ys[1] - ys[0])
        inside = ((col >= 0) & (col <= len(xs) - 1) &
                  (row >= 0) & (row <= len(ys) - 1))
        out = np.zeros(self.n_sites)
        if inside.any():
            out[inside] = map_coordinates(field, [row[inside], col[inside]],
                                          order=1, mode="constant")
        return out


class InversePercept:
    """Find the image that best explains a population response.

    Ridge regression learns E[image | activity] *under the distribution it was
    trained on*.  Applied to electrical stimulation that expectation is
    meaningless, because stimulation drives every site near an electrode
    regardless of what it is tuned for, and no natural image produces such a
    pattern.  Fitting a regression and extrapolating gives structured speckle
    rather than anything a person could plausibly report seeing.

    So instead of regressing, solve the inverse problem directly:

        minimise  || M I - a ||^2  +  lambda || grad I ||^2

    where M is the forward filter bank.  This asks "which image would have
    produced this activity", which is a well-posed question whatever the
    activity is, and the smoothness prior stands in for the fact that percepts
    are not white noise.  It also has only n_px^2 unknowns rather than one per
    cortical site, so it is both better conditioned and much smaller.
    """

    def __init__(self, code: CorticalCode, lam=0.05, iters=4,
                 precompute=False):
        self.code = code
        self.lam = lam
        self.iters = iters
        self.M = code.M
        self._fixed = None

        n_px, n_pix = code.n_px, code.n_px ** 2
        idx = np.arange(n_pix).reshape(n_px, n_px)
        rows = []
        for a, b in ((idx[:, :-1], idx[:, 1:]), (idx[:-1], idx[1:])):
            for p, q in zip(a.ravel(), b.ravel()):
                r = np.zeros(n_pix); r[p], r[q] = 1.0, -1.0
                rows.append(r)
        L = np.array(rows)
        self.LtL = L.T @ L
        self.scale = np.trace(self.M.T @ self.M) / max(len(L), 1)
        if precompute:
            self.precompute()

    def precompute(self):
        """Collapse the inverse to one fixed matrix.

        Refining the active set turns out to buy nothing -- R^2 is 0.660 at one
        iteration and 0.661 at four -- so for sweeps the whole decode can be a
        single matmul against a precomputed operator.  It also makes the
        readout exactly linear, which matters here: a nonlinear readout could
        manufacture electrode interaction of its own and confound the very
        thing being measured.
        """
        M = self.M
        A = M.T @ M + self.lam * self.scale * self.LtL
        self._fixed = np.linalg.solve(A, M.T)
        return self

    def _solve_one(self, a):
        """Least squares over the sites the model believes are active.

        Rectification means a silent site says only "my filter response was
        <= 0", which is an inequality, not an equation.  Fitting those as
        equalities biases the whole reconstruction, so they are dropped and
        the active set is refined a few times.
        """
        M, n_pix = self.M, self.code.n_px ** 2
        active = a > 1e-12
        img = np.zeros(n_pix)
        for _ in range(self.iters):
            if not active.any():
                break
            Ms, asub = M[active], a[active]
            A = Ms.T @ Ms + self.lam * self.scale * self.LtL
            img = np.linalg.solve(A, Ms.T @ asub)
            active = (a > 1e-12) | (M @ img > 0)
        return img

    def decode(self, activity):
        a = np.atleast_2d(np.asarray(activity, dtype=float))
        if self._fixed is not None:
            out = a @ self._fixed.T
        else:
            out = np.array([self._solve_one(row) for row in a])
        return out.reshape(-1, self.code.n_px, self.code.n_px)

    def round_trip(self, images):
        return self.decode(self.code.encode(images))


class PerceptDecoder:
    """Ridge decoder mapping population activity back to an image.

    Kept for comparison.  See :class:`InversePercept` for why it should not be
    used to interpret stimulation.
    """

    def __init__(self, code: CorticalCode, alpha=0.1):
        self.code = code
        self.alpha = alpha
        self.W = None
        self.b = None

    def fit(self, n_train=3000, seed=0, batch=250):
        rng = np.random.default_rng(seed)
        n_px = self.code.n_px
        XtX = np.zeros((self.code.n_sites, self.code.n_sites))
        XtY = np.zeros((self.code.n_sites, n_px * n_px))
        x_sum = np.zeros(self.code.n_sites)
        y_sum = np.zeros(n_px * n_px)
        n = 0
        for _ in range(0, n_train, batch):
            k = min(batch, n_train - n)
            imgs = pink_noise(k, n_px, rng)
            A = self.code.encode(imgs)
            Y = imgs.reshape(k, -1)
            XtX += A.T @ A
            XtY += A.T @ Y
            x_sum += A.sum(0)
            y_sum += Y.sum(0)
            n += k

        xm, ym = x_sum / n, y_sum / n
        XtX -= n * np.outer(xm, xm)
        XtY -= n * np.outer(xm, ym)
        XtX[np.diag_indices_from(XtX)] += self.alpha * n
        self.W = np.linalg.solve(XtX, XtY)
        self.b = ym - xm @ self.W
        self._xm = xm
        return self

    def decode(self, activity):
        a = np.atleast_2d(np.asarray(activity, dtype=float))
        img = a @ self.W + self.b
        return img.reshape(-1, self.code.n_px, self.code.n_px)

    def round_trip(self, images):
        """Encode then decode; the honest test of whether the code inverts."""
        return self.decode(self.code.encode(images))


def reconstruction_r2(truth, recon):
    t = np.asarray(truth).reshape(len(np.atleast_3d(truth)), -1) \
        if np.ndim(truth) == 3 else np.asarray(truth).reshape(1, -1)
    r = np.asarray(recon).reshape(t.shape)
    ss_res = ((t - r) ** 2).sum(axis=1)
    ss_tot = ((t - t.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    return float(np.mean(1.0 - ss_res / np.maximum(ss_tot, 1e-12)))
