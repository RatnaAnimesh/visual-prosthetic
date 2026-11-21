"""Retina and lateral geniculate nucleus: the input the column has to work with.

Kandel Fig. 25-9: neurons at the early relays have concentric receptive fields
with mutually antagonistic centre and surround.  They respond poorly to
uniform light covering both regions and optimally to *contrast* -- a light-dark
boundary.  That is a difference-of-Gaussians spatial filter followed by a
biphasic temporal filter, which is what this module implements.

Fig. 25-12: the nucleus is laminated, with two magnocellular layers (1-2) and
four parvocellular layers (3-6), and the two eyes terminate in different
layers -- contralateral to 1, 4, 6 and ipsilateral to 2, 3, 5.  The M and P
streams stay separate all the way into cortex, terminating in layers IVC-alpha
and IVC-beta respectively.  The model keeps eye, stream, and sign (ON/OFF) as
independent labels on every geniculate cell so that the cortical wiring can
respect that segregation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .stimuli import Dichoptic, RetinalGrid, Stimulus

# Stream parameters.  P cells: small centres, sustained responses, low contrast
# gain, and they are far more numerous.  M cells: larger centres, transient
# responses, high contrast gain and early saturation.
MAGNO = dict(
    sigma_c=0.10, surround_ratio=3.0, tau_c=8.0, tau_s=20.0,
    transience=0.85, c50=0.10, r_max=160.0, r_base=12.0, name="M",
)
PARVO = dict(
    sigma_c=0.045, surround_ratio=3.5, tau_c=16.0, tau_s=45.0,
    transience=0.45, c50=0.38, r_max=110.0, r_base=8.0, name="P",
)


@dataclass
class LGNPopulation:
    """A labelled sheet of geniculate relay cells.

    One population = one (eye, stream, sign) combination, tiled retinotopically
    over the grid.  Separate populations are what makes ocular-dominance and
    M/P segregation expressible in the cortical wiring.
    """

    eye: int  # 0 = left (ipsilateral), 1 = right (contralateral)
    stream: str  # 'M' or 'P'
    sign: int  # +1 = ON centre, -1 = OFF centre
    x: np.ndarray = field(repr=False, default=None)
    y: np.ndarray = field(repr=False, default=None)
    start: int = 0

    @property
    def n(self) -> int:
        return len(self.x)

    @property
    def idx(self) -> np.ndarray:
        return np.arange(self.start, self.start + self.n)


class LGN:
    """The full geniculate input array plus its spike generator."""

    def __init__(self, grid: RetinalGrid, spacing_deg: float = 0.11,
                 rng: np.random.Generator | None = None,
                 jitter_deg: float = 0.012, seed: int = 1):
        self.grid = grid
        self.rng = rng or np.random.default_rng(seed)
        self.populations: list[LGNPopulation] = []
        self.n = 0

        # Tile each labelled sheet over the patch.  M cells are sparser than P
        # cells, as in the retina, where parasol cells are far outnumbered by
        # midget cells.
        for stream, sp_mult in (("M", 1.8), ("P", 1.0)):
            step = spacing_deg * sp_mult
            half = grid.size_deg / 2 - step / 2
            ax = np.arange(-half, half + 1e-9, step)
            gx, gy = np.meshgrid(ax, ax)
            base_x, base_y = gx.ravel(), gy.ravel()
            for eye in (0, 1):
                for sign in (+1, -1):
                    jx = self.rng.normal(0, jitter_deg, base_x.size)
                    jy = self.rng.normal(0, jitter_deg, base_y.size)
                    pop = LGNPopulation(
                        eye=eye, stream=stream, sign=sign,
                        x=base_x + jx, y=base_y + jy, start=self.n,
                    )
                    self.populations.append(pop)
                    self.n += pop.n

        self.eye = np.concatenate([np.full(p.n, p.eye) for p in self.populations])
        self.stream = np.concatenate(
            [np.full(p.n, p.stream) for p in self.populations]
        )
        self.sign = np.concatenate([np.full(p.n, p.sign) for p in self.populations])
        self.x = np.concatenate([p.x for p in self.populations])
        self.y = np.concatenate([p.y for p in self.populations])

        self._build_kernels()
        self._reset_state()

    # -- spatial filtering --------------------------------------------------

    def _build_kernels(self) -> None:
        """Prepare the difference-of-Gaussians filtering.

        Every relay cell of a given stream has the same receptive-field
        *shape* and differs only in where it sits, so the whole sheet is one
        convolution followed by a sampling step rather than one dot product
        per cell.  Centre and surround are normalised so a spatially uniform
        field almost cancels -- the formal statement of Kandel's observation
        that these cells respond weakly when light covers both centre and
        surround, and respond optimally to a light-dark boundary.

        The surround is left slightly weaker than the centre; real cells are
        not perfectly balanced, and the residual keeps them responsive to
        large uniform patches.
        """
        n_pix = self.grid.n_pix
        dpp = self.grid.deg_per_pix
        half = self.grid.size_deg / 2.0

        self._filters = {}
        for stream, par in (("M", MAGNO), ("P", PARVO)):
            self._filters[stream] = dict(
                sigma_c_px=par["sigma_c"] / dpp,
                sigma_s_px=par["sigma_c"] * par["surround_ratio"] / dpp,
            )

        # Fractional pixel coordinates at which each cell samples the filtered
        # image, so positional jitter is preserved exactly.
        self._row = (self.y + half) / dpp
        self._col = (self.x + half) / dpp
        np.clip(self._row, 0, n_pix - 1, out=self._row)
        np.clip(self._col, 0, n_pix - 1, out=self._col)
        self._stream_mask = {s: self.stream == s for s in ("M", "P")}

        self._params = {
            k: np.array([
                (MAGNO if s == "M" else PARVO)[k] for s in self.stream
            ])
            for k in ("tau_c", "tau_s", "transience", "c50", "r_max", "r_base")
        }

    def _dog_sample(self, img: np.ndarray, out: np.ndarray) -> None:
        """Filter ``img`` for each stream and sample it at every cell."""
        from scipy.ndimage import gaussian_filter, map_coordinates

        for stream, f in self._filters.items():
            m = self._stream_mask[stream]
            if not m.any():
                continue
            centre = gaussian_filter(img, f["sigma_c_px"], mode="nearest")
            surround = gaussian_filter(img, f["sigma_s_px"], mode="nearest")
            dog = centre - 0.9 * surround
            out[m] = map_coordinates(
                dog, [self._row[m], self._col[m]], order=1, mode="nearest"
            )
        out *= self.sign

    # -- temporal filtering + spiking --------------------------------------

    def _reset_state(self) -> None:
        self.y_fast = np.zeros(self.n)
        self.y_slow = np.zeros(self.n)
        self.rate = self._params["r_base"].copy()
        self._last_frame_t = -1e9
        self._drive = np.zeros(self.n)

    def reset(self) -> None:
        self._reset_state()

    def _stimulus_drive(self, stim: Stimulus, t_ms: float) -> np.ndarray:
        """Instantaneous linear drive of every relay cell, before dynamics."""
        drive = np.zeros(self.n)
        if isinstance(stim, Dichoptic):
            for eye in (0, 1):
                img = stim.eye_image(self.grid, t_ms, eye)
                tmp = np.zeros(self.n)
                self._dog_sample(img, tmp)
                drive = np.where(self.eye == eye, tmp, drive)
        else:
            self._dog_sample(stim.image(self.grid, t_ms), drive)
        return drive

    def step(self, stim: Stimulus, t_ms: float, dt: float,
             frame_ms: float = 1.0) -> np.ndarray:
        """Advance the geniculate model one dt and return spiking cell indices.

        The temporal filter is a lagged difference of two low-pass stages.
        Subtracting the slower stage produces the biphasic impulse response
        that makes these cells transient: the M stream, with a larger
        ``transience``, responds mainly to change, while the more sustained P
        stream keeps signalling a standing contrast.
        """
        if t_ms - self._last_frame_t >= frame_ms:
            self._drive = self._stimulus_drive(stim, t_ms)
            self._last_frame_t = t_ms

        p = self._params
        self.y_fast += (self._drive - self.y_fast) * dt / p["tau_c"]
        self.y_slow += (self.y_fast - self.y_slow) * dt / p["tau_s"]
        signal = self.y_fast - p["transience"] * self.y_slow

        # Half-wave rectification plus Naka-Rushton saturation: a relay cell
        # cannot fire below zero, and its contrast response saturates.
        pos = np.maximum(signal, 0.0)
        n_exp = 1.5
        sat = pos ** n_exp / (pos ** n_exp + p["c50"] ** n_exp)
        self.rate = p["r_base"] + p["r_max"] * sat

        lam = self.rate * dt / 1000.0
        return np.flatnonzero(self.rng.random(self.n) < lam)

    # -- selection helpers used when wiring the cortex ----------------------

    def select(self, eye: int | None = None, stream: str | None = None,
               sign: int | None = None) -> np.ndarray:
        m = np.ones(self.n, dtype=bool)
        if eye is not None:
            m &= self.eye == eye
        if stream is not None:
            m &= self.stream == stream
        if sign is not None:
            m &= self.sign == sign
        return np.flatnonzero(m)

    def summary(self) -> str:
        rows = [
            f"  {p.stream} eye{p.eye} {'ON ' if p.sign > 0 else 'OFF'}: "
            f"{p.n:4d} cells"
            for p in self.populations
        ]
        return f"LGN: {self.n} relay cells\n" + "\n".join(rows)
