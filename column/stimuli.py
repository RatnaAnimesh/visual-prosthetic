"""Visual stimuli, rendered as luminance-contrast images on a retinal patch.

Everything is expressed in degrees of visual angle on a square patch centred
on the receptive-field position of the modelled column.  Contrast is Michelson
contrast in [0, 1]; the image returned by each stimulus is the deviation from
mean luminance, so 0 is grey and +-1 is full white/black.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RetinalGrid:
    """A square sampling grid over the visual field, in degrees."""

    size_deg: float = 2.0
    n_pix: int = 40

    def __post_init__(self):
        half = self.size_deg / 2.0
        ax = np.linspace(-half, half, self.n_pix)
        self.x, self.y = np.meshgrid(ax, ax)
        self.deg_per_pix = self.size_deg / (self.n_pix - 1)

    @property
    def shape(self):
        return (self.n_pix, self.n_pix)


class Stimulus:
    """Base class: a stimulus is a function of time returning an image."""

    def image(self, grid: RetinalGrid, t_ms: float) -> np.ndarray:
        raise NotImplementedError


class Blank(Stimulus):
    """Uniform mean luminance -- the baseline condition."""

    def image(self, grid: RetinalGrid, t_ms: float) -> np.ndarray:
        return np.zeros(grid.shape)


@dataclass
class DriftingGrating(Stimulus):
    """The workhorse of visual physiology.

    ``orientation_deg`` is the orientation of the *bars*; the grating drifts
    orthogonally to them.  Spatial frequency is in cycles/degree, temporal
    frequency in Hz.
    """

    orientation_deg: float = 0.0
    spatial_freq: float = 2.0
    temporal_freq: float = 4.0
    contrast: float = 1.0
    phase_deg: float = 0.0

    def image(self, grid: RetinalGrid, t_ms: float) -> np.ndarray:
        th = np.deg2rad(self.orientation_deg)
        # Project onto the axis perpendicular to the bars.
        proj = grid.x * np.sin(th) + grid.y * np.cos(th)
        phase = np.deg2rad(self.phase_deg)
        drift = 2 * np.pi * self.temporal_freq * t_ms / 1000.0
        return self.contrast * np.cos(
            2 * np.pi * self.spatial_freq * proj - drift + phase
        )


@dataclass
class StaticGrating(DriftingGrating):
    """A counterphase-free standing grating: reveals a simple cell's phase
    sensitivity, because the response depends on where the bright bar falls."""

    temporal_freq: float = 0.0


@dataclass
class Bar(Stimulus):
    """An oriented light or dark bar -- the stimulus Hubel and Wiesel used.

    ``on_ms``/``off_ms`` bracket its presentation so onset transients can be
    measured.
    """

    orientation_deg: float = 0.0
    width_deg: float = 0.15
    length_deg: float = 1.4
    contrast: float = 1.0
    offset_deg: float = 0.0  # displacement perpendicular to the bar axis
    on_ms: float = 0.0
    off_ms: float = 1e9

    def image(self, grid: RetinalGrid, t_ms: float) -> np.ndarray:
        if not (self.on_ms <= t_ms < self.off_ms):
            return np.zeros(grid.shape)
        th = np.deg2rad(self.orientation_deg)
        # Coordinates along and across the bar.
        across = grid.x * np.sin(th) + grid.y * np.cos(th) - self.offset_deg
        along = grid.x * np.cos(th) - grid.y * np.sin(th)
        m = (np.abs(across) <= self.width_deg / 2) & (
            np.abs(along) <= self.length_deg / 2
        )
        return self.contrast * m.astype(float)


@dataclass
class MovingBar(Bar):
    """A bar sweeping across the receptive field at ``speed_deg_s``."""

    speed_deg_s: float = 4.0
    start_offset_deg: float = -0.8

    def image(self, grid: RetinalGrid, t_ms: float) -> np.ndarray:
        self.offset_deg = (
            self.start_offset_deg + self.speed_deg_s * t_ms / 1000.0
        )
        return Bar.image(self, grid, t_ms)


@dataclass
class Edge(Stimulus):
    """A luminance step -- the stimulus that first drove a V1 cell for Hubel
    and Wiesel, when the edge of a slide swept across the receptive field."""

    orientation_deg: float = 0.0
    contrast: float = 1.0
    offset_deg: float = 0.0

    def image(self, grid: RetinalGrid, t_ms: float) -> np.ndarray:
        th = np.deg2rad(self.orientation_deg)
        across = grid.x * np.sin(th) + grid.y * np.cos(th) - self.offset_deg
        return self.contrast * np.sign(across)


@dataclass
class FullFieldFlash(Stimulus):
    """Uniform luminance step: strong for the retina, weak for oriented V1
    cells.  A useful negative control for orientation selectivity."""

    contrast: float = 1.0
    on_ms: float = 0.0
    off_ms: float = 1e9

    def image(self, grid: RetinalGrid, t_ms: float) -> np.ndarray:
        on = self.on_ms <= t_ms < self.off_ms
        return np.full(grid.shape, self.contrast if on else 0.0)


@dataclass
class Dichoptic(Stimulus):
    """Different images to the two eyes, for ocular-dominance measurements."""

    left: Stimulus = None
    right: Stimulus = None

    def image(self, grid: RetinalGrid, t_ms: float) -> np.ndarray:
        # Binocular mean, used only if an eye-agnostic image is requested.
        li = self.left.image(grid, t_ms) if self.left else np.zeros(grid.shape)
        ri = self.right.image(grid, t_ms) if self.right else np.zeros(grid.shape)
        return 0.5 * (li + ri)

    def eye_image(self, grid: RetinalGrid, t_ms: float, eye: int) -> np.ndarray:
        s = self.left if eye == 0 else self.right
        return s.image(grid, t_ms) if s else np.zeros(grid.shape)
