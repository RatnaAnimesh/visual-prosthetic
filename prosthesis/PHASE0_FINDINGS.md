# Phase 0 findings: what the existing models actually assume

> **Development note, not maintained.** Written while the pooled-divisor
> model was being built, against earlier versions of the code. Its numbers
> have drifted since. Current values are in `results/*.json`, and every
> number the manuscript quotes is checked by `paper/verify_claims.py`.

*Reproduced August 2026 with pulse2percept 0.9.1. Everything here is other
people's work; the point of the exercise is to know exactly what it does
before proposing to extend it.*

Run: `python3 prosthesis/phase0_baselines.py`

---

## 1. The landscape is further along than the scope assumed

The most useful result of Phase 0 is a correction to my own plan.

`pulse2percept` 0.9.1 **already ships cortical support**:

| what | what's there |
|---|---|
| cortical models | `cortex.DynaphosModel` (the van der Grinten eLife 2023 model), `cortex.ScoreboardModel` |
| cortical implants | `Cortivis`, `ICVP`, `Orion`, `Neuralink`, `LinearEdgeThread` |
| retinotopy | `Polimeni2006Map` (wedge-dipole), `CorticalMap`, **`NeuropythyMap`** (patient-specific, from MRI) |

Three consequences for the plan:

- **Do not build a retinotopic map.** `Polimeni2006Map` is the published
  wedge-dipole model, already validated and integrated. Phase 1 gets shorter.
- **Patient-specific anatomy is already solved** by `NeuropythyMap`, which was
  listed as a hard problem in the scope. It isn't ours to solve.
- **Build the circuit layer as a pulse2percept-compatible model**, not a
  standalone program. That makes the comparison against Dynaphos exact rather
  than approximate, and makes the work contributable upstream instead of
  orphaned.

## 2. The retinal cautionary tale, reproduced

`figures/p0_retinal_pixel_vs_axon.png`. Same Argus II implant, same
electrodes, same positions — only the model differs.

- **Scoreboard** (each electrode an independent round blob): the seven-electrode
  pattern renders as a clean, legible letter.
- **Axon map** (Beyeler et al.; electrodes also activate passing nerve fibre
  bundles): the same pattern becomes a set of overlapping comet-shaped smears,
  and the letter is close to unreadable.

The pixel assumption failed, and it failed because of *neuroanatomy the
geometric model did not contain*. This is the precedent the whole project
rests on.

## 3. The cortical baseline is exactly linear — measured, not assumed

`figures/p0_cortical_superposition.png`. Dynaphos, Cortivis array, two
electrodes ~0.8° apart in the visual field, 300 Hz biphasic trains at 200 µA.

| quantity | value |
|---|---|
| max \|percept(A+B) − (percept(A) + percept(B))\| | 3.7 × 10⁻⁹ |
| max \|percept(A) + percept(B)\| | 1.0 |
| **relative deviation** | **3.7 × 10⁻⁹** |

That is floating-point noise. Superposition holds *exactly*: the current state
of the art for cortical prosthesis simulation is perfectly linear in
electrodes, by construction. Its own docstring says so — "percepts from each
electrode are Gaussian blobs" — and the measurement confirms it.

This is the number the project exists to challenge. Any nonlinearity a circuit
model predicts is, by definition, entirely absent from the model the field
currently optimises its encoders against.

## 4. Baseline calibration curves (the Phase 1 gate)

`figures/p0_phosphene_size_current.png`. Single Cortivis electrode at ~1.8°
eccentricity.

| amplitude (µA) | 50 | 100 | 150 | 200 | 300 | 400 | 600 |
|---|---|---|---|---|---|---|---|
| phosphene diameter (deg) | 0.00 | 0.13 | 0.16 | 0.17 | 0.21 | 0.25 | 0.30 |
| peak brightness (a.u.) | 0.00 | 0.94 | 0.99 | 0.99 | 0.99 | 1.00 | 1.00 |

Size grows roughly as √I, which is what a current-spread model gives and what
Bosking and colleagues report. **A circuit model must reproduce this curve
before it has any standing to criticise a geometric one.** That is the Phase 1
kill criterion, and this table is the target.

One incidental observation worth carrying forward: the baseline's brightness
function is a near-step — nothing below ~50 µA, saturated by 100 µA. That is
the same all-or-nothing dynamic-range problem the column model had before
presynaptic normalisation (`presyn_norm_k`), and it arises for the same
structural reason: a fixed threshold with no gain control. Whether real
phosphene brightness is genuinely that step-like is a question worth putting
to the psychophysics data.

## 5. Assumptions audit

What Dynaphos models: retinotopy (wedge-dipole), cortical magnification,
current spread, amplitude/frequency/pulse-width effects on brightness and
threshold, temporal dynamics via a leaky integrator plus a memory trace for
accommodation.

What it does **not** model, and what a V1 circuit layer could contribute:

| not modelled | why it might matter |
|---|---|
| electrode interaction | assumed independent; measured here as exactly linear |
| orientation columns | stimulating across a pinwheel vs. within a column may not be equivalent |
| horizontal connections | like-to-like lateral spread is the cortical analogue of the retinal axon bundle — the exact mechanism that broke the pixel assumption in the retina |
| normalisation / gain control | brightness may be sublinear in the *number* of active electrodes, not just amplitude |
| excitatory/inhibitory dynamics | affects temporal response and adaptation |
| deafferented cortex | a blind patient's V1 is homeostatically hyperexcitable; the model has no such state |

## 6. Revised Phase 1

Shorter and better targeted than the original scope:

1. Implement the circuit layer as a pulse2percept `SpatialModel` subclass, so
   it drops into the same API as Dynaphos and the comparison is exact.
2. Use `Polimeni2006Map` for retinotopy rather than our idealised pinwheel.
3. Build the electrode → tissue activation front-end. Include axonal
   recruitment of horizontal fibres explicitly: on the retinal precedent, this
   is the most likely source of a real effect.
4. Gate on reproducing §4's size-vs-current curve.

The superposition test in `phase0_baselines.py` becomes the standing
regression: run it against the circuit model, and the number that comes out
*is* the result.
