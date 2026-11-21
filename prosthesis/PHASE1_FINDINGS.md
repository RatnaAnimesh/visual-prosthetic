# Phase 1 findings: it is the nonlinearity, not the wiring

> **Development note, not maintained.** Written while the pooled-divisor
> model was being built, against earlier versions of the code. Its numbers
> have drifted since. Current values are in `results/*.json`, and every
> number the manuscript quotes is checked by `paper/verify_claims.py`.

*Run `python3 prosthesis/phase1_circuit.py`. Raw numbers in
`results/phase1_circuit.json`.*

---

## The design

Four configurations of the circuit model, each **calibrated to reproduce the
same published single-electrode data** (phosphene diameter against current, as
encoded by the Dynaphos baseline). They agree on everything that has been
measured. They then differ only in what happens when more than one electrode
is on, which nobody has measured. That makes the disagreement a *prediction*
rather than a discrepancy.

| configuration | mechanism | fitted K (µA/mm²) | size-curve error |
|---|---|---|---|
| circuit off | current spread only | 1019 | 2.3% |
| horizontal only | + iso-orientation axons | 1125 | 2.4% |
| normalisation only | + divisive normalisation | 531 | 3.7% |
| full circuit | both | 589 | 4.0% |

(Refitted values, 2026-11-13. Later phases read them through
`phase1_circuit.calibrated()`, never as typed constants.)

All four pass the Phase 1 gate (`figures/p1_calibration.png`).

## The headline

Departure from linear superposition, two electrodes at 300 µA, as a function
of their separation on the cortical surface:

| separation (mm) | 0.4 | 0.57 | 0.8 | 1.2 | **1.6** | 2.0 | 2.4 | 3.2 | 4.5 |
|---|---|---|---|---|---|---|---|---|---|
| Dynaphos baseline | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| circuit off | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| horizontal only | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| normalisation only | 22.5% | 20.4% | 18.0% | 25.2% | 28.0% | 24.7% | 18.9% | 8.7% | 1.5% |
| **full circuit** | **26.8%** | 24.8% | 22.0% | 28.5% | **30.0%** | 26.1% | 20.2% | 9.5% | 1.7% |

The deviation is **negative** — two electrodes together produce *less* than the
sum of each alone. It peaks near 1.6 mm, close to the normalisation pooling
scale, and has fallen to noise by 4.5 mm. Cortivis electrode pitch is 400 µm
and the array spans ~4 mm, so the whole affected range is the range real
arrays occupy.

`figures/p1_pair.png` shows what this means perceptually: two electrodes
1.4 mm apart, which the baseline renders as two distinct bright spots, merge
into a single elongated smear under the circuit model. That is the cortical
analogue of the retinal letter becoming illegible.

## The correction: my starting hypothesis was wrong

The scope predicted that **horizontal connections** would be the mechanism, by
analogy with the retinal axon bundles that broke the pixel assumption. Phase 0
called them "the likeliest source of a real effect."

They contribute **exactly nothing**: 0.00% at every separation.

The reason is embarrassingly simple in hindsight. Horizontal spread, however
elaborate and however orientation-selective, is a **linear** operator — three
Gaussian convolutions. A linear operator applied to (A + B) is by definition
the sum of its application to A and to B. No amount of clever connectivity can
break superposition if the connectivity is linear.

What breaks it is the **nonlinearity**: divisive normalisation, where each
site's response is divided by pooled activity in its neighbourhood, so two
electrodes sharing a pool suppress each other. Normalisation alone reproduces
essentially the whole effect (22–28%); horizontal connections add a few points
on top (to 27–30%) by widening the pool each electrode draws on.

So the disanalogy with the retina is real and worth stating plainly: the
retinal axon-bundle effect is a *geometric* one that a linear model captures,
and that is why `AxonMapModel` works. The cortical effect, if it exists, is a
*dynamic* one that no linear model can capture at all — including any linear
extension of the current baselines.

This also retroactively justifies the last stretch of the column project. The
normalisation mechanism doing the work here is the same one that took several
attempts to get right in `column/`, and the lesson transfers exactly: what
matters is that the gain control is divisive and pooled, not where the
connectivity runs.

## What is and is not established

**Robust** (follows from the structure of the model, not its parameters):
- Linear mechanisms cannot produce electrode interaction. Certain.
- Divisive normalisation produces *suppressive* interaction, peaked at the
  pooling scale, decaying to nothing beyond it. This is the shape of the
  curve, and it does not depend on the gain.
- The affected range coincides with real array geometry.

**The magnitude** was flagged as the weak point, since `norm_gain` is not
calibrated against multi-electrode data. The sensitivity analysis below
settles how much that matters, and the answer is: less than expected.

**Also not established** — that the percept follows the activation. Everything
above is cortical activation. The step from activation to what a person
reports seeing is Phase 2, and none of these figures should be read as
predicted percepts.

## Sensitivity analysis

*`python3 prosthesis/phase1_sensitivity.py`, figure `p1_sensitivity.png`.*

`norm_gain` swept over 32× (0.25 to 8) with **K refitted at every point**, so
every model compared still reproduces the published single-electrode curve.
Without that refit, changing normalisation would be confounded with changing
phosphene size.

| norm_gain | 0.25 | 0.5 | 1.0 | 2.0 | 4.0 | 8.0 |
|---|---|---|---|---|---|---|
| fitted K | 585 | 589 | 608 | 614 | 614 | 620 |
| single-electrode fit error | 3.1% | 3.7% | 3.5% | 3.6% | 4.0% | 4.0% |
| **peak deviation** | 25% | 21% | 30% | 37% | 42% | 45% |
| **local-peak separation** | 1442 | 1600 | 1442 | 1442 | 1442 | 1442 µm |
| **half-decay distance** | 2400 | 2800 | 2800 | 2800 | 2800 | 2800 µm |

Three things fall out, and they change the conclusion for the better.

**The magnitude is far more robust than claimed.** A 32-fold change in the
uncalibrated parameter moves the prediction by about 2× — 21% to 45%. Even the
weakest normalisation tested predicts a 25% departure. The Phase 1 headline
was stated too tentatively: the *existence* of a large effect does not depend
on getting `norm_gain` right at all.

**The length scales do not move.** The local peak sits at 1442–1600 µm and the
half-decay at 2400–2800 µm across the entire range. These are set by the
pooling extent, not the gain — confirmed by the second sweep, where changing
`norm_sigma` from 750 to 3000 µm walks the peak from 400 to 2000 µm.

**Existing data does not constrain the gain.** Every value from 0.25 to 8 fits
the single-electrode curve to 3–4%, because refitting K absorbs the change.
This was worth checking and the answer is a clean negative: single-electrode
psychophysics cannot pin normalisation strength, so multi-electrode
measurements are the only way to get it.

One correction to my own analysis while making this figure. I first reported
the peak location as jumping to 400 µm at high gain, and captioned a panel
"location does not move" while the numbers said it did. Both were artefacts of
using a plain `argmax`: at strong normalisation the curve becomes monotonically
decreasing with a shoulder, so the global maximum lands on the smallest
separation sampled even though the interior feature has not moved a micron.
Reporting the local peak and the half-decay separately fixes it.

## The falsifiable prediction

Revised in light of the sensitivity analysis — and it is now a *measurement*,
not merely a test:

> Stimulate two electrodes at a separation near 1.5 mm on the cortical
> surface, each at an amplitude that alone produces a clearly visible
> phosphene. The pair will produce a percept **dimmer than, and spatially
> merged relative to**, the two phosphenes reported separately. The effect
> should exceed 20% and vanish by ~4 mm.
>
> The separation at which suppression is strongest, and the distance over
> which it decays, **measure the extent of the cortical normalisation pool** —
> a quantity not otherwise accessible in humans. The magnitude measures its
> strength.
>
> The current generation of simulators predicts no such effect at any
> separation.

Testing at two or three separations is the important part: it controls for
attention, adaptation, and reporting bias, which would affect all conditions
equally. And because the length scales are parameter-independent while the
magnitude is not, the *shape* of the result is the model test and the
*magnitude* is the parameter estimate.

## Next

Phase 2 (the percept decoder) is what turns these activation maps into
predicted percepts, and is required before any of this can be phrased as a
claim about what someone sees. Before that, two cheaper things worth doing:

1. **Sensitivity analysis on `norm_gain`.** The one number the conclusion
   hangs on.
2. **Check the literature for existing multi-electrode phosphene reports.**
   Fernández's 19-month study stimulated many electrodes; if any pairwise data
   exists, it constrains `norm_gain` directly and turns the prediction into a
   fit.
