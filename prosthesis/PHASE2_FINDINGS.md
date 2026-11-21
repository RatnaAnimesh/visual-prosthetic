# Phase 2 findings: from cortical activation to a predicted percept

> **Development note, not maintained.** Written while the pooled-divisor
> model was being built, against earlier versions of the code. Its numbers
> have drifted since. Current values are in `results/*.json`, and every
> number the manuscript quotes is checked by `paper/verify_claims.py`.

*Run `python3 prosthesis/phase2_percepts.py`. Numbers in
`results/phase2_percepts.json`.*

---

## What this replaces

Every result before this one stopped at cortical activation, and the step to a
percept was an assumption: that what you see is the activation painted back
through the retinotopic map. That assumption says a cortical site's
contribution to vision depends only on *where* it sits in the map and not at
all on *what it is tuned for*, which is not obviously true and is exactly what
a circuit model should be able to question.

The replacement: build the normal-vision pathway forwards — a V1 site is an
oriented filter with a receptive-field position from the retinotopy, an
orientation preference from the orientation map, and a spatial phase — then
invert it. The predicted percept is *the image that would have produced this
cortical activity*.

## The decoder

Held-out reconstruction R² = **0.66** (`figures/p2_decoder_validation.png`),
on 1/f noise images the decoder never saw during construction.

Two things had to be got right, and both were wrong on the first attempt.

**Ridge regression is the wrong tool here.** A ridge decoder scored a
comparable R² (0.65) on natural images and then produced structured *speckle*
when applied to stimulation — not anything a person could report seeing.
The reason is fundamental rather than fixable by tuning: ridge learns
E[image | activity] *under its training distribution*, and stimulation is
nowhere near that distribution. Visual input drives a site only if the
stimulus matches its orientation and phase; an electrode drives every site
near it regardless of tuning, which is an activity pattern of essentially zero
probability under natural viewing. Extrapolating a regression there is
meaningless.

The fix is to stop regressing and solve the inverse problem directly:

    minimise  || M I − a ||²  +  λ || ∇I ||²

which asks "which image would have produced this activity" — a well-posed
question whatever the activity is. It also has one unknown per *pixel* (256)
rather than per *cortical site* (5929), so it is far better conditioned.

**A convolution/correlation bug.** The encoder used `fftconvolve`, which flips
its kernel; the inverse operator used a dot product, which does not. Gabor
filters are not symmetric, so the two disagreed by exactly that flip, and the
first inverse returned R² = −3.8. Correlation between the operator and the
true response was 0.045; flipping the kernel took it to 0.54, and the
remainder was a second inconsistency in how the two rounded receptive-field
positions to pixels. The durable fix was to stop maintaining two
implementations: `encode` is now *defined* as `rectify(M · I)` using the same
operator the inverse solves with, so they cannot drift apart.

## Reconstruction is hypercolumn-limited, and that is a real result

Sweeping the reconstruction resolution:

| image resolution | 12×12 | 16×16 | 24×24 | 32×32 | 48×48 |
|---|---|---|---|---|---|
| degrees per pixel | 0.20 | 0.15 | 0.10 | 0.07 | 0.047 |
| held-out R² | 0.55 | **0.66** | 0.31 | 0.46 | 0.40 |

The optimum sits at ~0.15°/pixel. That is not a tuning artefact — it is the
hypercolumn. A full set of orientations and phases occupies ~750 µm of cortex,
which at this eccentricity is ~0.18° of visual field, so the population cannot
resolve the image more finely than that no matter how densely the sheet is
sampled. Sampling cortex at 100 µm does not help, because neighbouring sites
carry *different* orientations rather than repeating the same measurement.

The same limit sets the spatial frequency the code can carry, which is why
`PREF_SF_CPD` is at the low end of the physiological range: asking the code to
represent 3 cycles/deg degrades reconstruction to R² 0.57.

## Does circuit interaction survive the readout?

The Phase 1 worry was that a 30% change in cortical activation might be
perceptually invisible — or, conversely, that the readout might amplify it.
Five electrodes driven together, 300 µA each, spanning 0–4.5 mm of cortex:

| | change in activation | change in decoded percept |
|---|---|---|
| independent (baseline-like) | 0.0% | 0.0% |
| circuit | 36.9% | **39.2%** |

So it transfers roughly one-for-one. The interaction is neither washed out by
the readout nor amplified by it, and `figures/p2_pattern.png` shows the form
it takes: the circuit suppresses the whole percept fairly uniformly rather
than distorting its shape.

I initially reported this as 58% — a percept change substantially *larger*
than the activation change — which was an artefact of the broken ridge
decoder, and would have been an over-claim. The corrected number is the
unexciting one.

## What this does and does not license

**Now supportable:** statements of the form "the model predicts the percept
changes by X%", because there is a fitted, validated readout behind them
rather than an assumption.

**Still not supportable:** any claim about what a *patient* would report. The
decoder is trained on the model's own normal-vision pathway, so it inherits
every one of that pathway's inaccuracies, and it has never been compared
against a human percept. It answers "what image is most consistent with this
cortical activity, according to this model of V1" — which is the right
question, but it is a claim about the model, not about a person.

**A caveat specific to stimulation.** Electrical stimulation drives
opposite-phase sites equally. In a code where phase carries the sign of local
contrast, equal drive to both members of a phase pair carries no net contrast
information, and what the inverse returns in that regime is determined largely
by the smoothness prior rather than by the data. This is worth flagging as a
place where the model's answer is soft, and it may be a genuine prediction
about why phosphenes look like unstructured spots rather than image features —
but testing that properly needs the phase structure handled explicitly rather
than as a by-product.

## Next

Phase 3 in the scope is the full interaction map, most of which Phase 1
already delivered. The more valuable next step is Phase 4: take an
encoder trained against the independent baseline, evaluate it under the
circuit model with this decoder in place, and quantify what fraction of
percept fidelity is lost — which is the question with consequences for how
real devices are programmed.
