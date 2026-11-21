# Phase 4 findings: the assumption is wrong in a way encoders can't see

> **Development note, not maintained.** Written while the pooled-divisor
> model was being built, against earlier versions of the code. Its numbers
> have drifted since. Current values are in `results/*.json`, and every
> number the manuscript quotes is checked by `paper/verify_claims.py`.

*Run `python3 prosthesis/phase4_encoder.py`. Numbers in
`results/phase4_encoder.json`.*

---

## The question

Encoders for cortical prostheses are optimised against a forward model, and
every published one assumes electrodes are independent. Phases 1 and 2
established that a circuit cortex would violate that by about 32% in
activation and in the decoded percept. Does that *matter* for how a device should be
programmed?

## The design

No neural network. A learned encoder confounds forward-model accuracy with
network capacity, so instead the *optimal* amplitude vector is found directly
for each target under each forward model — the best any encoder of any
architecture could do given that model. 24 electrodes, amplitudes 0–600 µA.

Both configurations were already calibrated (Phase 1) to produce identical
single-electrode phosphene sizes, and are additionally anchored here to
produce identical single-electrode *brightness*, so nothing below is an
artefact of the two models living on different scales.

## The answer, in two parts

| | mean |
|---|---|
| linear-basis reference (see below) | 0.994 |
| programmed for circuit, delivered to circuit | 0.995 |
| **programmed as independent, delivered to circuit** | **0.988** |
| **pattern fidelity forgone** | **0.8%** |
| | |
| percept contrast, relative to target | **0.38** |
| contrast after re-optimising a brightness-sensitive objective | 0.44 |
| pattern fidelity that costs | 0.949 |

**Pattern: a null result.** Optimising against the wrong forward model costs
0.8% of pattern fidelity. The circuit's suppression is a smooth, largely
invertible distortion, and a re-optimised encoder undoes almost all of it. On
this axis the field's independence assumption is wrong but nearly harmless.

**Brightness: not a null, and not recoverable.** The delivered percept has 38%
of the intended contrast. Re-optimising with an objective that *wants*
brightness recovers it only to 44%, and costs pattern fidelity (0.995 → 0.949)
to get there. Four of the six targets stop short of the 600 µA bound with at
most half the intended contrast, so for them this is not a current limit. Two
reach the bound. An earlier version of this note said the bound was never
reached, a claim nothing recorded at the time. The run now saves each
target's amplitudes, and the claim did not survive.

The mechanism is exactly what divisive normalisation is: **you cannot
out-drive it**, because driving harder recruits more of the pool that divides
you. The brightness shortfall is a property of the cortex, not of the
programming.

## The methodological trap

The two halves above are measured with the same percepts and differ only in
the metric. Squared correlation — the natural choice, and close to what the
encoder-optimisation literature uses — is **invariant to scale**. A percept
of the right shape at 38% brightness scores identically to one at full
brightness.

So an encoder optimised against an independent forward model, evaluated with a
correlation-like objective, would report near-perfect success while delivering
a percept at well under half the intended contrast, and nothing in the
pipeline would flag it. That is a concrete, actionable warning: **brightness
budgets computed from independent forward models will be optimistic, and
scale-invariant metrics will not reveal it.**

My own first version of this experiment reported only the correlation metric
and concluded "0.6% lost, the assumption is harmless." That conclusion was
wrong by omission.

## Two things that went wrong first, and what they showed

**The targets were undeliverable.** The first run used geometric shapes —
bars, a cross, a disc — and every fidelity came out at 0.001–0.06. That was
not the optimiser failing. Fitting the target with an unconstrained linear
combination of the per-electrode percepts gave only 0.03–0.15: *sixteen
phosphenes cannot draw a thin bar in a 256-pixel image*, whatever the
amplitudes. Asking what a modelling error costs is meaningless when there is
no fidelity to lose. Targets are now defined as percepts the device can
actually deliver, and this linear-basis reference is reported beside every
result so the question stays well posed.

One correction on that diagnostic: I first called it a strict upper bound and
wrote a test asserting it as one, which duly failed. It is not a bound. The
model is nonlinear in amplitude — activated radius grows as √I and magnitude
saturates — so a percept produced at 450 µA is not a linear combination of
400 µA percepts, and optimisation can slightly exceed the figure. It is a
scale for whether a target is deliverable, not a ceiling.

**The brightness comparison was initially meaningless.** Normalisation divides
by a pooled term and changes the activation units, so the first contrast ratio
came out at 2.16 — the circuit percept "brighter" than the target, which is
backwards. Anchoring both models to the same single-electrode brightness (the
natural completion of Phase 1's size calibration) turned it into 0.39, and
0.38 once every phase read the same phase-1 calibration.

## What this licenses

**Supportable:** if the cortex normalises, an encoder built on an independent
forward model gets the pattern right and the brightness substantially wrong,
and a correlation-style objective cannot detect the shortfall.

**Not supportable:** the specific figure of 38%. It scales with normalisation
strength, which Phase 1's sensitivity analysis showed is unconstrained by any
existing single-electrode data. What is robust is the *direction* and the
*mechanism* — suppression that cannot be out-driven — not the number.

**Still untested against a person.** Everything rests on a decoder trained on
the model's own normal-vision pathway. The claim is about the model.

## The experiment this suggests

Phase 1's prediction was about phosphene position and merging. This adds a
brightness one, which is easier to measure and harder to explain away:

> Drive N electrodes simultaneously, each at an amplitude that alone gives a
> phosphene of reported brightness B. The reported brightness of the ensemble
> should fall increasingly far below N·B as N grows, and increasing current
> should fail to close the gap.
>
> Independent models predict brightness adds; they have no mechanism for a
> ceiling that current cannot overcome.
