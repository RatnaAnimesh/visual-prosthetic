# Divisive normalisation breaks electrode independence in cortical visual prostheses, and scale-invariant metrics cannot see it

> **Superseded.** An earlier draft, kept for history. The manuscript is
> `paper/manuscript_v2.tex`, and its numbers are checked by
> `paper/verify_claims.py`. Numbers here are not maintained.

**Draft — August 2026.** Target venue: *Journal of Neural Engineering* (or
*eLife*, where the baseline model appeared). All code, tests and figures:
this repository.

---

## Abstract

Simulators of cortical visual prostheses treat electrodes as independent:
each produces a Gaussian phosphene at its retinotopic position, and percepts
add linearly. That assumption is load-bearing, because the current generation
of stimulus encoders is optimised against these simulators. We built a
forward model that adds two properties of V1 the existing models omit —
long-range horizontal connections between iso-orientation domains, and
divisive normalisation — and calibrated it so that it reproduces the same
published single-electrode phosphene size-versus-current data as the baseline.
The models therefore agree on everything that has been measured and differ
only on multi-electrode stimulation, which has not been.

Three results follow. First, a negative one that is provable rather than
empirical: **linear mechanisms cannot break electrode independence**, however
elaborate. Horizontal spread, even when strongly orientation-selective,
contributes exactly zero interaction (< 10⁻⁹ at every separation and gain
tested), because a linear operator applied to a sum is the sum of its
applications. This disanalogises the cortical case from the retinal one, where
axon-bundle activation — a geometric, linear effect — famously broke the
"pixel" assumption. Second, divisive normalisation does break it, producing
suppressive interaction of 19–44% that peaks at 1.2–1.6 mm electrode
separation and decays to noise by ~4.5 mm — the range real arrays occupy. A
sensitivity analysis over a 32-fold range of normalisation strength moves the
magnitude by only ~2× and leaves the length scales unchanged, while showing
that single-electrode psychophysics cannot constrain that strength at all.
Third, using a decoder fitted to the model's own normal-vision pathway to
predict percepts, we find that optimising stimulation against the independent
model costs only 0.6% of *pattern* fidelity but delivers percepts at 39% of
the intended contrast, a shortfall that re-optimisation cannot recover and
that a scale-sensitive objective can barely improve (0.39 → 0.44, at a cost
in pattern fidelity) and a scale-invariant one cannot detect at all.

Human multi-electrode data supports this. Bosking and colleagues report that
concurrently stimulated surface electrodes produce phosphenes significantly
smaller than independent stimulation does (2.49° against 3.59°, *p* < .001)
[15] — a suppressive interaction in the direction we predict and one no
independent model can generate. Comparing spatial scale requires matching
stimulation regime, since those experiments used up to 4 mA against our
300 µA and activated radius grows as √I: at matched current the model predicts
39% interaction at the separation where percepts merge and 0% where two
distinct phosphenes are reported, separating the regimes correctly with no
parameter adjusted to make it do so. A residual remains at intermediate
separations (~13 mm), and because that study did not measure brightness our
central percept prediction is untested.

---

## 1. Introduction

A cortical visual prosthesis stimulates V1 directly, bypassing the eye and
optic nerve. It is the only route to restored vision when the retina or optic
nerve is destroyed — glaucoma, optic neuropathy, trauma — a larger population
than retinal implants can serve. Chronic phosphene generation has been
demonstrated safely over 19 months with a 96-channel intracortical array
[1,2], and Phase I trials of wireless intracortical devices are underway [3].
Recent work shows phosphene thresholds and brightness can be predicted from
neighbouring-electrode activity [4].

Turning that into useful vision is an encoding problem: given a camera image,
which electrodes fire, at what amplitude. The field solves it by optimising an
encoder against a differentiable forward model of what stimulation produces
[5–8]. For cortex the state of the art is Dynaphos [9], now distributed inside
`pulse2percept` [10]. Dynaphos models retinotopy via a reverse wedge-dipole
map, cortical magnification, current spread, and brightness and threshold as
functions of stimulation parameters. It does not model V1 circuitry, and its
documentation states the consequence plainly: percepts from each electrode are
Gaussian blobs, and electrodes are assumed independent.

There is a precedent for worrying about this. Retinal prosthesis simulation
made the same simplification and it failed: an epiretinal electrode also
activates the ganglion-cell axon bundles passing over it, so percepts are not
round pixels but elongated comets, and patterns intended as letters become
illegible [10]. The correction was anatomical — something the geometric model
did not contain.

The obvious cortical analogue is the long-range horizontal connection system:
superficial pyramidal axons running for millimetres parallel to the surface
and terminating in clusters on columns of similar orientation preference
[11,12]. We set out to test whether these produce the cortical equivalent of
the axon-bundle effect. They do not, and understanding why turned out to be
the most useful part of this work.

The assumption is also already known to be imperfect empirically. Bosking and
colleagues stimulated 2–6 subdural surface electrodes simultaneously in 11
volunteers and found that concurrent stimulation yields phosphenes reliably
smaller than independent stimulation, and that some pairs produce one
phosphene where two are expected [15]; more recent work predicts phosphene
threshold, brightness and number from the activity of neighbouring electrodes
[4]. What has been missing is a mechanism, and a forward model that expresses
one.

## 2. Methods

### 2.1 Baseline

All comparisons use `pulse2percept` 0.9.1 with the `Cortivis` implant
geometry and `Polimeni2006Map` retinotopy. The baseline is
`cortex.DynaphosModel` with default parameters, driven by 300 Hz biphasic
pulse trains (170 µs phase, 100 ms burst).

We first verified the independence assumption empirically rather than taking
it from documentation. For two electrodes 0.8° apart in the visual field at
200 µA, the maximum relative deviation from linear superposition was
3.7 × 10⁻⁹ — floating-point noise. Superposition holds exactly.

### 2.2 Circuit model

We implemented `CircuitSpatial` as a `pulse2percept` spatial model so the
comparison is like-for-like and the code is contributable upstream. Activation
is computed on a uniform cortical sheet (50 µm) and read out at each
visual-field grid point through the same retinotopy the implant uses.

**Current spread** reproduces Dynaphos exactly: activated tissue diameter
D = 2√(I/K). Activation magnitude additionally scales with a saturating
function of amplitude, which leaves the half-maximum contour — and therefore
the size calibration — unchanged.

**Orientation map.** Iso-orientation domains are generated by band-pass
filtering complex Gaussian noise and halving its phase [13], giving
quasi-periodic domains joined at pinwheels with the macaque V1 column spacing
of ~750 µm.

**Horizontal connections.** Activation spreads to sites weighted by cortical
distance (σ = 1 mm) and orientation similarity (σ = 27°). Expanding the
orientation term to first Fourier order makes this three plain Gaussian
convolutions rather than an N² kernel:

  G(A) + r·cos2θ·G(A cos2θ) + r·sin2θ·G(A sin2θ), all over (1 + r)

which keeps a millimetre-scale sheet tractable and the model differentiable.

**Divisive normalisation** [14]: each site's expansive response is divided by
a Gaussian-pooled measure of neighbourhood activity (σ = 1.5 mm).

Both mechanisms can be disabled independently, and with both off the model
reduces to pure current spread — the control condition that makes any
difference attributable to the circuit rather than to a different spread
model.

### 2.3 Calibration

Every configuration is fitted, by adjusting K alone, to reproduce the
baseline's single-electrode phosphene diameter across 100–600 µA. Mean
absolute error: 2.3% (circuit off), 2.4% (horizontal only), 3.6%
(normalisation only), 3.5% (both). All configurations therefore agree with the
published single-electrode data, and with each other, on everything that has
been measured.

### 2.4 Percept decoder

To make claims about percepts rather than activation we fit a readout. Each
cortical site is an oriented filter with a receptive-field position from the
retinotopy, an orientation from the map, and a spatial phase. The predicted
percept is the image that best explains the observed activity:

  minimise ‖M·I − a‖² + λ‖∇I‖²

We use this rather than regressing images on activity. A ridge decoder
achieves comparable reconstruction on natural images (R² 0.65 vs 0.66) and
then produces structured speckle on stimulation, because it estimates
E[image | activity] under a training distribution that stimulation is nowhere
near: visual input drives a site only if the stimulus matches its tuning,
whereas an electrode drives every site near it regardless.

Held-out reconstruction R² = 0.66 on 1/f noise images. Reconstruction
resolution optimises at ~0.15°/pixel and degrades on either side; this is the
hypercolumn limit, since a full set of orientations and phases occupies
~750 µm ≈ 0.18° of visual field, and sampling cortex more finely does not help
because neighbouring sites carry different orientations rather than repeating
a measurement.

### 2.5 Optimal stimulation

For encoder consequences we avoid training a network, which would confound
forward-model accuracy with network capacity. For each target we solve
directly for the amplitude vector maximising percept fidelity under a given
forward model (L-BFGS-B, 24 electrodes, 0–600 µA) — the best any encoder of
any architecture could achieve given that model.

Targets are defined as percepts the device can actually deliver. An initial
attempt used geometric shapes and every fidelity came out at 0.001–0.06;
fitting those targets with an unconstrained linear combination of
single-electrode percepts reached only 0.03–0.15, showing that a couple of
dozen phosphenes cannot render a thin bar in a 256-pixel image whatever the
amplitudes. Asking what a modelling error costs is meaningless when there is
no fidelity to lose.

## 3. Results

### 3.1 Linear mechanisms cannot break independence

Horizontal spread produces **exactly zero** departure from superposition:
< 10⁻⁹ at every separation (400–2400 µm) and gain (0.3–1.2) tested.

The reason is structural, not numerical. Horizontal spread as implemented — and
as implemented in any convolutional formulation — is a linear operator, and
L(A + B) = L(A) + L(B) by definition. No connectivity rule, however
orientation-selective or anatomically detailed, can create electrode
interaction if it acts linearly.

This is the paper's sharpest result and it was contrary to our expectation.
The retinal axon-bundle effect is *geometric*: it changes the shape of a
single phosphene, and a linear model captures it, which is why the axon-map
model works. A cortical interaction effect, if it exists, must be *dynamic*,
and no linear extension of any current baseline can express it.

### 3.2 Divisive normalisation does break it

With normalisation enabled, departure from superposition (300 µA per
electrode):

| separation (mm) | 0.4 | 0.8 | 1.2 | 1.6 | 2.4 | 3.2 | 4.5 |
|---|---|---|---|---|---|---|---|
| Dynaphos baseline | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| horizontal only | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| normalisation only | 11.8% | 10.8% | 18.9% | 22.4% | 15.6% | 7.1% | 1.2% |
| **full circuit** | 18.0% | 14.9% | 22.1% | **24.7%** | 17.0% | 8.0% | 1.4% |

The interaction is **suppressive** — a pair produces less than the sum — peaks
near the pooling scale, and is gone by 4.5 mm. Cortivis electrode pitch is
400 µm and the array spans ~4 mm, so the affected range is the range real
arrays occupy. Perceptually, two electrodes 1.4 mm apart that the baseline
renders as two distinct spots merge into a single elongated smear.

### 3.3 The conclusion survives not knowing the parameters

Normalisation strength is not calibrated by any public data. Sweeping it over
32× (0.25–8), refitting K at every point so all models still reproduce the
single-electrode curve:

| norm_gain | 0.25 | 0.5 | 1.0 | 2.0 | 4.0 | 8.0 |
|---|---|---|---|---|---|---|
| single-electrode fit error | 3.1% | 3.5% | 4.0% | 4.0% | 4.0% | 4.0% |
| peak deviation | 39% | 19% | 25% | 33% | 39% | 44% |
| local-peak separation (µm) | 1200 | 1442 | 1600 | 1442 | 1442 | 1442 |
| half-decay distance (µm) | 2154 | 2800 | 2800 | 2800 | 2800 | 2800 |

A 32-fold parameter change moves the prediction over a roughly two-fold range
(19–44%) and leaves the length scales essentially fixed. Sweeping the pooling
extent instead walks the peak from 400 µm to 2000 µm, confirming that geometry
is set by the pool and magnitude by the gain, separably.

The smallest gain is an outlier on both counts, and we report it rather than
smooth it: refitting K at each sweep point means gain and operating point are
not fully independent, and at the weakest normalisation the fitted K lands
somewhere that produces a different peak. The monotonic trend from 0.5 to 8 is
the interpretable part.

Note the third row: every strength from 0.25 to 8 fits the single-electrode
curve equally well, because refitting K absorbs the change. **Single-electrode
psychophysics cannot constrain normalisation strength.** Multi-electrode
measurement is the only route.

### 3.4 Comparison with human multi-electrode data

The model was built without reference to multi-electrode measurements, so
comparison with them is a test rather than a fit. It passes on the direction
of the effect, and on spatial scale once the comparison is made at matched
stimulation currents.

Bosking *et al.* [15] stimulated 2–6 subdural surface electrodes
simultaneously in 11 volunteers, analysing 232 pairs. Three findings bear on
this work.

**Concurrent stimulation shrinks phosphenes.** For pairs perceived as two
distinct phosphenes, sizes under concurrent stimulation were consistently
smaller than under independent stimulation — 2.49° against 3.59°
(*t*(34) = 8.14, *p* < .001, n = 35). A second class of pairs produced only
one phosphene where two were expected, the other being unreported. Both are
suppressive multi-electrode interactions, in the direction divisive
normalisation predicts and in a direction no independent model can produce.
This is the strongest existing evidence that the independence assumption is
false in humans.

**The spatial scale matches once the stimulation regime is matched.** Pairs
yielding a single merged phosphene had a mean separation of 9.57 mm (sd 6.56),
pairs yielding two 26.36 mm (sd 16.31), and the most tightly-spaced subclass,
where the merged percept overlapped both individual ones, 4.11 mm (sd 1.55).
Our interaction falls below 2% by 4.5 mm, which appears at first to be several
times too short — and an earlier version of this manuscript reported it as a
mismatch.

That comparison was invalid. Those experiments delivered 0.3–4.0 mA; the
intracortical predictions here were computed at 300 µA. Under the model's own
current-spread law the activated radius grows as √I, so a thirteen-fold
current increase enlarges it from 0.70 mm to 2.57 mm, and the distance over
which two electrodes share a normalisation pool grows with it. Sweeping
amplitude across their range and changing nothing else (§3.5):

| stimulation | interaction at 4.11 mm | at 9.57 mm | at 26.36 mm | 10% by |
|---|---|---|---|---|
| 0.3 mA | 2.5% | 0.0% | 0.0% | 4.4 mm |
| 1.0 mA | 12.0% | 0.0% | 0.0% | 5.2 mm |
| 2.0 mA | 22.5% | 1.0% | 0.0% | 6.4 mm |
| **4.0 mA** | **38.9%** | 3.2% | **0.0%** | 8.8 mm |

At the top of their range the model predicts strong interaction (39%) exactly
where percepts merge and overlap both individual phosphenes, and none at all
(0.0%) where two distinct phosphenes are reported. It separates the two
regimes correctly, without any parameter having been adjusted to make it do
so.

The residual is the intermediate class: pairs at a mean of 13.47 mm (n = 9)
that yield one phosphene with the other unreported. The model predicts under
2% interaction there. We tested whether a broader pool accounts for it and
conclude that it should not be made to — §3.5.

**The model's headline percept prediction is untested.** That study measured
phosphene number, size, location and eccentricity. It did not measure
brightness. Our central percept claim — that ensemble contrast falls to ~39%
of intent and cannot be recovered by current (§3.8) — is therefore neither
supported nor contradicted by it, and remains open.

One asymmetry worth noting rather than glossing: the most closely spaced pairs
produced merged phosphenes *larger* than the individual ones (6.25° against
3.07°). Two adjacent phosphenes merging into one percept spanning both would
be larger by construction, so this is not obviously evidence against amplitude
suppression, but our model does not reproduce an enlargement and we do not
claim it does.

### 3.5 Long-range single-phosphene suppression is not normalisation

V1 surround suppression is not one mechanism but two: a near component with
the reach of horizontal connections, and a far component mediated by feedback
from extrastriate areas with much larger receptive fields. We added the second
as an independent pool (`norm_far_gain`, `norm_far_sigma`) and asked whether
it can produce suppression at 13.5 mm while preserving the two regimes the
model already gets right.

It can, but only implausibly. Searching pool gain and width at 4 mA, the
requirement — suppression >15% at 13.5 mm, merging preserved (>30%) at
4.1 mm, independence preserved (<3%) at 26.4 mm — is met at exactly one
tested setting: a far pool of σ = 8 mm with **thirty times** the gain of the
near pool (40.3%, 16.6%, 0.6%). Every physiologically ordered setting, where
far suppression is weaker than near, produces at most 2–6% at 13.5 mm.

The arithmetic behind the required gain is simple. Two electrodes 13.5 mm
apart contribute exp(−13.5²/2σ²) ≈ 0.24 of peak to each other's pool at
σ = 8 mm, so for that quarter-contribution to change the response by more than
15% the pooled term must dominate the semi-saturation constant — which takes a
large gain. Widening σ instead of raising gain does not substitute for it, and
we make no stronger structural claim than that: the dependence of interaction
on pool width is not monotonic, and at short separations a broader pool
actually increases the departure from superposition rather than decreasing it.

What the search establishes is narrower but sufficient: no
physiologically-ordered parameterisation, meaning any in which far surround
suppression is weaker than near, reproduces the effect, and the one setting
that does inverts that ordering thirtyfold. Adopting it would be fitting nine
observations with a parameter nothing else constrains.

We think the model is right not to reproduce this. The original authors do not
claim the effect is cortical: they raise non-uniform spatial attention as a
candidate and state that resolving it will require direct measurement of
activity in early visual areas [15]. A forward model of cortical activation
should not be expected to reproduce a percept that may be selected after
cortex.

This yields a discriminating experiment rather than a parameter. Neural
suppression should be consistent — the same phosphene lost on every trial —
whereas perceptual competition should alternate. Trial-to-trial consistency
for these pairs has not been systematically reported, and measuring it would
settle whether the residual belongs in a forward model at all.

### 3.6 Interaction range scales with stimulation current

The amplitude sweep behind §3.4 is a result in its own right, and a prediction.
Holding every parameter fixed and varying only stimulation amplitude, the
distance at which interaction has decayed to a tenth of its peak grows from
4.4 mm at 0.3 mA to 8.8 mm at 4.0 mA (`figures/p5_scale.png`). The mechanism
is transparent: activated radius goes as √I, and two electrodes interact when
their activation overlaps the same normalisation pool.

This predicts that the spatial extent of electrode interaction is not a fixed
property of cortex but a function of how hard the device is driven — so
interaction measured at threshold currents will understate interaction at
the suprathreshold currents a device actually uses.

### 3.7 Percept-level consequences

Decoding five simultaneously driven electrodes: activation changes by 32.1%
relative to the independent prediction, and the decoded percept by 31.2%. The
interaction is neither washed out by the readout nor amplified — it transfers
approximately one-for-one, taking the form of a fairly uniform suppression of
the whole percept rather than a distortion of its shape.

### 3.8 Encoder consequences: the pattern survives, the brightness does not

Optimal stimulation computed under each model and cross-evaluated:

| | mean |
|---|---|
| programmed for circuit, delivered to circuit | 0.995 |
| programmed as independent, delivered to circuit | 0.990 |
| **pattern fidelity forgone** | **0.6%** |
| percept contrast relative to target | **0.39** |
| contrast after re-optimising a brightness-sensitive objective | 0.44 |
| pattern fidelity that costs | 0.949 |

On pattern fidelity this is a null result: the circuit's suppression is a
smooth, largely invertible distortion and a re-optimised encoder undoes almost
all of it.

On brightness it is not. The delivered percept carries 39% of the intended
contrast; an objective that actively seeks brightness recovers it only to 44%
and sacrifices pattern fidelity to do so. The amplitude bound is never
reached, so this is not a current limit. It is what divisive normalisation
*is*: driving harder recruits more of the pool that divides you, so the
suppression cannot be out-driven.

### 3.9 Neither metric rescues the brightness

The two conclusions above come from the same percepts and differ only in how
they are scored, so the choice of metric deserves stating carefully.

Squared correlation is invariant to scale: a percept of correct shape at 39%
brightness scores identically to one at full brightness. An evaluation using
it alone would report near-perfect success here while delivering under half
the intended contrast, and nothing in the pipeline would flag it. Our own first
analysis did exactly this and concluded the independence assumption was
harmless; that conclusion was wrong by omission.

We note that this is a caution about evaluation practice rather than a
criticism of published encoders, several of which optimise scale-*sensitive*
reconstruction losses [5,6]. But a scale-sensitive objective does not solve the
problem either — it merely reveals it. Re-optimising the amplitudes under a
plain squared-error objective, which is free to spend current on brightness,
raises contrast only from 0.39 to 0.44 while dropping pattern fidelity from
0.995 to 0.949, and never approaches the amplitude bound.

The brightness shortfall is therefore not an artefact of metric choice and not
a programming error. It is a property of the circuit: a divisive nonlinearity
cannot be out-driven, because the drive intended to overcome it also enlarges
the pool doing the dividing. The metric matters only for whether one notices.

## 4. Discussion

### 4.1 Predictions

Two experiments, both feasible in existing implanted volunteers.

**Is the lost phosphene lost in cortex or after it?** For pairs where one
phosphene of two goes unreported, measure trial-to-trial consistency.
Consistent loss of the same phosphene implicates a cortical mechanism, which
§3.5 shows is not divisive normalisation and would therefore require something
this class of model does not contain. Alternation implicates perceptual
competition or attention, and places the effect outside a forward model of
cortical activation altogether. This is the cheapest of the experiments
proposed here and it decides whether §3.5's residual is a modelling debt.

**Interaction range should scale with current.** The sharpest new prediction
(§3.5): the distance over which electrodes interact is not a fixed property of
cortex but grows as √I, from ~4.4 mm at 0.3 mA to ~8.8 mm at 4.0 mA. Repeating
a paired-stimulation protocol at two well-separated amplitudes should move the
separation at which merging gives way to two phosphenes. Independent models
predict no interaction at any current, and a fixed-radius interaction model
predicts no shift with amplitude, so the three are cleanly distinguishable.

**Shape of the function at intracortical pitches.** At 300 µA the model
predicts interaction is *not monotonic* in separation: an interior maximum
near 1.4–1.6 mm, weaker at both 0.4 mm and 3 mm. Measuring phosphene size or
brightness at three or four separations spanning 0.4–3 mm would test it.
Because the length scales are parameter-independent while the magnitude is
not, the *shape* tests the model and the *magnitude* estimates normalisation
strength — the separation of peak suppression measuring the cortical
normalisation pool extent, a quantity not otherwise accessible in humans.

**Brightness.** Drive N electrodes each individually producing brightness B.
Reported ensemble brightness should fall increasingly below N·B as N grows,
and increasing current should fail to close the gap. Independent models
predict brightness adds and have no mechanism for a ceiling current cannot
overcome.

Testing at two or more separations, or several N, controls for attention,
adaptation and reporting bias, which affect all conditions equally.

### 4.2 Implications for device programming

If the cortex normalises, encoders built on independent forward models get the
pattern approximately right and the brightness substantially wrong, and no
choice of objective recovers it. The practical recommendations are modest but
concrete: treat ensemble-brightness budgets derived from independent forward
models as optimistic; report a scale-sensitive metric alongside any
correlation-based one, so that a shortfall is at least visible; and do not
expect increased current to close it, because the mechanism that suppresses
also scales with drive.

The corollary for modelling is sharper. Because no linear mechanism can
produce electrode interaction (§3.1), a forward model intended to capture it
must include an explicit nonlinearity. Adding anatomical detail to a linear
model — finer retinotopy, richer connectivity, patient-specific maps — cannot
help however accurate that detail is.

### 4.3 Limitations

We are explicit about what this work does not establish.

- **Long-range single-phosphene suppression is deliberately not modelled.**
  The model reproduces the merging regime (~4 mm) and the independent regime
  (~26 mm) at matched currents but not the ~13.5 mm class where one phosphene
  goes unreported. §3.5 shows divisive normalisation can only produce this
  with a far pool thirty times stronger than the near one, inverting the known
  ordering of surround suppression. We treat it as plausibly post-cortical
  rather than fit it, but that is a judgement, and if the effect turns out to
  be consistent trial-to-trial the model will owe an explanation.
- **The model produces only suppression, and no enlargement.** The most
  closely spaced pairs in [15] give merged percepts larger than the individual
  ones. Divisive normalisation is intrinsically suppressive. Whether an
  enlargement mechanism is needed, or whether merging alone accounts for it,
  is unresolved.
- **Different electrode regime.** The available human data uses subdural
  surface contacts (0.5 mm diameter, 2–6 mm spacing) delivering current
  through CSF and dura; this model is of intracortical arrays at 400 µm pitch.
- **Normalisation strength is unconstrained.** §3.3 shows existing
  single-electrode data cannot pin it. The 30% and 39% figures are contingent;
  the direction, mechanism and length scales are not.
- **The percept decoder has never been compared to a human percept.** It is
  trained on the model's own normal-vision pathway and inherits that pathway's
  inaccuracies. It answers "what image is most consistent with this activity,
  according to this model of V1."
- **A field model, not a spiking one.** Deliberately, to keep the comparison
  with Dynaphos like-for-like and the sweeps tractable, but it means adaptation,
  spike timing and temporal dynamics beyond the baseline's are absent.
- **One eccentricity, one array geometry, one hemisphere.** All results are at
  ~1.8° eccentricity with `Cortivis`.
- **Deafferented cortex is not modelled.** A blind volunteer's V1 has had no
  input for years and is homeostatically altered; parameters here are from
  healthy cortex.

## 5. Conclusion

Electrode independence is assumed by every current cortical prosthesis
simulator and is exactly true in them by construction. Human data already
shows it is false: concurrently stimulated electrodes produce measurably
smaller phosphenes than independent ones. We find that no linear mechanism can
account for this — including the horizontal connectivity that seemed the
natural cortical analogue of the retinal axon-bundle effect — and that
divisive normalisation can, reproducing both the direction of the observed
suppression and — once stimulation currents are matched — the separations at
which percepts merge and at which they stay distinct, with no parameter
adjusted to fit. Suppression at intermediate separations (~13 mm) remains
unaccounted for. The model is a candidate mechanism with one confirmed
regime boundary, not a validated account.

The consequence for device programming is not a loss of pattern fidelity,
which re-optimisation recovers, but a brightness shortfall that no objective
recovers, because the drive intended to overcome a divisive nonlinearity also
feeds it.

---

## Author's note on the draft

Three claims in earlier versions of this manuscript were wrong and are
recorded here because they shaped the work.

1. We predicted horizontal connections would produce the interaction, by
   analogy with the retinal axon bundles. They cannot, for a reason that is
   provable rather than empirical (§3.1). The negative result is sharper than
   the hypothesis would have been.
2. We initially reported that no human multi-electrode data existed. It does
   [15]. On first reading, via a summary rather than the source, we then
   described it as showing a ~4 mm critical separation that matched our
   model's length scale. The full text says no such thing: 4.11 mm is the mean
   separation of one narrow subclass, and the actual contrast is between pairs
   at 9.57 mm yielding one phosphene and 26.36 mm yielding two. The data
   supports the *direction* of our prediction and contradicts its *scale*
   (§3.4). We record this because the erroneous version made the paper look
   stronger, which is exactly when a claim most needs checking against the
   source.
3. We initially concluded from a scale-invariant metric that the independence
   assumption was harmless. It costs little pattern fidelity but a great deal
   of brightness, which that metric cannot see (§3.7).

## Data and code availability

All code, the test suite (47 tests), and scripts regenerating every figure and
number are in this repository. `prosthesis/phase0_baselines.py` reproduces the
baselines; `phase1_circuit.py`, `phase1_sensitivity.py`, `phase2_percepts.py`
and `phase4_encoder.py` produce Results §3.1–3.6.

## References

1. Fernández E *et al.* Phosphene perceptions and safety of chronic visual
   cortex stimulation in a blind subject. https://pubmed.ncbi.nlm.nih.gov/31151104/
2. Development of a Cortical Visual Neuroprosthesis for the Blind.
   ClinicalTrials.gov NCT02983370
3. A Phase I Feasibility Study of an Intracortical Visual Prosthesis (ICVP).
   ClinicalTrials.gov NCT04634383
4. Neural correlates of phosphene perception in blind individuals.
   *Science Advances* (2025). https://www.science.org/doi/10.1126/sciadv.adv8846
5. Granley J *et al.* Deep learning–based perceptual stimulus encoder for
   bionic vision. https://bionicvisionlab.org/publications/2022-03-deep-stimulus-encoder/
6. de Ruyter van Steveninck J *et al.* End-to-end optimization of prosthetic
   vision. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8899855/
7. Optimization of neuroprosthetic vision via end-to-end deep reinforcement
   learning. https://pubmed.ncbi.nlm.nih.gov/36328967/
8. End-to-end learning of safe stimulation parameters for cortical
   neuroprosthetic vision. https://www.biorxiv.org/content/10.1101/2025.01.23.634543
9. van der Grinten M *et al.* Towards biologically plausible phosphene
   simulation for the differentiable optimization of visual cortical
   prostheses. *eLife* (2023). https://elifesciences.org/articles/85812
10. Beyeler M *et al.* pulse2percept: a Python-based simulation framework for
    bionic vision. https://bionicvisionlab.org/code/pulse2percept/
11. Gilbert CD, Wiesel TN. Clustered intrinsic connections in cat visual
    cortex. *J Neurosci* 3:1116–1133 (1983).
12. Kandel ER *et al.* *Principles of Neural Science*, 5th ed., ch. 25
    (laminar scheme Fig. 25‑12, intrinsic circuitry Fig. 25‑15, columnar
    architecture Fig. 25‑11, horizontal connections Fig. 25‑16).
13. Rojer AS, Schwartz EL. Cat and monkey cortical columnar patterns modeled
    by bandpass-filtered 2D white noise. *Biol Cybern* (1990).
14. Carandini M, Heeger DJ. Normalization as a canonical neural computation.
    *Nat Rev Neurosci* (2012).
15. Bosking WH *et al.* Percepts evoked by multi-electrode stimulation of
    human visual cortex. *Brain Stimulation* (2022).
    https://pmc.ncbi.nlm.nih.gov/articles/PMC9831085/
