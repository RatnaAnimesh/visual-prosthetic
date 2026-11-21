# Electrode independence in cortical visual prostheses: only a nonlinearity can break it

> **Superseded.** An earlier draft, kept for history. The manuscript is
> `paper/manuscript_v2.tex`, and its numbers are checked by
> `paper/verify_claims.py`. Numbers here are not maintained.

**Running title:** Electrode independence in cortical prostheses

**Author.** *[Name]*
*[Affiliation]*

**Corresponding author:** *[email]*

**Keywords:** visual prosthesis, visual cortex, phosphene, divisive
normalisation, computational model, neurostimulation

**Manuscript type:** Paper. Prepared for *Journal of Neural Engineering*.

---

## Abstract

**Objective.** Simulators of cortical visual prostheses treat electrodes as
independent: each produces a Gaussian phosphene at its retinotopic position,
and percepts add linearly. That assumption is load-bearing, because stimulus
encoders are optimised against these simulators. We asked which properties of
primary visual cortex, if any, break it, and whether that matters for how a
device should be programmed.

**Approach.** We implemented a forward model adding two properties the
existing models omit — long-range horizontal connections between
iso-orientation domains, and divisive normalisation — within the
`pulse2percept` framework, and calibrated every configuration to reproduce the
same published single-electrode phosphene size-versus-current data as the
Dynaphos baseline. The models therefore agree on all measured single-electrode
behaviour and differ only on multi-electrode stimulation. Percepts were
predicted using a decoder fitted to the model's own normal-vision pathway
rather than assumed from retinotopy.

**Main results.** Linear mechanisms cannot break electrode independence.
Orientation-selective horizontal spread contributes exactly zero interaction
(< 10⁻⁹) at every separation and gain tested, because a linear operator
applied to a sum is the sum of its applications. Divisive normalisation does
break it, producing suppressive interaction of 19–44% peaking at 1.2–1.6 mm
separation and decaying to noise by ~4.5 mm at 300 µA. A 32-fold change in
normalisation strength moves the magnitude roughly two-fold and leaves the
length scales unchanged, while single-electrode psychophysics cannot constrain
that strength at all. Compared with published human multi-electrode data at
matched stimulation currents, the model predicts 38.9% interaction at the
separation where phosphenes merge and 0.0% where two distinct phosphenes are
reported, with no parameter adjusted to fit. Optimising stimulation against
the independent model costs only 0.6% of pattern fidelity but delivers
percepts at 39% of intended contrast — a shortfall a brightness-sensitive
objective improves only to 44%, at a cost in pattern fidelity, and a
scale-invariant objective cannot detect.

**Significance.** Electrode interaction in cortex is a dynamic rather than a
geometric phenomenon, so it cannot be captured by adding anatomical detail to
a linear forward model. Its practical consequence is a brightness shortfall
that current encoder-evaluation practice would not reveal.

---

## 1. Introduction

A cortical visual prosthesis stimulates primary visual cortex directly,
bypassing the eye and optic nerve. It is the only route to restored vision
when the retina or optic nerve is destroyed — glaucoma, optic neuropathy,
trauma — a larger population than retinal implants can serve. Chronic
phosphene generation has been demonstrated safely over 19 months with a
96-channel intracortical array [1, 2], and Phase I trials of wireless
intracortical devices are underway [3]. Recent work predicts phosphene
threshold, brightness and number from the activity of neighbouring electrodes
[4].

Turning stimulation into useful vision is an encoding problem: given a camera
image, which electrodes fire and at what amplitude. The field solves it by
optimising an encoder against a differentiable forward model of what
stimulation produces [5–8]. For cortex the state of the art is Dynaphos [9],
distributed within `pulse2percept` [10]. Dynaphos models retinotopy via a
reverse wedge-dipole map, cortical magnification, current spread, and
brightness and threshold as functions of stimulation parameters. It does not
model cortical circuitry, and its documentation states the consequence: each
electrode produces a Gaussian blob, and electrodes are independent.

There is precedent for doubting that. Retinal prosthesis simulation made the
same simplification and it failed: an epiretinal electrode also activates the
ganglion-cell axon bundles passing over it, so percepts are elongated rather
than round and patterns intended as letters become illegible (figure 1(a),
(b)) [10]. The correction was anatomical — something the geometric model did
not contain.

The assumption is also known to be imperfect empirically. Bosking *et al.*
stimulated 2–6 subdural surface electrodes simultaneously in 11 volunteers and
found that concurrent stimulation yields phosphenes reliably smaller than
independent stimulation, and that some pairs produce one phosphene where two
are expected [15]. What has been missing is a mechanism, and a forward model
that expresses one.

The obvious cortical analogue of the retinal effect is the long-range
horizontal connection system: superficial pyramidal axons running for
millimetres parallel to the surface and terminating in clusters on columns of
similar orientation preference [11, 12]. We set out to test whether these
produce the cortical equivalent. They cannot, for a reason that turns out to
be more informative than the hypothesis.

## 2. Methods

### 2.1 Baseline and verification of the independence assumption

All comparisons use `pulse2percept` 0.9.1 with the `Cortivis` implant geometry
and `Polimeni2006Map` retinotopy. The baseline is `cortex.DynaphosModel` with
default parameters, driven by 300 Hz biphasic pulse trains (170 µs per phase,
100 ms burst).

We verified independence empirically rather than assuming it from
documentation. For two electrodes 0.8° apart in the visual field at 200 µA,
the maximum relative deviation from linear superposition was 3.7 × 10⁻⁹
(figure 1(c), (d)) — floating-point noise. Superposition holds exactly.

### 2.2 Circuit model

`CircuitSpatial` is implemented as a `pulse2percept` spatial model so that the
comparison is like-for-like. Activation is computed on a uniform cortical
sheet (50 µm resolution) and read out at each visual-field grid point through
the same retinotopy the implant uses.

*Current spread* reproduces Dynaphos: activated tissue diameter
D = 2√(I/K). Activation magnitude additionally scales with a saturating
function of amplitude, which leaves the half-maximum contour, and therefore
the size calibration, unchanged.

*Orientation map.* Iso-orientation domains are generated by band-pass
filtering complex Gaussian noise and halving its phase [13], giving
quasi-periodic domains joined at pinwheels with a macaque V1 column spacing of
~750 µm (figure 2(a)).

*Horizontal connections.* Activation spreads to sites weighted by cortical
distance (σ = 1 mm) and orientation similarity (σ = 27°). Expanding the
orientation term to first Fourier order reduces this to three Gaussian
convolutions,

  G(A) + r·cos2θ·G(A cos2θ) + r·sin2θ·G(A sin2θ), all divided by (1 + r),

which keeps a millimetre-scale sheet tractable and the model differentiable.

*Divisive normalisation* [14]: each site's expansive response is divided by a
Gaussian-pooled measure of neighbourhood activity (σ = 1.5 mm). An optional
second, broader pool is described in section 3.5.

Both mechanisms can be disabled independently; with both off the model reduces
to pure current spread, the control condition that makes any difference
attributable to the circuit rather than to a different spread model.

### 2.3 Calibration

Every configuration was fitted, by adjusting K alone, to reproduce the
baseline's single-electrode phosphene diameter across 100–600 µA (figure
2(b)). Mean absolute error was 2.3% (circuit off), 2.4% (horizontal only),
3.7% (normalisation only) and 4.0% (both). All configurations therefore agree
with the published single-electrode data and with each other.

### 2.4 Percept decoder

To make claims about percepts rather than activation we fitted a readout. Each
cortical site is an oriented filter with a receptive-field position from the
retinotopy, an orientation from the map, and a spatial phase. The predicted
percept is the image that best explains the observed activity,

  minimise ‖M·I − a‖² + λ‖∇I‖²,

solved on the same forward operator used for encoding. We use this rather than
regressing images on activity: a ridge decoder achieves comparable
reconstruction on natural images (R² 0.65 against 0.66) but produces
structured speckle on stimulation, because it estimates E[image | activity]
under a training distribution that stimulation is nowhere near — visual input
drives a site only if the stimulus matches its tuning, whereas an electrode
drives every nearby site regardless.

Held-out reconstruction R² = 0.66 on 1/f noise images (figure 6(a)).
Reconstruction resolution optimises at ~0.15°/pixel and degrades either side.
This is the hypercolumn limit: a full set of orientations and phases occupies
~750 µm ≈ 0.18° of visual field at this eccentricity, and sampling cortex more
finely does not help because neighbouring sites carry different orientations
rather than repeating a measurement.

### 2.5 Optimal stimulation

For encoder consequences we avoided training a network, which would confound
forward-model accuracy with network capacity. For each target we solved
directly for the amplitude vector maximising percept fidelity under a given
forward model (L-BFGS-B, 24 electrodes, 0–600 µA), which is the best any
encoder of any architecture could achieve given that model.

Targets were defined as percepts the device can actually deliver. An initial
attempt using geometric shapes produced fidelities of 0.001–0.06; fitting
those targets with an unconstrained linear combination of single-electrode
percepts reached only 0.03–0.15, establishing that a couple of dozen
phosphenes cannot render a thin bar in a 256-pixel image at any amplitude.
Asking what a modelling error costs is meaningless when there is no fidelity
to lose.

Pattern fidelity is reported as squared correlation and percept brightness as
RMS contrast relative to target, with both models anchored to identical
single-electrode brightness so that the comparison is not an artefact of
differing activation scales.

## 3. Results

### 3.1 Linear mechanisms cannot break independence

Orientation-selective horizontal spread produces no departure from
superposition: below 10⁻⁹ at every separation (400–2400 µm) and gain
(0.3–1.2) tested (figure 3(a), dashed line at zero).

The reason is structural rather than numerical. Horizontal spread as
implemented — and in any convolutional formulation — is a linear operator, and
L(A + B) = L(A) + L(B) by definition. No connectivity rule, however
orientation-selective or anatomically detailed, can create electrode
interaction if it acts linearly.

This was contrary to our expectation and disanalogises the cortical case from
the retinal one. The axon-bundle effect is *geometric*: it changes the shape
of a single phosphene, and a linear model captures it, which is why the
axon-map model works. A cortical interaction effect must be *dynamic*, and no
linear extension of any current baseline can express it.

### 3.2 Divisive normalisation does break it

With normalisation enabled, departure from superposition at 300 µA per
electrode reaches 24.7% and is suppressive — a pair produces less than the sum
(figure 3(a), table 1). It peaks near the pooling scale and is negligible
beyond ~4.5 mm. Cortivis electrode pitch is 400 µm and the array spans ~4 mm,
so the affected range is the range real arrays occupy. Perceptually, two
electrodes 1.44 mm apart that the baseline renders as two distinct spots merge
into a single elongated percept (figure 3(b)–(d)).

**Table 1.** Departure from linear superposition (%), 300 µA per electrode.

| separation (mm) | 0.4 | 0.8 | 1.2 | 1.6 | 2.4 | 3.2 | 4.5 |
|---|---|---|---|---|---|---|---|
| Dynaphos baseline | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| circuit off | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| horizontal only | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| normalisation only | 11.8 | 10.8 | 18.9 | 22.4 | 15.6 | 7.1 | 1.2 |
| full circuit | 18.0 | 14.9 | 22.1 | **24.7** | 17.0 | 8.0 | 1.4 |

### 3.3 The conclusion survives not knowing the parameters

Normalisation strength is not calibrated by any public data. Sweeping it over
32-fold (0.25–8) and refitting K at every point so that all models still
reproduce the single-electrode curve, the peak deviation spans 19–44% while
the local-peak separation stays at 1.2–1.6 mm and the half-decay distance at
2.2–2.8 mm (figure 4(a), (b)). Sweeping the pooling extent instead walks the
peak from 400 µm to 2000 µm (figure 4(c)), confirming that geometry is set by
the pool and magnitude by the gain, separably.

The smallest gain is an outlier on both counts and we report it rather than
smooth it: refitting K at each sweep point means gain and operating point are
not fully independent. The monotonic trend from 0.5 to 8 is the interpretable
part.

Every strength from 0.25 to 8 fits the single-electrode curve to within 3–4%.
**Single-electrode psychophysics therefore cannot constrain normalisation
strength**, and multi-electrode measurement is the only route to it.

### 3.4 Comparison with human multi-electrode data

The model was built without reference to multi-electrode measurements, so
comparison with them is a test rather than a fit.

Bosking *et al.* [15] stimulated 2–6 subdural surface electrodes
simultaneously in 11 volunteers, analysing 232 pairs. For pairs perceived as
two distinct phosphenes, sizes under concurrent stimulation were consistently
smaller than under independent stimulation (2.49° against 3.59°,
*t*(34) = 8.14, *p* < 0.001, n = 35); a second class produced one phosphene
where two were expected. Both are suppressive interactions, in the direction
divisive normalisation predicts and in a direction no independent model can
produce.

Comparing spatial scale requires matching stimulation regime. Those
experiments delivered 0.3–4.0 mA; our intracortical predictions were computed
at 300 µA, and under the model's own current-spread law the activated radius
grows as √I, from 0.70 mm to 2.57 mm across that range. Sweeping amplitude and
changing nothing else (figure 5, table 2), at the top of their range the model
predicts strong interaction where percepts merge and overlap both individual
phosphenes (38.9% at 4.11 mm) and none where two distinct phosphenes are
reported (0.0% at 26.36 mm). It separates the regimes correctly with no
parameter adjusted to make it do so.

**Table 2.** Model interaction (%) at the mean separations reported in [15],
as a function of stimulation amplitude.

| stimulation | 4.11 mm (merged) | 9.57 mm | 26.36 mm (two phosphenes) | decay to 10% |
|---|---|---|---|---|
| 0.3 mA | 2.5 | 0.0 | 0.0 | 4.4 mm |
| 1.0 mA | 12.0 | 0.0 | 0.0 | 5.2 mm |
| 2.0 mA | 22.5 | 1.0 | 0.0 | 6.4 mm |
| 4.0 mA | **38.9** | 3.2 | **0.0** | 8.8 mm |

That study measured phosphene number, size, location and eccentricity, but not
brightness, so our central percept prediction (section 3.8) is neither
supported nor contradicted by it.

### 3.5 Long-range single-phosphene suppression is not normalisation

One class of pairs in [15], at a mean separation of 13.47 mm (n = 9), yields
one phosphene with the other unreported. The model predicts under 2%
interaction there.

V1 surround suppression comprises a near component with the reach of
horizontal connections and a far component mediated by extrastriate feedback
with much larger receptive fields. We added the second as an independent pool
and asked whether it can produce suppression at 13.5 mm while preserving the
regimes of section 3.4. It can, but only at one tested setting: a pool of
σ = 8 mm with **thirty times** the gain of the near pool (40.3%, 16.6%, 0.6%
at 4.11, 13.47 and 26.36 mm). Every physiologically ordered setting, in which
far suppression is weaker than near, produces at most 2–6% at 13.5 mm.

The arithmetic is straightforward: two electrodes 13.5 mm apart contribute
exp(−13.5²/2σ²) ≈ 0.24 of peak to each other's pool at σ = 8 mm, so for that
to change the response by more than 15% the pooled term must dominate the
semi-saturation constant, which requires large gain. We make no stronger
structural claim, since the dependence of interaction on pool width is not
monotonic and at short separations a broader pool increases rather than
decreases the departure from superposition.

We do not fit this. The original authors do not claim the effect is cortical:
they raise non-uniform spatial attention as a candidate and state that
resolving it will require direct measurement of activity in early visual areas
[15]. A forward model of cortical activation should not be expected to
reproduce a percept that may be selected after cortex. Section 4.1 proposes an
experiment that decides the question.

### 3.6 Interaction range scales with stimulation current

Holding all parameters fixed and varying only amplitude, the distance at which
interaction has decayed to a tenth of its peak grows from 4.4 mm at 0.3 mA to
8.8 mm at 4.0 mA (figure 5, table 2). The mechanism is transparent: activated
radius goes as √I, and two electrodes interact when their activation overlaps
the same normalisation pool.

The spatial extent of electrode interaction is therefore not a fixed property
of cortex but a function of how hard the device is driven. Interaction
measured at threshold currents will understate interaction at the
suprathreshold currents a device uses.

### 3.7 Percept-level consequences

Decoding five simultaneously driven electrodes, activation changes by 32.1%
relative to the independent prediction and the decoded percept by 31.2%
(figure 6(b)). The interaction is neither washed out by the readout nor
amplified; it transfers approximately one-for-one, taking the form of a fairly
uniform suppression of the whole percept rather than a distortion of its
shape.

### 3.8 Encoder consequences: the pattern survives, the brightness does not

Optimal stimulation computed under each forward model and cross-evaluated
(figure 7) gives a mean pattern fidelity of 0.995 when programmed for the
circuit and delivered to it, against 0.990 when programmed as independent and
delivered to the circuit — 0.6% forgone. On this axis the result is a null:
the circuit's suppression is a smooth, largely invertible distortion that a
re-optimised encoder undoes.

On brightness it is not. The delivered percept carries 39% of the intended
contrast. An objective that actively seeks brightness recovers it only to 44%
and sacrifices pattern fidelity (0.995 → 0.949) to do so. The amplitude bound
is never reached, so this is not a current limit; it is what divisive
normalisation is. Driving harder recruits more of the pool that divides, so
the suppression cannot be out-driven.

### 3.9 Neither metric rescues the brightness

Both conclusions come from the same percepts and differ only in scoring.
Squared correlation is invariant to scale: a percept of correct shape at 39%
brightness scores identically to one at full brightness. An evaluation using
it alone would report near-perfect success while delivering under half the
intended contrast.

This is a caution about evaluation practice rather than a criticism of
published encoders, several of which optimise scale-sensitive reconstruction
losses [5, 6]. But a scale-sensitive objective does not solve the problem, it
only reveals it, as the 0.39 → 0.44 recovery shows. The shortfall is a
property of the circuit, not of the metric; the metric determines only whether
one notices.

## 4. Discussion

### 4.1 Proposed experiments

**Interaction range should scale with current.** The distance over which
electrodes interact is predicted to grow as √I, from ~4.4 mm at 0.3 mA to
~8.8 mm at 4.0 mA. Repeating a paired-stimulation protocol at two
well-separated amplitudes should move the separation at which merging gives
way to two distinct phosphenes. Independent models predict no interaction at
any current, and a fixed-radius interaction model predicts no shift with
amplitude, so the three are cleanly distinguishable.

**Shape of the function at intracortical pitches.** At 300 µA the model
predicts interaction is not monotonic in separation, with an interior maximum
near 1.2–1.6 mm and weaker effects at both 0.4 mm and 3 mm. Measuring
phosphene size or brightness at three or four separations spanning 0.4–3 mm
would test this. Because the length scales are parameter-independent while the
magnitude is not, the shape tests the model and the magnitude estimates
normalisation strength — the separation of peak suppression measuring the
cortical normalisation pool extent, a quantity not otherwise accessible in
humans.

**Is the lost phosphene lost in cortex or after it?** For pairs where one of
two phosphenes goes unreported, measure trial-to-trial consistency. Consistent
loss of the same phosphene implicates a cortical mechanism, which section 3.5
shows is not divisive normalisation. Alternation implicates perceptual
competition and places the effect outside a forward model of cortical
activation. This is the cheapest of the three experiments.

**Ensemble brightness.** Drive N electrodes each individually producing
brightness B. Reported ensemble brightness should fall increasingly below N·B
as N grows, and increasing current should fail to close the gap. Independent
models predict brightness adds and have no mechanism for a ceiling current
cannot overcome.

### 4.2 Implications for device programming

If the cortex normalises, encoders built on independent forward models get the
pattern approximately right and the brightness substantially wrong, and no
choice of objective recovers it. Three practical consequences follow: treat
ensemble-brightness budgets derived from independent forward models as
optimistic; report a scale-sensitive metric alongside any correlation-based
one so that a shortfall is visible; and do not expect increased current to
close it, because the mechanism that suppresses also scales with drive.

The corollary for modelling is sharper. Because no linear mechanism can
produce electrode interaction (section 3.1), a forward model intended to
capture it must include an explicit nonlinearity. Adding anatomical detail to
a linear model — finer retinotopy, richer connectivity, patient-specific maps
— cannot help, however accurate that detail is.

### 4.3 Limitations

- **Normalisation strength is unconstrained.** Section 3.3 shows existing
  single-electrode data cannot pin it. The 24.7% and 39% figures are
  contingent; the direction, mechanism and length scales are not.
- **Long-range single-phosphene suppression is deliberately not modelled**
  (section 3.5). If that effect proves consistent trial-to-trial, the model
  owes an explanation.
- **The percept decoder has never been compared to a human percept.** It is
  trained on the model's own normal-vision pathway and inherits that pathway's
  inaccuracies. It answers what image is most consistent with a given
  cortical activity pattern according to this model of V1.
- **A field model, not a spiking one**, chosen to keep the comparison with
  Dynaphos like-for-like and the parameter sweeps tractable. Adaptation, spike
  timing and temporal dynamics beyond the baseline's are absent.
- **Different electrode regime from the available human data**, which uses
  subdural surface contacts (0.5 mm diameter, 2–6 mm spacing) delivering
  current through CSF and dura, against intracortical arrays at 400 µm pitch
  here.
- **One eccentricity, one array geometry, one hemisphere** (~1.8°
  eccentricity, `Cortivis`).
- **Deafferented cortex is not modelled.** A blind volunteer's V1 has had no
  input for years and is homeostatically altered; parameters here are from
  healthy cortex.

## 5. Conclusion

Electrode independence is assumed by every current cortical prosthesis
simulator and is exactly true in them by construction. Human data already
shows it is false: concurrently stimulated electrodes produce measurably
smaller phosphenes than independent ones. We find that no linear mechanism can
account for this, including the horizontal connectivity that seemed the
natural cortical analogue of the retinal axon-bundle effect, and that divisive
normalisation can — reproducing both the direction of the observed suppression
and, once stimulation currents are matched, the separations at which percepts
merge and at which they remain distinct, with no parameter adjusted to fit.
Suppression at intermediate separations remains unaccounted for and may not be
cortical in origin.

The consequence for device programming is not a loss of pattern fidelity,
which re-optimisation recovers, but a brightness shortfall that no objective
recovers, because the drive intended to overcome a divisive nonlinearity also
feeds it.

---

## Figure captions

**Figure 1.** *The independence assumption and the precedent for doubting it.*
(a) A seven-electrode pattern on an Argus II retinal implant rendered by a
scoreboard model, in which each electrode is an independent round phosphene.
(b) The same electrodes under an axon-map model that includes activation of
passing ganglion-cell axon bundles; the pattern is no longer legible. (c) Two
Cortivis electrodes driven together at 200 µA under the Dynaphos cortical
baseline. (d) The difference between (c) and the sum of the two electrodes
driven separately: 3.7 × 10⁻⁹ of the peak, i.e. floating-point noise.
Superposition holds exactly in the cortical baseline by construction.

**Figure 2.** *Model and calibration.* (a) Simulated V1 orientation preference
map, generated by band-pass filtering complex Gaussian noise, with a column
spacing of 750 µm. (b) Phosphene diameter against stimulus amplitude for the
Dynaphos baseline (black) and for all four configurations of the circuit
model after fitting the excitability constant alone. Legend gives mean
absolute error against the baseline. Every configuration reproduces the same
single-electrode data, so subsequent differences between them concern only
multi-electrode stimulation.

**Figure 3.** *Only a nonlinearity breaks electrode independence.* (a)
Departure from linear superposition against electrode separation, 300 µA per
electrode. The circuit-off and horizontal-only curves lie on zero and on each
other; the latter is dashed to remain visible. Orientation-selective
horizontal spread contributes nothing because it is linear. (b)–(d) Cortical
activation for two electrodes 1.44 mm apart: the prediction if the electrodes
were independent (b), the circuit model's prediction (c), and their difference
(d). Two distinct maxima merge into a single elongated region.

**Figure 4.** *Robustness to the uncalibrated normalisation parameters.* (a)
Interaction against separation for normalisation pool gains spanning 32-fold,
with the excitability constant refitted at every point so that all models
still reproduce the single-electrode curve. (b) Peak deviation (left axis) and
the separation at which the interior peak occurs (right axis) against pool
gain. Magnitude varies roughly two-fold; the length scale does not move. (c)
The same sweep over pool width instead, which walks the peak from 0.4 mm to
2 mm.

**Figure 5.** *Interaction range scales with stimulation current.* Departure
from independence against separation at four amplitudes spanning the range
used in human surface-stimulation experiments, with all other parameters
fixed. Dotted lines mark the mean separations reported in [15] for pairs whose
phosphenes merged and for pairs perceived as two distinct phosphenes. At 4 mA
the model predicts strong interaction at the former and none at the latter.

**Figure 6.** *Percept prediction.* (a) Held-out validation of the decoder:
1/f noise images shown to the model's normal-vision pathway (top) and
recovered from the resulting cortical activity (bottom); R² = 0.66,
estimated over 24 held-out images of which four are shown. (b)
Predicted percepts for five electrodes, comparing the sum of the electrodes
driven individually with all five driven together, under the independent
configuration (top) and the circuit (bottom). The circuit suppresses the
percept approximately uniformly.

**Figure 7.** *Encoder consequences.* (a) Pattern fidelity for six deliverable
target percepts, comparing stimulation optimised for the circuit and delivered
to it against stimulation optimised as though electrodes were independent and
delivered to the circuit. The difference is 0.6%. (b) Percept contrast
relative to target: as delivered, and after re-optimising an objective that
explicitly seeks brightness. The shortfall is largely irrecoverable.

---

## Data availability

All code, the test suite, and scripts regenerating every figure and number in
this manuscript are available in the accompanying repository.
`prosthesis/run_all.py` regenerates all results; `paper/make_figures.py`
regenerates all figures; `paper/verify_claims.py` asserts every quantitative
claim above against the regenerated output.

## References

[1] Fernández E *et al.* 2019 Phosphene perceptions and safety of chronic
visual cortex stimulation in a blind subject
https://pubmed.ncbi.nlm.nih.gov/31151104/

[2] Development of a Cortical Visual Neuroprosthesis for the Blind
ClinicalTrials.gov NCT02983370

[3] A Phase I Feasibility Study of an Intracortical Visual Prosthesis (ICVP)
for People with Blindness ClinicalTrials.gov NCT04634383

[4] 2025 Neural correlates of phosphene perception in blind individuals: a
step toward a bidirectional cortical visual prosthesis *Sci. Adv.*
https://www.science.org/doi/10.1126/sciadv.adv8846

[5] Granley J *et al.* 2022 Deep learning-based perceptual stimulus encoder
for bionic vision
https://bionicvisionlab.org/publications/2022-03-deep-stimulus-encoder/

[6] de Ruyter van Steveninck J *et al.* 2022 End-to-end optimization of
prosthetic vision https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8899855/

[7] 2022 Optimization of neuroprosthetic vision via end-to-end deep
reinforcement learning https://pubmed.ncbi.nlm.nih.gov/36328967/

[8] 2025 End-to-end learning of safe stimulation parameters for cortical
neuroprosthetic vision
https://www.biorxiv.org/content/10.1101/2025.01.23.634543

[9] van der Grinten M *et al.* 2023 Towards biologically plausible phosphene
simulation for the differentiable optimization of visual cortical prostheses
*eLife* https://elifesciences.org/articles/85812

[10] Beyeler M *et al.* pulse2percept: a Python-based simulation framework for
bionic vision https://bionicvisionlab.org/code/pulse2percept/

[11] Gilbert C D and Wiesel T N 1983 Clustered intrinsic connections in cat
visual cortex *J. Neurosci.* **3** 1116–33

[12] Kandel E R *et al.* 2013 *Principles of Neural Science* 5th edn, ch 25

[13] Rojer A S and Schwartz E L 1990 Cat and monkey cortical columnar patterns
modeled by bandpass-filtered 2D white noise *Biol. Cybern.*

[14] Carandini M and Heeger D J 2012 Normalization as a canonical neural
computation *Nat. Rev. Neurosci.* **13** 51–62

[15] Bosking W H *et al.* 2022 Percepts evoked by multi-electrode stimulation
of human visual cortex *Brain Stimul.*
https://pmc.ncbi.nlm.nih.gov/articles/PMC9831085/
