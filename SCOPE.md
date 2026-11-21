# Cortical visual prosthesis: project scope

*Drafted August 2026. This is a plan, not a result. Every claim about the
existing literature is cited; every claim about what our model would show is
a hypothesis.*

---

## 1. The clinical problem

A cortical visual prosthesis stimulates V1 directly with an electrode array,
bypassing the eye and optic nerve entirely. That makes it the only option for
blindness where the retina or optic nerve is destroyed — glaucoma, optic
neuropathy, trauma, advanced diabetic retinopathy — which is a far larger
population than the retinitis pigmentosa cohort that retinal implants serve.

Stimulating an electrode produces a **phosphene**: a small spot of light at a
position in the visual field set by V1 retinotopy. The engineering problem is
the inverse one: *given an image from a camera, which electrodes do you fire,
at what amplitude, to make the person perceive that image?*

## 2. What already exists

The field is real, funded, and further along than a newcomer might assume.
Building anything without knowing this would waste months.

**Clinical.** Only a handful of blind people have received intracortical
visual prostheses, all late-blind. Fernández and colleagues demonstrated
19 months of safe, reliable phosphene generation with a 96-channel Utah array
([Nature/JCI line of work][fdz], [trial NCT02983370][nct1]). The Illinois
ICVP Phase I trial is implanting wireless floating microelectrode arrays in
five participants ([NCT04634383][nct2]). Recent work shows phosphene
thresholds and brightness can be predicted from neighbouring-electrode
activity — a step toward bidirectional, closed-loop devices
([Science Advances 2025][sciadv]).

**Simulation.** `pulse2percept` is a mature open-source Python framework for
*retinal* prostheses ([Bionic Vision Lab][p2p]). Its key scientific result is
a cautionary tale for us: retinal phosphenes are **not** pixels. Electrodes
incidentally activate passing axon bundles, so percepts come out elongated
and comet-shaped, and the naive "implant = low-res screen" assumption is
simply wrong ([Beyeler et al.][p2p]).

**Cortical simulation.** The state of the art is van der Grinten et al.,
*eLife* 2023 ([link][elife]) — a differentiable, PyTorch-based cortical
phosphene simulator built for gradient-based encoder optimisation. Critically,
it is **geometric and phenomenological**: reverse wedge-dipole retinotopy,
cortical magnification, current-spread physics, brightness/threshold curves,
and a leaky-integrator temporal model. It does *not* model V1 circuitry — no
orientation columns, no lateral connections, no normalisation. The authors
state plainly that it "assumes independence between electrodes" and list
"multi-phosphene interactions" and "feature-specific percepts" as future work.

**Encoder optimisation.** The frontier is end-to-end deep learning: train a
CNN encoder against a differentiable phosphene simulator so that the decoded
percept matches the target image ([deep stimulus encoder][dse],
[end-to-end optimisation][e2e], [RL variant][rl],
[safe stimulation parameters][safe]).

## 3. The gap

Put those last two together and a problem appears.

> The encoders are optimised against a forward model that assumes electrodes
> act independently and that V1 contributes nothing but a retinotopic map.
> V1 demonstrably has orientation columns, strong lateral connectivity between
> like-oriented sites, and divisive normalisation. If those circuit properties
> make the cortical response to a *multi-electrode* pattern differ from the sum
> of its single-electrode responses, then every encoder trained this way is
> optimised against the wrong objective.

This is exactly the mistake the retinal field already made once and corrected:
axon-bundle activation broke the pixel assumption. Nobody has done the
equivalent circuit-level check for cortex.

**The question, stated so it can be answered:** *Does V1 circuit structure
produce systematic, predictable deviations from electrode independence — and
if so, are they large enough to change how stimulation should be encoded?*

That is tractable computationally, needs no patients to *ask*, and is
falsifiable by existing implanted-patient labs.

## 4. What we can reuse, and what we cannot

We built a spiking V1 hypercolumn (`column/`). Honest accounting:

**Directly reusable**
- `network.py` — sparse projections, axonal delay buffers, integration loop.
  Tested and fast enough.
- `neurons.py`, `synapses.py` — AdEx cell classes, conductance-based receptors,
  short-term plasticity. Standard and sound.
- `connectivity.py` — *the* most relevant piece. Orientation-tuned
  (like-to-like) horizontal connections and distance-dependent rules are
  precisely the mechanisms that would break electrode independence.
- `analysis.py` — population decoding.
- The normalisation work (`presyn_norm_k`). Stimulation has the same
  dynamic-range problem we hit: fixed-threshold circuits are blind at low
  drive and saturate at high drive. We already have a mechanism that helps.
- The test discipline. Two measurement bugs got caught by tests; this project
  will have more.

**The LGN front-end becomes the measuring instrument, not dead weight.**
This is the neat part. To say what a stimulation pattern "looks like" you need
a mapping from V1 activity to a perceived image. Our normal-vision pathway
(image → LGN → V1) supplies exactly the dictionary: simulate V1 responses to
many natural images, fit a decoder image ← V1 activity, then apply that
decoder to *stimulation-evoked* activity. The predicted percept is then "the
image that would have produced this V1 response," which is a well-posed
inverse problem rather than hand-waving.

**Must be built new**
- **Electrode → tissue activation.** New physics: current spread, distance–
  threshold curves, and — the retinal lesson — *axonal* vs somatic activation.
  Stimulating a cortical site activates passing axons, including the
  long-range horizontal fibres. This is the single most likely source of
  surprise.
- **A multi-hypercolumn patch.** We have one ~0.5 mm pinwheel. Arrays span
  several millimetres and many hypercolumns. Needs a proper wedge-dipole
  retinotopic map, not our idealised ring.
- **Deafferented cortex.** Our model is a healthy V1 driven by the LGN. A
  blind patient's V1 has had no input for years and is homeostatically
  hyperexcitable. Different operating point, and it matters.
- **Recalibration against stimulation psychophysics.** Non-negotiable, see §6.

**Must be discarded**
- Our hand-tuned operating point. It was fit to reproduce Hubel-Wiesel
  results, not stimulation data. Keeping it would be building on sand.

## 5. Phased plan

Timings assume one person, part-time, and are the honest kind (i.e. likely
optimistic by 50%).

### Phase 0 — Orient ✅ **done**
See [`prosthesis/PHASE0_FINDINGS.md`](prosthesis/PHASE0_FINDINGS.md) and
`prosthesis/phase0_baselines.py`. Three results, one of which revises the plan:

- **pulse2percept 0.9 already ships cortical support** — Dynaphos itself, the
  Cortivis/ICVP/Orion/Neuralink implants, wedge-dipole retinotopy
  (`Polimeni2006Map`), and patient-specific maps from MRI (`NeuropythyMap`).
  So: don't build retinotopy, don't build patient-specific anatomy, and build
  the circuit layer *as a pulse2percept model* so the comparison is exact and
  the work is contributable upstream.
- **The retinal cautionary tale reproduces cleanly.** Same electrodes: a
  legible letter under the pixel assumption, an illegible smear once axon
  bundles are included.
- **The cortical baseline is exactly linear.** Measured deviation from
  superposition: 3.7 × 10⁻⁹ — floating-point noise. Not approximately
  independent; independent by construction. That is the number this project
  exists to challenge.

### Phase 1 — Stimulation front-end (3–5 weeks, revised down)
Implement the circuit layer as a pulse2percept `SpatialModel` subclass so it
drops into the same API as Dynaphos. Use `Polimeni2006Map` for retinotopy
rather than our idealised pinwheel. Build the electrode → tissue activation
front-end, including **axonal recruitment of horizontal fibres** — on the
retinal precedent that is the likeliest source of a real effect. Set the
deafferented operating point.
**Deliverable:** single-electrode stimulation produces a localised V1
activation whose position follows retinotopy and whose spread grows with
current.
**Gate:** phosphene size vs. current and vs. eccentricity must match published
curves within a stated tolerance. *If it cannot be made to, stop and
reconsider — see §8.*

### Phase 2 — The percept decoder (3–4 weeks)
Build the image ← V1 decoder described in §4 using the existing normal-vision
pathway.
**Deliverable:** a function turning simulated V1 activity into a predicted
percept image, validated by round-tripping natural images.

### Phase 3 — The core experiment (6–8 weeks)
The actual question. Stimulate single electrodes, then pairs, then patterns.
Measure deviation from linear superposition as a function of:
- electrode separation (does it track the horizontal-connection length scale?)
- relative position on the orientation map (same column vs. across a pinwheel)
- total current (does normalisation make brightness sublinear in electrode
  count?)
- timing (simultaneous vs. interleaved pulses)

**Deliverable:** a quantified interaction map, and a stated effect size for
how far multi-electrode percepts depart from independence.
**This is the publishable unit**, positive *or* negative. A rigorous null —
"circuit structure does not meaningfully break independence at clinically
relevant electrode spacings" — would be genuinely valuable, because it
licenses the simpler models the field already uses.

### Phase 4 — Does it change encoder design? (4–6 weeks)
Train an end-to-end encoder against the independent-electrode model, then
evaluate it under the circuit model, and vice versa. Quantify the penalty.
**Deliverable:** either "the simple model is fine, carry on" or "encoders are
leaving X% of percept fidelity on the table," with the mechanism identified.

## 6. Validation strategy

This is what separates the project from the last one. Non-negotiable:

1. **Calibrate against published psychophysics, not intuition.** Phosphene
   size vs. eccentricity and vs. current (Bosking et al.), threshold
   distributions, brightness–amplitude curves. These are published and
   quantitative. Phase 1's gate is a fit to them.
2. **Hold out.** Fit on some conditions, predict others. Report held-out
   error, not fit quality.
3. **Make falsifiable, cheap-to-test predictions.** The deliverable that
   earns collaborators is a short list of the form: *"stimulate this electrode
   pair at this separation and the reported phosphene will be elongated along
   the local orientation-preference axis by roughly this much."* Labs with
   implanted patients can test that in an afternoon.
4. **Never claim a percept the model cannot see.** The model predicts neural
   activity. The step from activity to percept is an inference, and every
   figure should label which side of that line it is on.

## 7. What would make it publishable

Realistic venues: *Journal of Neural Engineering*, *IEEE TBME*, *eLife* (where
the cortical simulator itself landed), or NeurIPS/ICLR workshops for the
encoder work.

What it needs that the column model lacked:
- Quantitative fit to published data with held-out validation
- A pre-registered question with a falsifiable answer
- Comparison against the existing baseline model, not against nothing
- Open code and reproducible figures (this we already do well)

## 8. Risks and kill criteria

Stated in advance so they can't be rationalised away later.

| Risk | Kill / mitigation |
|---|---|
| Phase 1 can't fit published phosphene size/threshold curves | **Kill criterion.** If a circuit model can't reproduce the basic phenomenology the geometric model already gets right, it has no standing to critique it. Fall back to contributing a stimulation module to `pulse2percept`. |
| Interaction effects turn out negligible | Not a failure — publish the null. But decide the effect-size threshold *before* running Phase 3. |
| Deafferented-cortex parameters are unconstrained by data | Real risk; the data are thin. Mitigate with sensitivity analysis: show conclusions hold across a plausible range, or report which conclusions depend on it. |
| The percept decoder is circular (assumes what it should show) | Guard: validate the decoder only on normal-vision round-trips, never tune it on stimulation results. |
| Solo work without patient access | Accept. The output is predictions *for* labs with access, not clinical claims. Identify collaborators early (Fernández/Miguel Hernández; Illinois ICVP; Bionic Vision Lab for methods). |

## 9. What this project is not

- Not a device, and not a path to one without a lab, hardware, and trials.
- Not clinically actionable on its own.
- Not a replacement for `pulse2percept` or the eLife simulator — it is a
  circuit-level *layer* whose value is proving whether that layer matters.

---

[fdz]: https://pubmed.ncbi.nlm.nih.gov/31151104/
[nct1]: https://clinicaltrials.gov/study/NCT02983370
[nct2]: https://clinicaltrials.gov/study/NCT04634383
[sciadv]: https://www.science.org/doi/10.1126/sciadv.adv8846
[p2p]: https://bionicvisionlab.org/code/pulse2percept/
[elife]: https://elifesciences.org/articles/85812
[dse]: https://bionicvisionlab.org/publications/2022-03-deep-stimulus-encoder/
[e2e]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8899855/
[rl]: https://pubmed.ncbi.nlm.nih.gov/36328967/
[safe]: https://www.biorxiv.org/content/10.1101/2025.01.23.634543.full.pdf
