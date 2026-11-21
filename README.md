# Electrode interaction in cortical visual prostheses

A forward model of intracortical microstimulation built from published
measurements, used to ask when electrodes in a cortical visual prosthesis
stop acting independently, and what that would cost a device. A second,
phenomenological model stands in for the alternative hypothesis, a cortex
in which every electrode feeds one wide divisive pool.

Every current cortical prosthesis simulator treats electrodes as
independent. Each produces a Gaussian phosphene and percepts add linearly.
Encoders are optimised against those simulators, so a lot rests on the
assumption. The manuscript is `paper/manuscript_v2.tex`, "Necessary
Conditions for Electrode Interaction in Cortical Visual Prostheses".

## The short version

**Where a nonlinearity sits decides whether electrodes interact.**

- No *linear* mechanism can make electrodes interact. Orientation-selective
  horizontal spread, the obvious cortical analogue of the retinal
  axon-bundle effect, adds exactly zero at every separation and gain
  tested. That follows from linearity, not from a parameter choice.
- Neither can a nonlinearity applied to each electrode's own drive *before*
  the electrodes are summed. `f(I_a) + f(I_b)` stays additive however
  compressive `f` is. This is why Dynaphos departs from superposition by
  only 3.7 × 10⁻⁹ despite its saturating brightness and threshold functions.
- Interaction needs a nonlinearity acting **after** the electrodes combine.

**Cortex built from measured parts supplies two, and the effect is
modest.** The forward model (`prosthesis/cortex_ssn.py`,
`BiophysicalCortex`) takes recruitment and the network from published
physiology, with two fitted numbers, neither fitted to multi-electrode
data. A cell reached by two electrodes is recruited once (occupancy), and a
published stabilised supralinear network of layer 2/3 responds to the
combined drive. Occupancy does most of the work.

| prediction | measured parts | pooled divisor (alternative) |
|---|---|---|
| summed interaction, 0.4 mm, 100 µA | −2.6% | facilitatory at low drive |
| peak departure, 0.4 mm, 300 µA | 31% | 18% |
| where interaction peaks | shortest separation, in all 19 parameter variants | interior, near 1.5 mm |
| decay to a tenth, 0.3 mA | 2.5 mm | 4.4 mm |
| sign reverses with current | no, to 600 µA at ≤ 0.8 mm | yes, at 96–223 µA |
| later of two phosphenes dimmer | by at most 3.5% | by up to 39% |
| 100 electrodes, 0.4 mm, 300 µA | 46× one electrode | 3.9× one electrode |
| five electrodes, change in decoded percept | 11.5% | 32% |
| at 4.11 mm, up to 4 mA | at most 6.0% | 39% |

The model's single-electrode phosphene is 0.124° wide at half maximum at
100 µA, against 0.126° in Dynaphos, without being fitted to it.

**What the literature does not pin down.** Across nineteen parameter
variants the interaction at 300 µA and 0.4 mm stays suppressive and its
peak stays at the shortest separation. At 100 µA the sign is not settled.
The largest lever is how many inhibitory cells an electrode drives directly
(−7.6% with none, +9.8% at three times the calibrated density). That has
been measured once, in mouse. It is the measurement this work most needs.

**Telling the two hypotheses apart.** Measuring interaction at nine
separations from 0.4 to 4.5 mm with a standard error of 20 percentage
points at each identifies the right hypothesis 80% of the time, and three
separations (0.4, 1.6, 3.2 mm) tolerate the same error. Separation order needs
to be randomised, because session drift bends a pooled curve toward the
measured-parts shape.

## Results that go against the model

Kept visible on purpose.

- **Paired stimulation in mouse V1.** Kim *et al.* 2025 saw net
  enhancement at 12.7 µA, concentrated in cells that responded to neither
  electrode. The model puts attenuation on the directly driven cells and
  the modulation between the electrodes, as they found, but it enhances
  nothing. Its net is −0.23% at that current.
- **Spread of recruited somata.** Their median distance grows from 349 to
  372 µm between 5 and 50 µA, against 294 to 455 µm in Kumaravelu *et
  al.*'s 3-D simulations.
- **Phosphene growth with current.** The half-max diameter grows from
  0.124° at 100 µA to only 0.20° at 600 µA, where Dynaphos gives 0.30°. A
  doubled-firing-rate readout gives 1.1 to 1.5 times the Dynaphos size, so
  the two readouts bracket it.
- **Sequence position.** Armenta Salas *et al.* report a later,
  lower-amplitude pulse judged as bright as an earlier one in two of three
  Orion subjects. The model keeps a second phosphene within 3.5% of its
  brightness alone, so it neither fits nor contradicts them. A pooled
  divisor predicts the opposite ordering.
- **Human surface-electrode data.** Bosking *et al.* report merged pairs
  at 4.11 mm, lost phosphenes at 13.47 mm and pairs pushed 43% apart at
  26.36 mm. The model predicts at most 6% interaction at 4.11 mm even at
  4 mA, 0.21% at 13.47 mm and essentially nothing at 26.36 mm. It pushes
  activity centroids apart by up to 9.3% at 600 µA, but only within about
  2.4 mm. A pooled divisor reaches 13.47 mm only with an 8 mm far pool at
  30 times the near pool's gain, one setting of 24 tried.

## Corrections made when the experiments were rerun on this model

The figures in the earlier draft came from the phenomenological model,
which reproduces bit for bit. Rerunning everything on the measured-parts
model changed these claims.

- Every pooled-divisor experiment now reads its excitability from the
  phase-1 fit (589), where phases 2, 4 and 5 had carried a stale 608.
- A second phosphene stays within 3.5% of its brightness alone, not 3%.
- The pooled divisor's delay slope is 1.80τ, not 1.81τ.
- The five electrodes in the percept test are a zig-zag group 0.9–1.8 mm
  from their nearest neighbours, not a line 1.4 mm apart.
- The peak-position repulsion estimator now takes one maximum on each side
  of the midpoint. Taking the two largest maxima of the profile put both
  beside one electrode on this model's textured field and gave +1592%.
- Interaction range grows with current in both models, so the comparison
  is now made over the same 0.3–4 mA.
- Numbers with no generating script (a ridge decoder's R², a pixel scale,
  an old bar-fidelity range) were dropped or regenerated
  (`v2_paper.py reach`).

## Layout

```
column/         spiking cortical column model (AdEx, 7 cell classes)
prosthesis/     both forward models and every experiment
paper/          manuscript sources, figures, tables, and the claim checker
tests/          the test suite (pytest)
results/        JSON written by the experiments, read by the paper
```

### The forward model: built from measured parts

`prosthesis/cortex_ssn.py`, `BiophysicalCortex`.

- **Recruitment is axonal and sparse.** The electrode initiates spikes in a
  ball of radius `sqrt(I/K)`, with `K` derived from Kumaravelu et al. 2022.
  It recruits the cells whose axons have excitable sites inside that ball,
  so the driven population is scattered over hundreds of microns and fills
  in with current instead of spreading (Histed et al. 2009). A cell can be
  recruited once, so two electrodes reaching the same cells is itself a
  nonlinearity on the combined input.
- **The network is a stabilised supralinear network.** Obeid & Miller's
  published layer-2/3 model (eNeuro 2025), not retuned. Normalisation and
  surround suppression emerge from E–I dynamics instead of a formula.
- **Two numbers are fitted**: excitatory site density, to Histed's count of
  responding cells, and inhibitory site density, to the I/E recruitment
  ratio of Dadarlat et al. 2024. Everything else is a published value or a
  prediction.
- **Readout.** Cortical activity is mapped to the visual field through the
  same retinotopy as Dynaphos, and decoded into percepts through the
  model's own orientation map (`prosthesis/v2_paper.py`).

[`prosthesis/BIOPHYSICAL_MODEL.md`](prosthesis/BIOPHYSICAL_MODEL.md) lists
every parameter with its source and status, every check against data, what
the model leaves out, and the numerical bugs found while building it.

### The alternative hypothesis: a pooled divisor

`prosthesis/cortex_circuit.py`, `CircuitSpatial`, a `pulse2percept` spatial
model so comparisons with Dynaphos stay like-for-like. Current spread
reproduces Dynaphos exactly (`D = 2√(I/K)`), and two mechanisms can be
enabled independently: horizontal connections between iso-orientation
domains, and divisive normalisation whose pooling width is the parameter
that matters. Setting `norm_sigma = 0` collapses it to a pointwise
Naka-Rushton saturation, the control condition. Every configuration is
calibrated to the same published single-electrode size-versus-current curve
in phase 1, and later phases read that fit (`phase1_circuit.calibrated`).

### Reproducing everything

```bash
python3 prosthesis/run_all.py     # every experiment, in dependency order (hours)
python3 paper/make_figures.py     # all manuscript figures
python3 paper/v2_tables.py        # the manuscript tables
python3 paper/verify_claims.py    # assertions against the regenerated JSON
python3 -m pytest tests/ -q       # the test suite
```

`prosthesis/v2_paper.py` runs the measured-parts experiments the
manuscript reports, one part at a time if named (`size`, `bosking`,
`repulsion`, `percepts`, `reach`, `encoder`, `power`). The encoder part
alone takes a few hours, because every full-model evaluation runs the
network.

`verify_claims.py` exists because a manuscript drifts from its codebase
quietly. It asserts every quantitative claim in the paper against freshly
regenerated results and fails loudly on disagreement.

### Building the paper

```bash
cd paper && pdflatex manuscript_v2 && bibtex manuscript_v2 && pdflatex manuscript_v2 && pdflatex manuscript_v2
```

## Conventions

- Numbers in the manuscript are never typed by hand without a check. They
  are asserted by `verify_claims.py` against the JSON that produced them.
- Each stage of the measured-parts model is tested against the property it
  was taken from (`tests/test_biophysical.py`), and each readout against
  what it claims to measure (`tests/test_v2_paper.py`).
- Negative results stay in.
