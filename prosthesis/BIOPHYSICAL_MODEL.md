# The biophysical forward model

`prosthesis/cortex_ssn.py` (`BiophysicalCortex`) is the manuscript's forward
model of intracortical microstimulation. It is built from published
measurements. The phenomenological divisive-normalisation model in
`cortex_circuit.py` stays as the alternative hypothesis, a pooled divisor
with a free gain, which the paper uses to show what a cortex that pooled
activity across millimetres would do and how an experiment could tell the
two apart.

This document lists every number in the model, where it came from and how
confident we are in it, then the checks the model passes and fails, and then
what it leaves out.

## What the model does

```
electrode current I
  │  spike-initiation ball, radius R = sqrt(I / K_ax)
  ▼
excitable axonal sites inside the ball
  │  each cell's sites lie along its axon; its axon follows its outgoing
  │  connection kernel, so a cell at offset d has  s · (4/3)πR³ · F(d)
  │  expected sites in the ball (F = axonal footprint, unit integral)
  ▼
fraction of E and of I cells driven directly:  p = 1 − exp(−hits)
  │  hits from several electrodes add; a cell can be recruited only once
  ▼
directly driven cells transmit at least ρ, the rate at which a depressed
synapse driven at the pulse rate delivers the same output (a floor: an axonal
spike cannot be vetoed by somatic inhibition)
  ▼
stabilised supralinear network of E and I populations on a 2-D sheet with an
orientation map (Obeid & Miller 2025), started from spontaneous activity
  ▼
evoked excitatory output = steady-state E rate − spontaneous E rate
```

The first two stages follow Histed, Bonin & Reid (2009) and Kumaravelu et
al. (2022). The electrode excites axons, so the somata it recruits are sparse
and spread over hundreds of microns, and their number grows with current
much faster than their spread does. The network stage follows Ahmadian, Rubin
& Miller (2013) and Rubin, Van Hooser & Miller (2015). There, normalisation,
surround suppression and the shift from supralinear to sublinear summation
all come out of recurrent E–I dynamics instead of being written in as a
formula.

There are two nonlinearities after the electrodes combine, and both have a
physical origin. The first is **occupancy**: two electrodes that reach the
same cell recruit it once, not twice. The second is the **recurrent network**.
Both act on the combined input, which is the condition the manuscript's
section 3.1 shows any interaction mechanism must meet.

For phosphenes the evoked output is read out exactly as the Dynaphos
baseline is, sampled through the implant's retinotopy on a 0.02° visual-field
grid (`v2_paper.Retinotopy`). Percepts are decoded through a decoder built on
this model's own orientation map (`v2_paper.bio_code`).

## Parameters

Status: **published** = taken from a paper unchanged. **derived** = computed
from published numbers by stated arithmetic. **calibrated** = fitted to one
measurement, named. **assumed** = chosen, with the reason given, and swept in
the sensitivity analysis. **extension** = not in any source, added to cover a
regime the source never reached.

### Network (`SSNParams`), Obeid & Miller, eNeuro 2025

| name | value | status | note |
|---|---|---|---|
| `k` | 0.01 | published | power-law gain |
| `n` | 2.2 | published | power-law exponent, E and I |
| `tau_e`, `tau_i` | 10, 6.67 ms | published | match measured V1 decay of ~10 ms (Reinhold et al. 2015) |
| `ref_step_um` | 107 µm | published | grid interval the weights are quoted on (8 mm / 75) |
| `L_o` | 321 µm | published | dense E projection radius (3 grid intervals; their spiking section gives 324) |
| `sigma_ee` | 321 µm | published | Gaussian skirt of E→E beyond `L_o` |
| `sigma_ie` | 642 µm | published | Gaussian skirt of E→I beyond `L_o` |
| `sigma_i` | 214 µm | published | I projections, Gaussian |
| `J_ee_near`, `J_ie_near` | 0.072, 0.060 | published | within `L_o` |
| `J_ee_far`, `J_ie_far` | 0.036, 0.036 | published | beyond `L_o` |
| `J_ei`, `J_ii` | 0.0528, 0.0288 | published | |
| `A_near`, `B_near`, `sori_near` | 0.20, 0.80, 55° | published | orientation dependence, I and near E |
| `A_far`, `B_far`, `sori_far` | 0.14, 0.86, 25° | published | orientation dependence, far E |
| `map_period_um` | 1000 µm | published | 8 map periods across their 8 mm sheet |
| `r_max_e`, `r_max_i` | 200, 400 Hz | extension | soft ceilings; never reached by visual stimuli, so the validated regime is unchanged. Swept (removed) |
| `slow_inh_frac` | 0 | assumed | share of inhibition carried by a GABA-B-like trace. ICMS is followed by 100–150 ms of inhibition reduced by GABA-B blockade (Butovas & Schwarz 2003; Butovas et al. 2006). Steady states do not depend on it (tested); only the sequence experiment uses 0.3 |
| `tau_slow` | 100 ms | assumed | from the same duration |
| `harmonics` | 4 | numerical | orientation kernel expanded in cos 2mθ. Reconstruction error 2.5% of range for the wide (55°) kernel, which has a kink where it wraps; 0.07% for the narrow one (tested) |

Two deliberate departures from the published model:

1. **The orientation map is exactly periodic.** It is generated as half the
   phase of complex noise filtered onto a ring at the column frequency. That
   is the continuum version of Kaschube et al.'s superposition of plane waves,
   and it lets the sheet be convolved by FFT with no seam. Obeid & Miller's
   map is not periodic and has a discontinuity at the border.
2. **The weights are rescaled with grid resolution.** At step `h`, each
   connection carries `(h / 107)²` of its published weight, so the input a
   cell receives from a patch of cortex does not depend on the grid. The first
   version had this factor inverted. That was invisible on the 107 µm grid
   every experiment used, and wrong by 16× at half that step. The resolution
   test in `tests/test_biophysical.py` exists because of it.

### Recruitment (`RecruitmentParams`)

| name | value | status | source and arithmetic |
|---|---|---|---|
| `K_ax` | 3800 µA/mm² | derived | Kumaravelu et al. 2022 report median axonal spike-initiation radii of 26, 56 and 105 µm at 5, 20 and 50 µA. The median of a uniform ball is 0.794 R, so R = 33, 71, 132 µm, and K = I/R² = 4700, 4000, 2900. 3800 is their geometric mean. Compare 1292 µA/mm² (Stoney et al. 1968, pyramidal-tract somata) and 675 µA/mm² (Tehovnik), which both describe a ball of activated cell bodies, a different quantity. Swept: 1292 and 10000 |
| `site_density_e` | 9.0 /mm | calibrated | excitable sites per neuron per mm of depth, set so that about five cells in a two-photon optical section respond at 8.5 µA (Histed et al. 2009). Section density 1300 cells/mm² = ~1.2×10⁵ neurons/mm³ in mouse L2/3 × ~11 µm soma. Swept ×1/3 and ×3 |
| `site_density_i` | 6.0 /mm | calibrated | set so the recruited I/E fraction within 300 µm of the tip is ~1.1, the ratio Dadarlat, Sun & Stryker (2024) measured up to 50 µA in awake mouse V1. Equal densities would give 1.7, because interneuron axons are compact and put more of their sites near the tip. Swept from 0 to ×3 |
| `U_e`, `tau_rec_e` | 0.45, 300 ms | assumed | Tsodyks–Markram depression at pyramidal synapses. U ≈ 0.4–0.5 (Tsodyks & Markram 1997). Fast recovery component ~380 ms in L2/3 of rat V1 (Varela et al. 1997). Swept: τ_rec ×½ and ×2 |
| `U_i`, `tau_rec_i` | 0.35, 150 ms | assumed | representative depressing inhibitory synapse; enters only through ρ_i. Swept with the excitatory values |
| `ref_rate_e`, `ref_rate_i` | 20, 30 Hz | assumed | firing rate at which the network's effective weights are taken to apply (the visually driven regime of the source model) |

`effective_rate(f) = f · (ref + r_d) / (f + r_d)` with `r_d = 1000 / (U τ_rec)`
equals `ref` at `f = ref` and saturates at `ref + r_d`. At 300 Hz this gives
ρ_E ≈ 27 Hz and ρ_I ≈ 46 Hz. Driving harder than a few tens of hertz buys
almost no extra synaptic output, which is why ρ, not the pulse rate, matters.

### The forward model (`BiophysicalCortex`)

| name | value | status | note |
|---|---|---|---|
| `step_um` | 107 µm | published | the source grid |
| `margin_um` | 3200 µm | numerical | sheet extends this far beyond the outermost electrode; > `L_o` + 3 `sigma_ie` |
| `pulse_hz` | 300 Hz | protocol | the manuscript's stimulation protocol (300 Hz biphasic) |
| `background` | 10 | calibrated | uniform input giving spontaneous rates of 1.4 Hz (E) and 5.0 Hz (I), against ~1.5 and ~3 Hz in Obeid & Miller's spiking network. A supralinear network at zero rate has zero gain and cannot show either facilitation or the suppression of ongoing firing. Swept: 5 and 14 |
| `seed` | 0 | numerical | orientation map; seeds 1 and 2 swept |

## Checks

`prosthesis/v2_validate.py` writes `results/v2_validation.json`. The test
suite (`tests/test_biophysical.py`, 27 tests) asserts each property below
quickly on small sheets.

| check | data | model | fitted? |
|---|---|---|---|
| surround suppression index, E | 0.79 (Obeid & Miller, C = 16.4) | 0.73 | no: reimplementation test |
| surround suppression index, I | 0.27 | 0.29 | no |
| summation field size, E / I | 1.14° / 1.75° | 1.32° / 1.95° | no |
| summation field grows at low contrast | yes | 1.32° → 3.33° at C = 8 | no |
| inhibition-stabilised at spontaneous rates (paradoxical I response) | yes in cortex (Sanzeni et al. 2020) | yes | no |
| supralinear when weak, sublinear when strong | SSN signature | log-log slope 2.3 → < 1 | no |
| responding cells at 8.5 µA | ~5 (Histed et al. 2009) | 5.5 | **yes** |
| peak recruited fraction at 8.5 µA | sparse | 0.007 | no |
| median distance of recruited somata | 294 → 369 → 455 µm at 5, 20, 50 µA (Kumaravelu, 3-D) | 349 → 357 → 372 µm (lateral) | no; growth too shallow, see below |
| I/E recruited ratio near the tip | ~1.1 (Dadarlat et al. 2024) | 1.13–1.14 within 300 µm, 10–50 µA | **yes** |
| inhibitory cells recruited closer to the tip than excitatory | yes (Dadarlat et al.) | yes, median 230–242 µm against 349–372 µm | no |
| ongoing E firing suppressed around the electrode | yes (Butovas & Schwarz 2003); E cells beyond ~500 µm mainly inhibited (Dadarlat et al.) | yes; deepest suppression −12%, −28%, −79% at 50, 100, 300 µA | no |
| visual + electrical drive sums sub-additively | yes (Nassi et al. 2015) | both/sum 0.79–0.96 | no |
| paired ICMS in mouse V1 (Kim et al. 2025) | minimal at 7 µA, net enhancement at 13 µA, net zero at 16 µA; attenuation in strong single-electrode responders, enhancement in non-responders, peak between electrodes | **partly wrong.** Modulation sits between the electrodes and attenuation falls on the directly driven cells, as in the data. But nothing is enhanced. At 485 µm the net is −0.09%, −0.23%, −0.32% of the summed response at 6.8, 12.7, 15.9 µA. About 70% of it is occupancy, and the network-driven cells are slightly suppressed too. Every separation from 370 to 850 µm gives the same sign | no |

### Where it falls short

- **The spread of recruited somata grows too slowly with current.** In the
  model it is set almost entirely by the axonal footprint, and the initiation
  ball only blurs it slightly. Part of the gap is geometric: Kumaravelu's
  distances are 3-D, from a column with several layers, and these are
  lateral distances in one sheet.
- **It misses the enhancement Kim et al. saw at mouse currents.** The model
  places the attenuation correctly, on cells both electrodes reach directly,
  and puts the modulation between the electrodes. But at 7–16 µA it
  enhances nothing. The net is small and suppressive at every separation,
  and the network-driven cells are slightly suppressed as well, where Kim et
  al. found enhancement in cells that responded to neither electrode alone.
  The authors' spiking model ties part of that enhancement to cell-to-cell
  variability, which a population rate model cannot have, so this may be a
  limit of the model class. An earlier run on a loose convergence tolerance
  showed network-driven enhancement and a sign change near 16 µA. Both were
  numerical residue (item 2 below) and disappear once the fields converge.
- **The model has population rates, not cells.** It cannot produce fractions
  of enhanced and attenuated neurons, or anything that depends on
  cell-to-cell variability.

## Numerical problems found and fixed

Each of these produced wrong numbers without raising an error. Each now has
a test.

1. **Weight scaling with resolution was inverted** (`SSNSheet.__init__`).
   It is exact on the 107 µm grid, where the factor is 1, and wrong by 16× at
   53.5 µm. No experiment had used another grid, so no reported number was
   affected. Test: `test_recurrent_input_does_not_depend_on_grid_resolution`.
2. **The convergence tolerance was too loose for summed quantities.** The
   original early stop (`tol = 1e-4`) leaves ~10⁻³ Hz of residual per site.
   That is negligible at any one site. But interaction is measured as a sum
   over the sheet, and summed over ~10⁴ sites it came to a spurious +1.1%
   interaction between electrodes 4.5 mm apart at 10 µA. It also produced an
   apparent sign change near 16 µA that was pure noise. Evoked fields now
   converge to 10⁻⁷ and the spontaneous state to 10⁻⁹. The same pair reads
   0.02%. Test: `test_distant_electrodes_do_not_interact_numerically`.
3. **A lone electrode sat between four grid nodes** on even-sized grids, so
   two readouts of "the electrode's site" could pick neighbouring nodes. The
   grid is now centred on a node, and every readout goes through
   `site_index`. Test: `test_time_course_settles_to_the_steady_state`.

## What the model leaves out

- **Depth.** One sheet standing for layer 2/3, with the electrode tip in it.
  Utah-array tips sit at about 1–1.5 mm, nearer layer 4 or 5 in human V1, and
  the laminar origin of phosphenes is not known.
- **Spikes and pulses.** Rates only. Pulse waveform, charge balance and
  pulse-by-pulse timing enter only through ρ and K_ax.
- **Slow adaptation of directly driven cells.** Under sustained high-rate
  ICMS, directly activated neurons drop out over seconds (Michelson, Eles,
  Kozai and colleagues). Synaptic depression enters at its steady-state
  level, and every result except the sequence experiment (50 ms trains,
  integrated in time) is the network's steady state. The model therefore
  speaks for trains long enough for both to settle and too short for this
  drop-out to begin. (An earlier version said every train here is
  ≤ 300 ms, which described no duration the steady-state results have.)
- **Human-specific parameters.** The network is a cat/macaque-scale sheet.
  A human-scaled variant (all lateral lengths ×1.63, the human/macaque
  ocular-dominance column ratio of Adams, Sincich & Horton 2007) is in the
  sensitivity sweep as a hypothesis, not a measurement.
- **Deafferented cortex.** All parameters come from sighted animals.
- **A calibrated phosphene readout.** Phosphene size is read from the evoked
  rate through the retinotopy and is not fitted. Its half-maximum diameter
  matches Dynaphos at 100 µA (0.124° against 0.126°) and falls below it at
  higher currents (0.20° against 0.30° at 600 µA). The region where firing
  at least doubles is 1.1–1.5 times the Dynaphos size. How a subject's
  reports map onto cortical rate is not known.

## Reproducing

```bash
python3 prosthesis/v2_validate.py      # checks against data -> results/v2_validation.json
python3 prosthesis/v2_experiments.py   # pairs, arrays, charge, sequences, sensitivity
python3 prosthesis/v2_paper.py         # phosphene size, human data, repulsion, percepts,
                                       # reach, encoder, power (parts can run alone)
python3 -m pytest tests/test_biophysical.py tests/test_v2_paper.py -q
python3 paper/v2_tables.py             # every table in the manuscript, from the JSON
python3 paper/make_figures.py          # every figure
python3 paper/verify_claims.py         # every number the manuscript quotes, against the JSON
```

`python3 prosthesis/v2_experiments.py charge range` reruns only the
charge-to-hold and interaction-range parts and merges them into the existing
result file. Everything else needs a full run.
`python3 prosthesis/v2_paper.py encoder` (or `size`, `bosking`, `repulsion`,
`percepts`, `reach`, `power`) runs one part. The encoder part is the slow
one, a few hours, because every full-model evaluation runs the network.
`reach` fits each target with the best unconstrained linear combination of
single-electrode percepts, which separates targets the electrodes can draw
from thin shapes they cannot. Repulsion is measured two ways, on activity
centroids and on the two maxima along the electrode axis, and the two can
disagree in sign. What the
model predicts, and how a pooled divisor would differ, is in sections 3 and
3.9 and table 5 of `paper/manuscript_v2.tex`.

## References

- Adams DL, Sincich LC, Horton JC (2007). Complete pattern of ocular dominance columns in human primary visual cortex. *J Neurosci* 27:10391–10403.
- Ahmadian Y, Rubin DB, Miller KD (2013). Analysis of the stabilized supralinear network. *Neural Comput* 25:1994–2037.
- Butovas S, Schwarz C (2003). Spatiotemporal effects of microstimulation in rat neocortex: a parametric study using multielectrode recordings. *J Neurophysiol* 90:3024–3039.
- Butovas S, Hormuzdi SG, Monyer H, Schwarz C (2006). Effects of electrically coupled inhibitory networks on local neuronal responses to intracortical microstimulation. *J Neurophysiol* 96:1227–1236.
- Dadarlat MC, Sun YJ, Stryker MP (2024). Activity-dependent recruitment of inhibition and excitation in the awake mammalian cortex during electrical stimulation. *Neuron* 112:821–834.
- Histed MH, Bonin V, Reid RC (2009). Direct activation of sparse, distributed populations of cortical neurons by electrical microstimulation. *Neuron* 63:508–522.
- Kim K, Sonego V, Strohmer B, et al. (2025). Single-cell and population-level neuromodulation dynamics in dual-electrode intracortical stimulation. *bioRxiv* 10.1101/2025.06.25.661479.
- Kumaravelu K, Sombeck J, Miller LE, Bensmaia SJ, Grill WM (2022). Stoney vs. Histed: quantifying the spatial effects of intracortical microstimulation. *Brain Stimul* 15:141–151.
- Nassi JJ, Avery MC, Cetin AH, Roe AW, Reynolds JH (2015). Optogenetic activation of normalization in alert macaque visual cortex. *Neuron* 86:1504–1517.
- Obeid D, Miller KD (2025). Stabilized supralinear network model of responses to surround stimuli in primary visual cortex. *eNeuro* 12(5).
- Rubin DB, Van Hooser SD, Miller KD (2015). The stabilized supralinear network: a unifying circuit motif underlying multi-input integration in sensory cortex. *Neuron* 85:402–417.
- Sanzeni A, Akitake B, Goldbach HC, et al. (2020). Inhibition stabilization is a widespread property of cortical networks. *eLife* 9:e54875.
- Tsodyks MV, Markram H (1997). The neural code between neocortical pyramidal neurons depends on neurotransmitter release probability. *PNAS* 94:719–723.
- Varela JA, Sen K, Gibson J, Fost J, Abbott LF, Nelson SB (1997). A quantitative description of short-term plasticity at excitatory synapses in layer 2/3 of rat primary visual cortex. *J Neurosci* 17:7926–7940.
