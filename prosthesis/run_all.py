"""Regenerate every prosthesis result and figure from scratch.

Worth doing periodically rather than trusting accumulated JSON: several
mechanisms (amplitude-dependent activation magnitude, the far normalisation
pool) were added *after* some results were first produced, so stored numbers
can silently fall out of date with the code that is supposed to generate them.

Order matters only in that the figures come last.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

JOBS = [
    ("phase0_baselines.py", "baselines: retinal pixel-vs-axon, Dynaphos linearity"),
    ("phase1_circuit.py", "circuit model: calibration + separation sweep"),
    ("phase1_sensitivity.py", "sensitivity to normalisation parameters"),
    ("phase2_percepts.py", "percept decoder + multi-electrode percepts"),
    ("phase4_encoder.py", "optimal stimulation under each forward model"),
    ("phase5_scale.py", "interaction range vs stimulation current"),
    ("phase6_controls.py", "pointwise-saturation control + repulsion test"),
    ("phase7_regimes.py", "drive regimes, sign flip, and pool dynamics"),
    ("phase8_channels.py", "channel-count ceiling and charge budget"),
    ("phase9_sequence.py", "sequence-position bias + experiment power"),
    ("v2_validate.py", "biophysical model: checks against measurements"),
    ("v2_experiments.py", "biophysical model: the paper's experiments"),
    ("v2_paper.py", "biophysical model: size, human data, repulsion, "
                    "percepts, reach, encoder, power"),
]

# phase7 reads phase1's calibration, phase9 reads phase1's separation sweep,
# and phase6 recalibrates from the Dynaphos baseline, so the order above is a
# dependency order.  v2_validate and v2_experiments depend on nothing above
# them.  v2_paper reads phase1's Dynaphos sizes and separation sweep.  The v2
# scripts run last because they are the slowest.  The figures and tables
# (paper/make_figures.py, paper/v2_tables.py) read all of these and run after.


def main() -> int:
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    failures, t_all = [], time.time()

    for script, desc in JOBS:
        print(f"\n=== {desc}\n    {script}", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, str(HERE / script)], cwd=ROOT)
        dt = time.time() - t0
        if r.returncode != 0:
            failures.append(script)
            print(f"    FAILED after {dt:.0f}s", flush=True)
        else:
            print(f"    done in {dt:.0f}s", flush=True)

    print(f"\n=== total {time.time() - t_all:.0f}s")
    if failures:
        print("failed:", ", ".join(failures))
        return 1
    print("all prosthesis results regenerated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
