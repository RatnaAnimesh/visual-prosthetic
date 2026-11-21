"""Run every experiment and render the figures.

Each experiment writes JSON to results/ and is independent, so a failure in
one does not lose the others.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

JOBS = [
    # (script, extra args, description)
    ("orientation_tuning.py", ["--sf", "2.6",
                               "--out", "results/orientation_tuning.json"],
     "orientation tuning, parvocellular-optimal spatial frequency"),
    ("orientation_tuning.py", ["--sf", "1.4",
                               "--out", "results/orientation_tuning_magno.json"],
     "orientation tuning, magnocellular-optimal spatial frequency"),
    ("spatial_frequency.py", [], "magno / parvo stream separation"),
    ("circuit_dissection.py", [], "lesioning each circuit element"),
    ("simple_vs_complex.py", [], "simple and complex receptive fields"),
    ("ocular_dominance.py", [], "ocular dominance across the layers"),
    ("contrast_response.py", [], "contrast response and invariance"),
    ("moving_bar.py", [], "the Hubel-Wiesel moving bar"),
    # Decoding reads the orientation-tuning output, so it must come last.
    ("population_decoding.py", [], "population vector decoding"),
]


def main() -> int:
    (ROOT / "results").mkdir(exist_ok=True)
    failures = []
    for script, args, desc in JOBS:
        print(f"\n=== {desc}\n    {script} {' '.join(args)}", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, str(HERE / script), *args],
                           cwd=ROOT)
        dt = time.time() - t0
        if r.returncode != 0:
            failures.append(script)
            print(f"    FAILED after {dt:.0f}s", flush=True)
        else:
            print(f"    done in {dt:.0f}s", flush=True)

    print("\n=== figures", flush=True)
    subprocess.run([sys.executable, str(HERE / "make_figures.py")], cwd=ROOT)

    if failures:
        print("\nfailed:", ", ".join(failures))
        return 1
    print("\nall experiments completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
