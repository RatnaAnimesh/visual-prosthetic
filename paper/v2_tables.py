"""LaTeX tables for the manuscript, written from the result files.

The manuscript \\input{}s these, so no number in them is typed by hand.

    paper/tables/structure.tex        departure from superposition by mechanism
    paper/tables/human.tex            interaction at the separations of the
                                      human multi-electrode data
    paper/tables/v2_validation.tex    the model against measurements
    paper/tables/v2_sensitivity.tex   headline numbers across the parameters
                                      the literature does not pin down
    paper/tables/comparison.tex       measured parts against a pooled divisor
    paper/tables/inhibition.tex       the inhibitory density against Kim et
                                      al. and Dadarlat et al. together

Run after prosthesis/run_all.py:
    python3 paper/v2_tables.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def _minus(s: str) -> str:
    """Typeset a leading sign as a real minus (or plus) in LaTeX."""
    return s.replace("-", "$-$", 1) if s.startswith("-") else \
        s.replace("+", "$+$", 1) if s.startswith("+") else s


def load(name):
    return json.loads((RESULTS / name).read_text())


def _nearest(rows, sep, key="separation_um"):
    return min(rows, key=lambda r: abs(r[key] - sep))


def _span(vals):
    """'-10.2 to +2.3%' for a list of fractions."""
    lo, hi = (_minus(f"{100 * v:+.1f}") for v in (min(vals), max(vals)))
    return f"{lo} to {hi}\\%"


def _slope(pairs):
    """Which way a crossover current moves with separation, over where it exists."""
    if not pairs:
        return "none"
    (s0, c0), (s1, c1) = min(pairs), max(pairs)
    return f"{'rises' if c1 > c0 else 'falls'}, {s0 / 1000:g} to {s1 / 1000:g}~mm"


def _dk(dk, cap_ma):
    return f"$+${dk:.2f}" if dk is not None else f"none to {cap_ma:g}~mA"


def _pct(v, digits=1):
    """Percent, with values that are zero to machine precision shown as 0."""
    return "0" if abs(v) < 1e-6 else f"{100 * v:.{digits}f}"


# ---------------------------------------------------------------------------

STRUCT_SEPS = (400.0, 800.0, 1200.0, 1600.0, 2400.0, 3200.0, 4500.0)


def structure() -> str:
    sw = load("phase1_circuit.json")["separation_sweep"]
    pairs = load("v2_experiments.json")["pairs"]

    def phen(k):
        return [_pct(_nearest(sw[k], s)["max_dev"]) for s in STRUCT_SEPS]

    def bio(key):
        return [_pct(next(r[key] for r in pairs if r["amp_ua"] == 300.0
                          and r["separation_um"] == s)) for s in STRUCT_SEPS]

    rows = [
        ("Dynaphos baseline", ["0"] * len(STRUCT_SEPS)),
        ("current spread", phen("circuit off")),
        ("\\quad + horizontal", phen("horizontal only")),
        ("\\midrule measured parts", bio("max_dev")),
        ("\\quad occupancy share", bio("max_dev_direct")),
        ("\\midrule pointwise saturation", phen("pointwise saturation")),
        ("pooled, norm.\\ only", phen("normalisation only")),
        ("pooled divisor", phen("full circuit")),
    ]
    head = " & ".join(f"{s / 1000:g}" for s in STRUCT_SEPS)
    body = []
    for name, vals in rows:
        rule = ""
        if name.startswith("\\midrule "):
            rule, name = "\\midrule\n", name[len("\\midrule "):]
        body.append(f"{rule}{name} & " + " & ".join(vals) + " \\\\")
    return ("\\begin{tabular}{l" + "c" * len(STRUCT_SEPS) + "}\n\\toprule\n"
            f"separation (mm) & {head} \\\\\n\\midrule\n"
            + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n")


HUMAN_SEPS = (4110.0, 9570.0, 13470.0, 26360.0)


def human() -> str:
    b = load("v2_paper_bosking.json")
    rows = b["rows"]
    rng = b["range_to_tenth_um"]
    p5 = load("phase5_scale.json")
    sx = np.array(p5["separations_um"], float)

    def cur(a):
        return f"{a / 1000:g}~mA"

    lines = []
    for a in sorted({r["amp_ua"] for r in rows}):
        vals = [_pct(next(r["max_dev"] for r in rows if r["amp_ua"] == a
                          and r["separation_um"] == s)) for s in HUMAN_SEPS]
        d = rng.get(str(a))
        lines.append(f"{cur(a)} & " + " & ".join(vals) + " & "
                     + ("--" if d is None else f"{d / 1000:.1f}~mm")
                     + " \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{6}{l}{\\textit{pooled divisor}} \\\\")
    for a, v in sorted(p5["by_amplitude"].items(), key=lambda kv:
                       float(kv[0])):
        dv = np.array(v["deviations"])
        vals = [_pct(float(np.interp(s, sx, dv))) for s in HUMAN_SEPS]
        lines.append(f"{cur(float(a))} & " + " & ".join(vals)
                     + f" & {v['decay10_um'] / 1000:.1f}~mm \\\\")
    return ("\\begin{tabular}{lccccc}\n\\toprule\n"
            "current & 4.11~mm & 9.57~mm & 13.47~mm & 26.36~mm & tenth \\\\\n"
            " & merged & one & one lost & two & by \\\\\n"
            "\\midrule\n"
            "\\multicolumn{6}{l}{\\textit{measured parts}} \\\\\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def validation() -> str:
    v = load("v2_validation.json")
    nb = v["network"]["16.4"]
    rec = {r["amp_ua"]: r for r in v["recruitment"]["rows"]}
    inh = {r["amp_ua"]: r for r in v["inhibition"]["rows"]}
    vs = [r["ratio_to_sum"] for r in v["visual_plus_stim"]]
    ong = {r["amp_ua"]: r for r in v["ongoing"]}
    du = {(r["separation_um"], r["amp_ua"]): r for r in v["dual"]}
    size = {r["amp_ua"]: r for r in load("v2_paper_size.json")["rows"]}
    fm = load("v2_awake_cells.json")["field_medians"]

    def pct(x):
        return _minus(f"{100 * x:+.2f}\\%")

    rows = [
        ("Surround suppression index, E / I", "0.79 / 0.27",
         f"{nb['si_e']:.2f} / {nb['si_i']:.2f}", "no",
         "\\citet{obeid2025}"),
        ("Summation field, E / I", "1.14\\textdegree{} / 1.75\\textdegree",
         f"{nb['sf_e_deg']:.2f}\\textdegree{{}} / "
         f"{nb['sf_i_deg']:.2f}\\textdegree", "no", "\\citet{obeid2025}"),
        ("Cells responding at 8.5~\\uA", "about 5",
         f"{rec[8.5]['cells_in_plane']:.1f}", "yes", "\\citet{histed2009}"),
        ("Peak recruited fraction at 8.5~\\uA", "sparse",
         f"{rec[8.5]['peak_fraction_e']:.3f}", "no", "\\citet{histed2009}"),
        ("Median soma distance, 5/20/50~\\uA{} (\\um)", "294 / 369 / 455",
         f"{rec[5.0]['median_distance_um']:.0f} / "
         f"{rec[20.0]['median_distance_um']:.0f} / "
         f"{rec[50.0]['median_distance_um']:.0f}", "no",
         "\\citet{kumaravelu2022}"),
        ("Recruited I/E ratio within 300~\\um", "about 1.1",
         f"{inh[50.0]['ratio_within_300um']:.2f}", "yes",
         "\\citet{dadarlat2024}"),
        ("Imaged median, I vs E at 50~\\uA{} (\\um)",
         f"{fm['data_i50']:.0f} vs {fm['data_e50']:.0f}",
         f"{fm['calibration_i50']:.0f} vs {fm['calibration_e50']:.0f}", "no",
         "\\citet{dadarlat2023data}"),
        ("Ongoing firing near the tip at 100~\\uA", "suppressed",
         _minus(f"down by {-100 * ong[100.0]['deepest_suppression_frac']:.0f}\\%"),
         "no", "\\citet{butovas2003}"),
        ("Visual + electrical, both/sum", "below 1",
         f"{min(vs):.2f}--{max(vs):.2f}", "no", "\\citet{nassi2015}"),
        ("Paired net modulation, 6.8 / 12.7 / 15.9~\\uA", "0 / $+$ / 0",
         " / ".join(pct(du[(485.0, a)]["net_fraction"])
                    for a in (6.8, 12.7, 15.9)), "no", "\\citet{kim2025}"),
        ("Phosphene diameter, 100 / 600~\\uA",
         f"{size[100.0]['dynaphos_diameter_deg']:.3f}\\textdegree{{}} / "
         f"{size[600.0]['dynaphos_diameter_deg']:.3f}\\textdegree$^\\ast$",
         f"{size[100.0]['halfmax_diameter_deg']:.3f}\\textdegree{{}} / "
         f"{size[600.0]['halfmax_diameter_deg']:.3f}\\textdegree", "no",
         "\\citet{vandergrinten2023}"),
    ]
    body = "\n".join(f"{a} & {b} & {c} & {d} & {e} \\\\" for a, b, c, d, e
                     in rows)
    return ("\\begin{tabular}{lllcl}\n\\toprule\n"
            "quantity & measured & model & fitted & source \\\\\n\\midrule\n"
            + body + "\n\\bottomrule\n\\end{tabular}\n")


def sensitivity() -> str:
    s = load("v2_experiments.json")["sensitivity"]
    cols = [("signed_100ua_0.4mm", "s"), ("signed_300ua_0.4mm", "s"),
            ("signed_300ua_1.6mm", "s"), ("peak_300ua_0.4mm", "p"),
            ("efficiency_16ch_0.4mm_100ua", "e"),
            ("efficiency_16ch_0.4mm_300ua", "e")]

    def fmt(v, kind):
        if kind == "s":
            return _minus(f"{100 * v:+.1f}")
        if kind == "p":
            return f"{100 * v:.1f}"
        return f"{v:.2f}"

    lines = []
    for name, h in s.items():
        tex = name.replace("x0.5", "$\\times$0.5").replace("x1.5", "$\\times$1.5") \
            .replace("x0 ", "$\\times$0 ").replace("x3", "$\\times$3") \
            .replace("x1/3", "$\\times$1/3").replace("x2", "$\\times$2") \
            .replace("x1.63", "$\\times$1.63") \
            .replace("K_ax", "$K$").replace("tau_rec", "$\\tau_\\mathrm{rec}$")
        shape = "" if h["monotonic_decay_300ua"] else "$^\\dagger$"
        lines.append(tex + shape + " & " + " & ".join(
            fmt(h[k], kind) for k, kind in cols) + " \\\\")
        if name == "baseline":
            lines.append("\\midrule")
    return ("\\begin{tabular}{lrrrrrr}\n\\toprule\n"
            " & \\multicolumn{3}{c}{signed interaction (\\%)} & peak (\\%) & "
            "\\multicolumn{2}{c}{16 electrodes} \\\\\n"
            " & 100~\\uA & 300~\\uA & 300~\\uA & 300~\\uA & 100~\\uA & "
            "300~\\uA \\\\\n"
            "variant & 0.4~mm & 0.4~mm & 1.6~mm & 0.4~mm & "
            "\\multicolumn{2}{c}{delivered / sum} \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def comparison() -> str:
    """What each model predicts, side by side."""
    sw = load("phase1_circuit.json")["separation_sweep"]
    e = load("v2_experiments.json")
    S = e["sensitivity"]
    P = {(r["amp_ua"], r["separation_um"]): r for r in e["pairs"]}
    surf = load("phase7_regimes.json")["surface"]["full circuit"]
    p5 = load("phase5_scale.json")["by_amplitude"]
    ph4 = load("phase4_encoder.json")
    ph8 = load("phase8_channels.json")
    ph9 = load("phase9_sequence.json")
    rp = load("v2_paper_repulsion.json")
    bk = load("v2_paper_bosking.json")
    pc = load("v2_paper_percepts.json")
    enc = load("v2_paper_encoder.json")["mean"]
    ch = {(r["pitch_um"], r["n"]): r for r in e["charge"]}
    C = {(r["amp_ua"], r["pitch_um"], r["n"]): r for r in e["channels"]}
    peaks = [100 * h["peak_300ua_0.4mm"] for h in S.values()]
    full = sorted(sw["full circuit"], key=lambda r: r["separation_um"])
    pk = max(full, key=lambda r: r["max_dev"])
    cross = [r["crossover_ua"] for r in surf if r["crossover_ua"]]
    dk = {r["n"]: r.get("delta_k") for r in ph8["shannon"]["by_pitch"]["800.0"]}
    dk4 = {r["n"]: r.get("delta_k") for r in ph8["shannon"]["by_pitch"]["400.0"]}
    seqs = [x["second_over_solo"] for res in e["sequence"].values()
            for r in res for x in r["rows"]]
    prep = [r["repulsion"] for r in rp["pooled_divisor"]]
    brep = [r["repulsion"] for r in rp["measured_parts"]]
    ppk = [r["peak_shift"] for r in rp["peaks_pooled_divisor"]
           if r["peak_shift"] is not None]
    bpk = [r["peak_shift"] for r in rp["peaks_measured_parts"]
           if r["peak_shift"] is not None]
    b411 = [r["max_dev"] for r in bk["rows"] if r["separation_um"] == 4110.0]
    tenth = bk["range_to_tenth_um"]
    dn = load("v2_awake_dense.json")["awake_all"]
    DP = {(r["amp_ua"], r["separation_um"]): r for r in dn["pairs"]}
    DC = {(r["amp_ua"], r["pitch_um"], r["n"]): r for r in dn["channels"]}
    dch = {(r["pitch_um"], r["n"]): r for r in dn["charge"]}
    dcross = [v for v in dn["crossover_ua"].values() if v]
    dseq = [x["second_over_solo"] for r in dn["sequence"] for x in r["rows"]]
    dpk = dn["signature"]

    nr = "not run"
    rows = [
        ("Linear mechanisms add nothing", "exact", "exact", "exact"),
        ("Peak departure, 300~\\uA, 0.4~mm",
         f"{_pct(full[0]['max_dev'])}\\%",
         f"{_pct(P[(300.0, 400.0)]['max_dev'])}\\% "
         f"({min(peaks):.1f}--{max(peaks):.1f}\\%)",
         f"{_pct(DP[(300.0, 400.0)]['max_dev'])}\\%"),
        ("Where that departure peaks",
         f"{pk['separation_um'] / 1000:.1f}~mm (interior)",
         "shortest separation, all 19 variants",
         "shortest separation" if dpk["peak_at_um"] == 400.0
         else f"{dpk['peak_at_um'] / 1000:.1f}~mm"),
        ("Sign reverses with current",
         f"at {min(cross):.0f}--{max(cross):.0f}~\\uA",
         "no, to 600~\\uA{} at $\\le$0.8~mm",
         (f"at {min(dcross):.0f}--{max(dcross):.0f}~\\uA" if dcross else "no")),
        ("Crossover current against separation",
         _slope([(r["separation_um"], r["crossover_ua"]) for r in surf
                 if r["crossover_ua"]]),
         _slope([(float(k), v) for k, v in e["crossover_ua"].items() if v]),
         _slope([(float(k), v) for k, v in dn["crossover_ua"].items() if v])),
        ("Decay to a tenth, 0.3--4~mA",
         f"{p5['300.0']['decay10_um'] / 1000:.1f}--"
         f"{p5['4000.0']['decay10_um'] / 1000:.1f}~mm",
         f"{tenth['300.0'] / 1000:.1f}--{tenth['4000.0'] / 1000:.1f}~mm", nr),
        ("Peak departure at 4.11~mm, up to 4~mA",
         f"up to {100 * float(np.interp(4110, np.array(load('phase5_scale.json')['separations_um'], float), np.array(p5['4000.0']['deviations']))):.0f}\\%",
         f"at most {100 * max(b411):.1f}\\%", nr),
        ("Centroids, change in distance, 0.1--0.6~mA",
         _span(prep), _span(brep), nr),
        ("Maxima, change in distance, 0.1--0.6~mA",
         _span(ppk), _span(bpk), nr),
        ("Five electrodes, percept change",
         f"{100 * pc['pooled_divisor']['percept_change']:.0f}\\%",
         f"{100 * pc['percept_change']:.0f}\\%", nr),
        ("Brightness, programmed as independent",
         f"{100 * ph4['mean_contrast_ratio']:.0f}\\% of target",
         f"{100 * enc['contrast_ratio_mismatched']:.0f}\\% of target", nr),
        ("Later phosphene, brightness against alone",
         f"dimmer by up to {ph9['max_predicted_dimming_pct']:.0f}\\%",
         f"within {100 * max(abs(1 - x) for x in seqs):.1f}\\%",
         # dense recruitment saturates the tip the readout sits on
         "unchanged, tip saturated" if max(abs(1 - x) for x in dseq) < 1e-3
         else f"within {100 * max(abs(1 - x) for x in dseq):.1f}\\%"),
        ("100 electrodes, 0.4~mm, 300~\\uA",
         f"{ph8['ceilings']['circuit']['400.0']['relative_total'][-1]:.1f}"
         "$\\times$ one",
         f"{C[(300.0, 400.0, 100)]['relative_to_one']:.0f}$\\times$ one",
         f"{DC[(300.0, 400.0, 100)]['relative_to_one']:.0f}$\\times$ one"),
        ("$\\Delta k$ to hold brightness, $N=16$, 0.4~mm",
         _dk(dk4[16], 200), _dk(ch[(400.0, 16)]["delta_k"], 20),
         _dk(dch[(400.0, 16)]["delta_k"], 20)),
        ("$\\Delta k$ to hold brightness, $N=64$, 0.4~mm",
         _dk(dk4[64], 200), _dk(ch[(400.0, 64)]["delta_k"], 20),
         _dk(dch[(400.0, 64)]["delta_k"], 20)),
        ("$\\Delta k$ to hold brightness, $N=64$, 0.8~mm",
         f"$+${dk[64]:.1f}",
         f"$+${ch[(800.0, 64)]['delta_k']:.2f}",
         f"$+${dch[(800.0, 64)]['delta_k']:.2f}"),
    ]
    body = "\n".join(" & ".join(r) + " \\\\" for r in rows)
    return ("\\begin{tabular}{llll}\n\\toprule\n"
            " & & \\multicolumn{2}{c}{measured parts} \\\\\n"
            "\\cmidrule(lr){3-4}\n"
            "prediction & pooled divisor & calibrated & dense \\\\\n\\midrule\n"
            + body + "\n\\bottomrule\n\\end{tabular}\n")


def inhibition() -> str:
    d = load("v2_paper_inhibition.json")

    def sgn(v, digits):
        return _minus(f"{100 * v:+.{digits}f}")

    lines = []
    for r in d["rows"]:
        kim = {x["amp_ua"]: x for x in r["kim"] if x["separation_um"] == 485.0}
        k = f"{r['mult']:g}" + (" (calibrated)" if r["mult"] == 1.0 else "")
        lines.append(" & ".join(
            [k, f"{r['ie_ratio_300um_20ua']:.2f}"]
            + [sgn(kim[a]["net_fraction"], 2) for a in (6.8, 12.7, 15.9)]
            + [sgn(kim[12.7]["net_fraction_e1_view"], 2),
               sgn(r["signed_100ua_0.4mm"], 1),
               sgn(r["signed_300ua_0.4mm"], 1)]) + " \\\\")
    return ("\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
            "inhibitory & I/E & \\multicolumn{4}{c}{paired net modulation, "
            "485~\\um{} (\\%)} & \\multicolumn{2}{c}{signed, 0.4~mm (\\%)} "
            "\\\\\n"
            "site density & ratio & 6.8~\\uA & 12.7~\\uA & 15.9~\\uA & "
            "12.7~\\uA$^\\ast$ & 100~\\uA & 300~\\uA \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


TABLES = {"structure.tex": structure, "human.tex": human,
          "v2_validation.tex": validation, "v2_sensitivity.tex": sensitivity,
          "comparison.tex": comparison, "inhibition.tex": inhibition}


def main() -> None:
    for name, fn in TABLES.items():
        (OUT / name).write_text(fn())
        print(f"wrote {OUT / name}")


if __name__ == "__main__":
    main()
