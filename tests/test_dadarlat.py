import numpy as np
import pytest

from prosthesis import dadarlat as D


def _synthetic(ratio, n=20000, seed=0):
    """Cells within 300 um whose expected sites are h_e and ratio * h_e."""
    rng = np.random.default_rng(seed)
    red = rng.random(n) < 0.3
    d = rng.uniform(0, 300, n)
    h = np.where(red, ratio, 1.0) * 0.4
    resp = rng.random(n) < 1 - np.exp(-h)
    floor = rng.random(n) < 0.0
    lab = {a: resp for a in D.AMPS}
    lab[3] = floor
    return [{"m": "x", "d": d, "red": red, "sig": lab, "early": lab}]


def test_site_ratio_undoes_saturation():
    # fractions 0.33 and 0.55 have a ratio of 1.67, the sites behind them 2.0
    assert D.site_ratio(_synthetic(2.0), 20, "sig") == pytest.approx(2.0, rel=0.05)


def test_bootstrap_interval_contains_the_truth():
    lo, hi = D.bootstrap(_synthetic(1.5, n=6000), lambda c: D.site_ratio(c, 20, "sig"), n=200)
    assert lo < 1.5 < hi


@pytest.mark.skipif(not D.available(), reason="run prosthesis/dadarlat.py fetch/extract first")
def test_reproduces_the_published_fractions():
    cs = D.cells()
    assert sum(int((~c["red"]).sum()) for c in cs) == 6162      # as published
    assert sum(int(c["red"].sum()) for c in cs) == 532
    f = D.published_fractions(cs)
    # Dadarlat, Sun & Stryker 2024: 49% +- 2.9 and 54% +- 4 at 40 uA,
    # 11.1% +- 1.3 of inhibitory cells at 10 uA, 2.7% +- 0.2 of excitatory at 5 uA
    assert round(100 * f[40]["e"], 1) == 49.0 and round(100 * f[40]["e_sem"], 1) == 2.9
    assert round(100 * f[40]["i"], 1) == 54.0 and round(100 * f[40]["i_sem"]) == 4
    assert round(100 * f[10]["i"], 1) == 11.1 and round(100 * f[10]["i_sem"], 1) == 1.3
    assert round(100 * f[5]["e"], 1) == 2.7


def test_field_medians_read_model_and_data_over_the_same_cells():
    from prosthesis.v2_awake import field_medians
    rng = np.random.default_rng(1)
    d = rng.uniform(0, 400, 4000)
    red = np.arange(4000) % 2 == 1
    near = d < 100                                   # only cells within 100 um respond
    lab = {a: near for a in (20, 50)}
    cs = [{"d": d, "red": red, "sig": lab}]
    flat = {str(a): [1e-3, 1e-3] for a in (20, 50)}  # the model drives every distance alike
    H = {"r": [0.0, 1000.0], "e": flat, "i": flat}
    fits = {"sig_local": {"s_e": 1.0, "s_i": 1.0, "s_loc_e": 0.0, "s_loc_i": 0.0, "L_um": 90.0}}
    m = field_medians(cs, H, fits)
    assert m["data_e50"] == pytest.approx(50, abs=5)
    assert m["calibration_i20"] == pytest.approx(np.median(d[red]), abs=2)
