import math

import numpy as np

from visiontune.evaluation.metrics import (bootstrap_ci, group_summary, mcnemar_exact,
                                           paired_bootstrap, summarize, wilson_ci)


def test_wilson_matches_known_value():
    lo, hi = wilson_ci(50, 100)
    assert math.isclose(lo, 0.4038, abs_tol=1e-3) and math.isclose(hi, 0.5962, abs_tol=1e-3)


def test_bootstrap_ci_contains_mean():
    x = [True] * 70 + [False] * 30
    lo, hi = bootstrap_ci(x, n_boot=2000)
    assert lo < 0.7 < hi


def test_paired_bootstrap_detects_clear_improvement():
    rng = np.random.default_rng(1)
    a = rng.random(500) < 0.6
    b = a | (rng.random(500) < 0.3)
    res = paired_bootstrap(a, b, n_boot=2000)
    assert res["diff"] > 0 and res["ci_low"] > 0 and res["p_value"] < 0.01


def test_paired_bootstrap_identical_models():
    a = [True, False] * 50
    res = paired_bootstrap(a, a, n_boot=500)
    assert res["diff"] == 0 and res["p_value"] == 1.0


def test_mcnemar_exact():
    a = [True] * 10 + [False] * 0
    b = [False] * 10
    res = mcnemar_exact(a, b)
    assert res["a_only"] == 10 and res["b_only"] == 0
    assert math.isclose(res["p_value"], 2 / 2**10)


def test_group_summary_counts():
    recs = [{"subject": "natural science"}] * 3 + [{"subject": "language science"}]
    out = group_summary(recs, [True, True, False, True], "subject")
    assert out["natural science"]["n"] == 3 and math.isclose(out["natural science"]["accuracy"], 2 / 3)
    assert list(out) == ["natural science", "language science"]
    assert summarize([])["n"] == 0
    flags = group_summary([{"f": True}, {"f": False}, {}], [True, False, True], "f")
    assert set(flags) == {"True", "False", "unknown"}
