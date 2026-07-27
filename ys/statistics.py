"""Pure statistics helpers for `ys compare`/`ys report` -- feature 3 in
IMPROVEMENTS.md ("statistics worth the name").

Everything here is a pure function over plain lists of floats/ints: no
sqlite, no `Experiment`, no rich/HTML formatting. `ys/render.py` is what
turns these into the strings a user actually reads (arm labels, colors,
table cells); this module only computes numbers, which is what makes it
testable against hand-computed known-answer cases without a database or an
`Experiment` in the loop.

No new dependencies. `scipy` is not declared in pyproject.toml and this
module doesn't need it: a bootstrap is a resampling loop, a permutation
test at n<=a few dozen per arm can be enumerated exactly with
`itertools.combinations`, and a Wilson interval and the normal quantile
function used by the minimum-detectable-effect helper are both closed-form
`math`. Every source of randomness here (the bootstrap resampler, and the
permutation test's Monte Carlo fallback for n too large to enumerate
exactly) is driven by a `random.Random(seed)` instance constructed fresh
inside the call, never the module-global `random` -- so calling any
function here twice with the same arguments always returns the same
result. That determinism is load-bearing: a p-value or confidence interval
that changed between two runs of `ys compare` on unchanged data would
destroy trust in the number (see IMPROVEMENTS.md feature 3).

The single most important thing this module encodes is a negative result:
`repeats: 3` (the experiment schema's own default) is too small an n for a
two-sided exact permutation test to ever reach p<0.05, regardless of effect
size -- `min_two_sided_p`/`min_n_for_exact_significance` make that a
computable fact instead of a rule of thumb. Reporting *that* plainly is
more useful than printing a p-value that implies more precision than three
repeats can support.
"""
import itertools
import math
import random
import statistics
from dataclasses import dataclass
from typing import Optional

# Fixed, arbitrary seed shared by the bootstrap resampler and the
# permutation test's Monte Carlo fallback. Not derived from the data --
# determinism only requires that the *same* seed is used on every call, and
# a fixed constant is simpler to reason about (and to test against) than a
# data-derived one. A fresh `random.Random(DEFAULT_SEED)` is constructed
# per call rather than seeding the module-global `random`, so this module
# never has a side effect on any other code in the process that happens to
# use `random`.
DEFAULT_SEED = 20260101

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.8

# 95% two-sided Wilson interval by default, matching the confidence level
# `_norm_ppf`-driven z-scores elsewhere in this module default to (alpha=0.05).
_Z_95 = 1.959963984540054  # norm.ppf(0.975), pinned as a literal below too


# ---------------------------------------------------------------------------
# Normal quantile function (inverse CDF), for the minimum-detectable-effect
# helper's z_alpha/z_power lookup. Peter Acklam's rational approximation --
# public domain, widely used exactly for this "no scipy" situation. Accurate
# to about 1.15e-9 relative error, far more precision than three repeats of
# noisy agent-run data could ever need.
# ---------------------------------------------------------------------------

_ACKLAM_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_ACKLAM_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_ACKLAM_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_ACKLAM_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_ACKLAM_P_LOW = 0.02425


def _norm_ppf(p: float) -> float:
    """Standard normal quantile function (inverse CDF). Raises ValueError
    outside (0, 1) -- there is no finite quantile at or beyond the tails."""
    if not (0.0 < p < 1.0):
        raise ValueError(f"_norm_ppf domain is (0, 1), got {p}")
    a, b, c, d = _ACKLAM_A, _ACKLAM_B, _ACKLAM_C, _ACKLAM_D
    p_low = _ACKLAM_P_LOW
    p_high = 1 - p_low
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals -- one per metric.
# ---------------------------------------------------------------------------


@dataclass
class BootstrapResult:
    point: float  # the observed mean, not a resampled statistic
    low: float
    high: float
    n_resamples: int


def bootstrap_ci(
    values: list,
    *,
    alpha: float = DEFAULT_ALPHA,
    n_resamples: int = 2000,
    seed: int = DEFAULT_SEED,
) -> Optional[BootstrapResult]:
    """Percentile bootstrap confidence interval for the mean of `values`.

    Resamples `values` with replacement `n_resamples` times, computes the
    mean of each resample, and takes the `alpha/2`/`1 - alpha/2` percentiles
    of that distribution as the interval bounds. `None` if there are fewer
    than 2 observations -- a single point has no within-sample variability
    to resample, so any interval computed from it would just be a single
    repeated value, not a real estimate of uncertainty.

    Deterministic: `random.Random(seed)` is a fresh instance built for this
    call, so the same `values` (in the same order) and the same `seed`
    always produce the same interval.
    """
    n = len(values)
    if n < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo_idx = max(0, min(n_resamples - 1, round((alpha / 2) * (n_resamples - 1))))
    hi_idx = max(0, min(n_resamples - 1, round((1 - alpha / 2) * (n_resamples - 1))))
    return BootstrapResult(
        point=sum(values) / n,
        low=means[lo_idx],
        high=means[hi_idx],
        n_resamples=n_resamples,
    )


# ---------------------------------------------------------------------------
# Wilson score interval -- for success rate (a proportion, not a mean).
# ---------------------------------------------------------------------------


@dataclass
class WilsonInterval:
    point: float  # successes / n
    low: float
    high: float


def wilson_interval(successes: int, n: int, *, alpha: float = DEFAULT_ALPHA) -> Optional[WilsonInterval]:
    """Wilson score interval for a binomial proportion (`successes` out of
    `n`), the standard fix for the normal approximation's well-known bad
    behaviour near 0 and 1 (e.g. a 100% or 0% observed rate, unremarkable at
    n=3, would give the normal approximation a zero-width interval). `None`
    for n=0 -- there's no rate to bound with zero observations.
    """
    if n <= 0:
        return None
    z = _norm_ppf(1 - alpha / 2)
    phat = successes / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    low = (center - margin) / denom
    high = (center + margin) / denom
    return WilsonInterval(point=phat, low=max(0.0, low), high=min(1.0, high))


# ---------------------------------------------------------------------------
# Permutation test -- on the primary metric, arm vs. baseline.
# ---------------------------------------------------------------------------


@dataclass
class PermutationResult:
    observed_diff: float  # mean(b) - mean(a)
    p_value: float  # two-sided
    exact: bool  # True if every relabeling was enumerated, not sampled
    n_permutations: int
    min_possible_p: Optional[float]  # only set when exact -- see min_two_sided_p


def min_two_sided_p(n_a: int, n_b: int) -> Optional[float]:
    """The smallest two-sided p-value an exact permutation test on groups of
    size `n_a`/`n_b` could ever report, assuming all `n_a + n_b` values are
    distinct (no ties). With every value distinct, exactly 2 of the
    `C(n_a+n_b, n_a)` relabelings reproduce the same |mean(b) - mean(a)| as
    the extreme, fully-separated case -- the true split itself, and its
    mirror image (swap which group is "a" and which is "b", which leaves
    the absolute difference unchanged). No possible dataset at this sample
    size can beat that floor, however large the real effect is: this is
    exactly the fact behind IMPROVEMENTS.md feature 3's headline claim that
    n=3 vs n=3 cannot produce p<0.05 (2/C(6,3) = 2/20 = 0.1). `None` if
    either group is empty.
    """
    if n_a <= 0 or n_b <= 0:
        return None
    return 2 / math.comb(n_a + n_b, n_a)


def permutation_test(
    baseline: list,
    arm: list,
    *,
    seed: int = DEFAULT_SEED,
    max_exact_combinations: int = 200_000,
    n_monte_carlo: int = 20_000,
) -> Optional[PermutationResult]:
    """Two-sided exact (or, past `max_exact_combinations`, Monte Carlo)
    permutation test on the difference of means between `baseline` and
    `arm`. `None` if either list is empty.

    Exact whenever `C(n_a + n_b, n_a) <= max_exact_combinations` (always
    true at the tiny n this rig runs at by default -- `repeats: 3` gives
    C(6,3) = 20): every way of relabeling the pooled values into two groups
    of sizes n_a/n_b is enumerated via `itertools.combinations`, and the
    p-value is the exact fraction of relabelings at least as extreme as the
    observed split. This is the natural fit the plan calls for -- no
    normal-approximation assumptions, and, at these sample sizes, no
    randomness at all: two calls with the same data give bit-identical
    results, not just statistically-similar ones.

    Falls back to `n_monte_carlo` random relabelings (deterministic via
    `random.Random(seed)`) only when the full enumeration would be too
    large -- large enough n that no run of this rig's default `repeats`
    would ever reach it in practice, but the fallback exists so this
    function doesn't hang or blow up memory if it did.
    """
    n_a, n_b = len(baseline), len(arm)
    if n_a == 0 or n_b == 0:
        return None
    pooled = list(baseline) + list(arm)
    n = n_a + n_b
    observed = statistics.mean(arm) - statistics.mean(baseline)
    # Floating-point tolerance so the observed split itself is always
    # counted as "at least as extreme as itself" despite summation-order
    # rounding noise.
    threshold = abs(observed) - 1e-9

    total_combinations = math.comb(n, n_a)
    if total_combinations <= max_exact_combinations:
        as_extreme = 0
        for combo in itertools.combinations(range(n), n_a):
            combo_set = set(combo)
            group_a = [pooled[i] for i in combo]
            group_b = [pooled[i] for i in range(n) if i not in combo_set]
            diff = statistics.mean(group_b) - statistics.mean(group_a)
            if abs(diff) >= threshold:
                as_extreme += 1
        return PermutationResult(
            observed_diff=observed,
            p_value=as_extreme / total_combinations,
            exact=True,
            n_permutations=total_combinations,
            min_possible_p=min_two_sided_p(n_a, n_b),
        )

    rng = random.Random(seed)
    as_extreme = 0
    for _ in range(n_monte_carlo):
        shuffled = pooled[:]
        rng.shuffle(shuffled)
        group_a, group_b = shuffled[:n_a], shuffled[n_a:]
        diff = statistics.mean(group_b) - statistics.mean(group_a)
        if abs(diff) >= threshold:
            as_extreme += 1
    return PermutationResult(
        observed_diff=observed,
        p_value=as_extreme / n_monte_carlo,
        exact=False,
        n_permutations=n_monte_carlo,
        min_possible_p=None,
    )


# ---------------------------------------------------------------------------
# Minimum detectable effect / required-repeats helper.
# ---------------------------------------------------------------------------


def min_n_for_exact_significance(alpha: float = DEFAULT_ALPHA, max_n: int = 1000) -> Optional[int]:
    """The smallest equal per-arm n at which an exact two-sided permutation
    test (see `min_two_sided_p`) could *structurally* reach p<`alpha`, for
    any data whatsoever. At the default alpha=0.05 this is 4 (2/C(8,4) =
    2/70 ~= 0.0286 < 0.05; 2/C(6,3) = 0.1 is still too big at n=3). This is
    a hard floor independent of effect size or variance -- no amount of a
    real difference between arms can produce significance below it, which
    is exactly why the verdict line needs to say so rather than just
    printing a p-value. `None` if no n up to `max_n` clears the bar (only
    possible for an unreasonably small alpha).
    """
    n = 1
    while n <= max_n:
        p = min_two_sided_p(n, n)
        if p is not None and p < alpha:
            return n
        n += 1
    return None


def pooled_relative_stdev(a: list, b: list) -> Optional[float]:
    """Pooled within-group standard deviation of `a`/`b` (the standard
    two-sample pooled-variance estimator), expressed relative to the grand
    mean of both groups -- a scale-free coefficient of variation so it can
    be combined with a relative effect size in `required_repeats_per_arm`
    regardless of the metric's units. `None` if either group has fewer than
    2 observations (no within-group spread to estimate at all) or the
    grand mean is 0 (a relative figure is meaningless against a zero mean).
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return None
    grand_mean = statistics.mean(list(a) + list(b))
    if grand_mean == 0:
        return None
    pooled_var = ((n_a - 1) * statistics.variance(a) + (n_b - 1) * statistics.variance(b)) / (n_a + n_b - 2)
    return math.sqrt(pooled_var) / abs(grand_mean)


def required_repeats_per_arm(
    relative_effect: Optional[float],
    pooled_cv: Optional[float],
    *,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> Optional[int]:
    """Minimum-detectable-effect helper: how many repeats per arm (total,
    not "more than today") would be needed to reliably (`power`, default
    80%) detect an effect of `relative_effect` (a fraction, e.g. -0.18 for
    "18% lower") given `pooled_cv` (from `pooled_relative_stdev`) as the
    noise level, at significance `alpha`.

    Uses the standard two-sample z-test sample-size formula,
    `n = 2 * ((z_alpha/2 + z_power) * cv / effect) ** 2`: a normal
    approximation, not an exact permutation-test calculation, since the
    latter has no closed form -- but the result is then floored at
    `min_n_for_exact_significance(alpha)`, since no amount of averaging
    away noise changes the hard combinatorial floor an exact permutation
    test is subject to (see `min_two_sided_p`). This is what keeps the
    number honest at very large effect sizes / very low noise, where the
    z-test formula alone would suggest an n below that floor.

    `None` if `relative_effect` is 0/None (nothing to detect) or
    `pooled_cv` is None (not enough repeats yet to estimate the noise
    level -- need at least 2 per arm). A `pooled_cv` of exactly 0.0 (every
    observation in each arm identical at this sample size -- possible for
    an integer-valued metric like `turns` at n=3) is *not* treated as
    "insufficient data": it means no measured noise at all, so nothing
    stands between detecting the effect and `min_n_for_exact_significance`'s
    combinatorial floor, which is returned directly rather than dividing by
    zero in the z-test formula.
    """
    if not relative_effect or pooled_cv is None:
        return None
    floor = min_n_for_exact_significance(alpha) or 2
    if pooled_cv <= 0:
        return floor
    z_alpha = _norm_ppf(1 - alpha / 2)
    z_power = _norm_ppf(power)
    n = 2 * ((z_alpha + z_power) * pooled_cv / abs(relative_effect)) ** 2
    return max(floor, math.ceil(n))


# ---------------------------------------------------------------------------
# Composite: everything a single "is this arm's primary metric different
# from baseline" verdict needs, bundled so render.py doesn't have to
# re-derive any of the above by hand. Still no strings/formatting here --
# render.py owns the wording (arm labels, "cheaper" vs "higher", etc).
# ---------------------------------------------------------------------------


@dataclass
class MetricVerdict:
    baseline_mean: float
    arm_mean: float
    relative_effect: Optional[float]  # (arm_mean - baseline_mean) / |baseline_mean|, None if baseline_mean == 0
    n_baseline: int
    n_arm: int
    permutation: PermutationResult
    significant: bool
    required_repeats: Optional[int]


def metric_verdict(
    baseline_values: list,
    arm_values: list,
    *,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    seed: int = DEFAULT_SEED,
) -> Optional[MetricVerdict]:
    """Bundle a permutation test, an effect-size estimate, and a required-
    repeats estimate into one result for a single metric, one arm compared
    against the baseline arm. `None` if either side has no observations at
    all (nothing to compare)."""
    perm = permutation_test(baseline_values, arm_values, seed=seed)
    if perm is None:
        return None
    baseline_mean = statistics.mean(baseline_values)
    arm_mean = statistics.mean(arm_values)
    relative_effect = ((arm_mean - baseline_mean) / abs(baseline_mean)) if baseline_mean != 0 else None
    cv = pooled_relative_stdev(baseline_values, arm_values)
    required = required_repeats_per_arm(relative_effect, cv, alpha=alpha, power=power)
    return MetricVerdict(
        baseline_mean=baseline_mean,
        arm_mean=arm_mean,
        relative_effect=relative_effect,
        n_baseline=len(baseline_values),
        n_arm=len(arm_values),
        permutation=perm,
        significant=perm.p_value < alpha,
        required_repeats=required,
    )
