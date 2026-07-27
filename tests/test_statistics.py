"""Known-answer tests for ys/statistics.py (feature 3, IMPROVEMENTS.md).

Statistical code is easy to get subtly wrong, so every test here either
hand-computes the expected number from the underlying formula/enumeration,
or pins an exact value traced directly from the (documented, seeded)
algorithm -- "it returns a number" is not a test.
"""
import pytest

from ys import statistics as s


# ---------------------------------------------------------------------------
# _norm_ppf -- standard normal quantile function (Acklam's approximation).
# ---------------------------------------------------------------------------


def test_norm_ppf_matches_well_known_z_values():
    # z_0.975 (two-sided 95% CI) and z_0.80 (80% power) are two of the most
    # commonly tabulated normal quantiles -- 1.959964 and 0.841621
    # respectively, to 6 decimal places, in any standard statistics table.
    assert s._norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)
    assert s._norm_ppf(0.8) == pytest.approx(0.841621, abs=1e-5)
    assert s._norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)


def test_norm_ppf_rejects_values_outside_open_unit_interval():
    with pytest.raises(ValueError):
        s._norm_ppf(0.0)
    with pytest.raises(ValueError):
        s._norm_ppf(1.0)


# ---------------------------------------------------------------------------
# wilson_interval
# ---------------------------------------------------------------------------


def test_wilson_interval_matches_hand_computed_formula():
    # Hand-computed via the standard Wilson score formula for x=2, n=3,
    # z=1.959963984540054 (95%): center=(phat + z^2/2n)/(1+z^2/n), margin=
    # z*sqrt(phat(1-phat)/n + z^2/4n^2)/(1+z^2/n).
    w = s.wilson_interval(2, 3)
    assert w.point == pytest.approx(2 / 3)
    assert w.low == pytest.approx(0.207660, abs=1e-5)
    assert w.high == pytest.approx(0.938508, abs=1e-5)


def test_wilson_interval_full_success_is_not_zero_width():
    # The normal approximation collapses to a zero-width interval at 100%
    # (or 0%) observed -- exactly the case Wilson exists to fix, and exactly
    # what n=3 runs of a real experiment produce whenever nothing fails.
    w = s.wilson_interval(3, 3)
    assert w.point == 1.0
    assert w.low > 0.0
    assert w.high == pytest.approx(1.0)
    assert w.high - w.low > 0.0


def test_wilson_interval_none_for_zero_n():
    assert s.wilson_interval(0, 0) is None


# ---------------------------------------------------------------------------
# min_two_sided_p / min_n_for_exact_significance -- the headline structural
# fact: n=3 vs n=3 cannot reach p<0.05 in an exact two-sided permutation
# test, no matter the data.
# ---------------------------------------------------------------------------


def test_min_two_sided_p_at_n3_is_one_tenth():
    # C(6,3) = 20 total relabelings; exactly 2 (the true split and its
    # mirror) reproduce the maximal separation -> 2/20 = 0.1.
    assert s.min_two_sided_p(3, 3) == pytest.approx(0.1)


def test_min_two_sided_p_at_n4_clears_the_005_bar():
    # C(8,4) = 70 -> 2/70 ~= 0.02857, the first n at which significance is
    # even structurally possible for equal-sized groups.
    assert s.min_two_sided_p(4, 4) == pytest.approx(2 / 70)


def test_min_n_for_exact_significance_is_four_at_alpha_05():
    assert s.min_n_for_exact_significance() == 4
    # n=3 must fail the 0.05 bar, n=4 must clear it -- pin both directions.
    assert s.min_two_sided_p(3, 3) >= 0.05
    assert s.min_two_sided_p(4, 4) < 0.05


# ---------------------------------------------------------------------------
# permutation_test -- exact enumeration on tiny, hand-enumerable datasets.
# ---------------------------------------------------------------------------


def test_permutation_test_exact_p_value_on_fully_separated_n3_groups():
    # a and b don't overlap at all -- the most extreme possible split at
    # this n. Exactly 2 of C(6,3)=20 relabelings (the true split and its
    # mirror-image relabeling) are at least this extreme -> p = 2/20 = 0.1,
    # matching min_two_sided_p(3, 3) exactly since this *is* the extreme case.
    result = s.permutation_test([1, 2, 3], [4, 5, 6])
    assert result.exact is True
    assert result.n_permutations == 20
    assert result.p_value == pytest.approx(0.1)
    assert result.min_possible_p == pytest.approx(0.1)
    assert result.observed_diff == pytest.approx(3.0)


def test_permutation_test_hand_enumerated_n2_vs_n2():
    # a=[1,2], b=[3,4]: pooled=[1,2,3,4], observed = mean(b)-mean(a) = 2.0.
    # All C(4,2)=6 ways to split 2-and-2, hand-enumerated (group A / group B
    # / |diff|):
    #   {1,2}/{3,4} -> |2.0| >= 2.0  yes
    #   {1,3}/{2,4} -> |1.0|         no
    #   {1,4}/{2,3} -> |0.0|         no
    #   {2,3}/{1,4} -> |0.0|         no
    #   {2,4}/{1,3} -> |1.0|         no
    #   {3,4}/{1,2} -> |2.0| >= 2.0  yes
    # 2 of 6 are as extreme as observed -> p = 1/3.
    result = s.permutation_test([1, 2], [3, 4])
    assert result.exact is True
    assert result.n_permutations == 6
    assert result.p_value == pytest.approx(1 / 3)


def test_permutation_test_p_is_1_when_groups_are_indistinguishable():
    # Identical multisets in both groups -> observed diff is 0, and abs(any
    # relabeling's diff) >= 0 - epsilon is true for every relabeling.
    result = s.permutation_test([1, 2, 3], [1, 2, 3])
    assert result.p_value == pytest.approx(1.0)
    assert result.observed_diff == pytest.approx(0.0)


def test_permutation_test_none_when_either_group_empty():
    assert s.permutation_test([], [1, 2, 3]) is None
    assert s.permutation_test([1, 2, 3], []) is None


def test_permutation_test_is_deterministic_across_repeated_calls():
    """The load-bearing property for feature 3: `ys compare` on unchanged
    data must produce the exact same verdict every time, not merely a
    statistically-similar one. Uses n large enough to force the Monte Carlo
    fallback (n=15 vs n=15 is C(30,15) ~= 1.55e8, far past
    max_exact_combinations) -- the exact-enumeration path is already
    deterministic by construction (no randomness at all), so this is the
    path that actually exercises the seeded RNG."""
    a = list(range(15))
    b = [x + 2 for x in range(15)]
    first = s.permutation_test(a, b)
    second = s.permutation_test(a, b)
    assert first.exact is False
    assert first == second


# ---------------------------------------------------------------------------
# bootstrap_ci -- pinned against a value traced directly from the seeded RNG.
# ---------------------------------------------------------------------------


def test_bootstrap_ci_pinned_to_traced_rng_output():
    """With `random.Random(DEFAULT_SEED)` and 5 resamples of [10, 20, 30],
    tracing `rng.randrange(3)` by hand against the algorithm in
    `bootstrap_ci` produces resample means
    [13.333.., 10.0, 20.0, 16.666.., 16.666..] -> sorted
    [10.0, 13.333.., 16.666.., 16.666.., 20.0]. `low_idx`/`high_idx` at
    alpha=0.05 over 5 resamples both round to index 0 and 4 respectively
    (0.025*4=0.1 -> 0, 0.975*4=3.9 -> 4), giving low=10.0, high=20.0 --
    pinned here so a change to the resampling algorithm (not just its
    result on this data) is caught."""
    result = s.bootstrap_ci([10, 20, 30], n_resamples=5, seed=s.DEFAULT_SEED)
    assert result.point == pytest.approx(20.0)
    assert result.low == pytest.approx(10.0)
    assert result.high == pytest.approx(20.0)
    assert result.n_resamples == 5


def test_bootstrap_ci_is_deterministic_across_repeated_calls():
    values = [3.1, 4.7, 2.9, 5.5, 4.0]
    first = s.bootstrap_ci(values)
    second = s.bootstrap_ci(values)
    assert first == second


def test_bootstrap_ci_none_with_fewer_than_two_observations():
    assert s.bootstrap_ci([]) is None
    assert s.bootstrap_ci([5.0]) is None


def test_bootstrap_ci_brackets_the_observed_mean_on_symmetric_data():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    result = s.bootstrap_ci(values, n_resamples=5000)
    assert result.low <= result.point <= result.high


# ---------------------------------------------------------------------------
# pooled_relative_stdev / required_repeats_per_arm -- the minimum-detectable-
# effect helper.
# ---------------------------------------------------------------------------


def test_pooled_relative_stdev_none_with_fewer_than_two_per_group():
    assert s.pooled_relative_stdev([1.0], [1.0, 2.0]) is None
    assert s.pooled_relative_stdev([1.0, 2.0], []) is None


def test_pooled_relative_stdev_zero_for_identical_values_within_each_group():
    assert s.pooled_relative_stdev([5.0, 5.0, 5.0], [4.0, 4.0, 4.0]) == pytest.approx(0.0)


def test_required_repeats_per_arm_none_without_effect_or_variance():
    assert s.required_repeats_per_arm(None, 0.1) is None
    assert s.required_repeats_per_arm(0.0, 0.1) is None
    assert s.required_repeats_per_arm(-0.18, None) is None


def test_required_repeats_per_arm_floors_at_structural_minimum_with_zero_noise():
    """Perfectly separated, noise-free data (pooled_cv == 0.0) still can't
    beat the combinatorial floor an exact permutation test is subject to --
    required_repeats_per_arm must return that floor (4 at the default
    alpha=0.05), not None and not some smaller number from a formula that
    would otherwise divide by zero."""
    assert s.required_repeats_per_arm(-0.2, 0.0) == s.min_n_for_exact_significance()


def test_required_repeats_per_arm_increases_with_more_noise():
    low_noise = s.required_repeats_per_arm(-0.18, 0.05)
    high_noise = s.required_repeats_per_arm(-0.18, 0.5)
    assert low_noise <= high_noise


def test_required_repeats_per_arm_decreases_with_larger_effect():
    small_effect = s.required_repeats_per_arm(-0.05, 0.3)
    large_effect = s.required_repeats_per_arm(-0.5, 0.3)
    assert large_effect <= small_effect


# ---------------------------------------------------------------------------
# metric_verdict -- the composite the verdict line is built from. Pins the
# specific claim IMPROVEMENTS.md feature 3 makes by name: at n=3 vs n=3, no
# permutation test can claim significance, however large the effect.
# ---------------------------------------------------------------------------


def test_metric_verdict_at_n3_never_claims_significance():
    baseline = [1.02, 0.98, 1.00]
    arm = [0.80, 0.85, 0.81]  # ~18% lower, cleanly separated from baseline
    v = s.metric_verdict(baseline, arm)
    assert v.n_baseline == 3
    assert v.n_arm == 3
    assert v.significant is False
    assert v.permutation.p_value == pytest.approx(0.1)
    assert v.permutation.min_possible_p == pytest.approx(0.1)
    assert v.relative_effect == pytest.approx(-0.18, abs=1e-9)
    assert v.required_repeats is not None


def test_metric_verdict_none_when_either_side_has_no_data():
    assert s.metric_verdict([], [1.0, 2.0]) is None
    assert s.metric_verdict([1.0, 2.0], []) is None


def test_metric_verdict_is_deterministic_across_repeated_calls():
    baseline = [1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.03, 0.97, 1.01, 0.99, 1.04]
    arm = [0.8, 0.82, 0.78, 0.81, 0.79, 0.83, 0.77, 0.80, 0.82, 0.79, 0.81, 0.78]
    first = s.metric_verdict(baseline, arm)
    second = s.metric_verdict(baseline, arm)
    assert first == second


def test_metric_verdict_can_be_significant_at_larger_n():
    # Same ~30% effect as the n=3 case above but with n=8 per arm and clean
    # separation -- large enough to clear both the combinatorial floor
    # (n=4) and reach p < 0.05 on this exact data.
    baseline = [10, 11, 9, 10, 10, 11, 9, 10]
    arm = [7, 8, 6, 7, 7, 8, 6, 7]
    v = s.metric_verdict(baseline, arm)
    assert v.significant is True
    assert v.permutation.p_value < 0.05


def test_metric_verdict_relative_effect_none_for_zero_baseline_mean():
    v = s.metric_verdict([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
    assert v.relative_effect is None
    # required_repeats_per_arm can't compute anything from a None effect.
    assert v.required_repeats is None


def test_min_two_sided_p_none_for_empty_group():
    assert s.min_two_sided_p(0, 3) is None
    assert s.min_two_sided_p(3, 0) is None
