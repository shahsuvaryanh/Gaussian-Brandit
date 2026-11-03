import unittest

import numpy as np
import pandas as pd

from gaussian_bandit import select_alternative, simulate, update_posterior


class GaussianBanditTests(unittest.TestCase):
    def test_single_observation_closed_form(self):
        mean, variance = update_posterior(0, 25, -2, 1)
        self.assertAlmostEqual(variance, 25 / 26)
        self.assertAlmostEqual(mean, -50 / 26)

    def test_sequential_updates_match_batch_formula(self):
        rewards = [-3.0, 0.5, 8.0, -1.0]
        mean, variance = 0, 25
        for reward in rewards:
            mean, variance = update_posterior(mean, variance, reward, 3)
        expected_variance = 1 / (1 / 25 + len(rewards) / 9)
        self.assertAlmostEqual(variance, expected_variance)
        self.assertAlmostEqual(mean, expected_variance * sum(rewards) / 9)

    def test_only_selected_posterior_changes_and_batch_agreement(self):
        result = simulate([1, 2, 3], 3, 1000, 42)
        previous = np.zeros(3)
        for row in result.history.itertuples():
            current = np.array([row.estimated_mean_1, row.estimated_mean_2, row.estimated_mean_3])
            others = np.arange(3) != row.alternative - 1
            np.testing.assert_array_equal(current[others], previous[others])
            previous = current
        for row in result.summary.itertuples():
            rewards = result.history.loc[result.history.alternative == row.alternative, "reward"]
            variance = 1 / (1 / 25 + len(rewards) / 9)
            self.assertAlmostEqual(row.posterior_std ** 2, variance)
            self.assertAlmostEqual(row.posterior_mean, variance * rewards.sum() / 9)

    def test_required_experiments_reproducible_and_counts_correct(self):
        for sigma, rounds in [(1, 100), (1, 1000), (3, 1000)]:
            with self.subTest(sigma=sigma, rounds=rounds):
                a = simulate([1, 2, 3], sigma, rounds, 42)
                b = simulate([1, 2, 3], sigma, rounds, 42)
                pd.testing.assert_frame_equal(a.summary, b.summary)
                pd.testing.assert_frame_equal(a.history, b.history)
                self.assertEqual(int(a.summary.selection_count.sum()), rounds)
                self.assertAlmostEqual(a.total_reward, a.history.reward.sum())
                self.assertEqual(a.recommendation, int(a.summary.loc[a.summary.posterior_mean.idxmax(), "alternative"]))

    def test_negative_rewards_are_preserved(self):
        result = simulate([-100, -100, -100], 1, 10, 42)
        self.assertTrue((result.history.reward < 0).all())
        self.assertLess(result.total_reward, 0)

    def test_unselected_alternatives_retain_prior(self):
        result = simulate([1, 2, 3], 1, 1, 0)
        unselected = result.summary.query("selection_count == 0")
        self.assertEqual(len(unselected), 2)
        self.assertTrue((unselected.posterior_mean == 0).all())
        self.assertTrue((unselected.posterior_std == 5).all())

    def test_selection_samples_learned_posteriors(self):
        class FixedRng:
            def normal(self, means, stds):
                np.testing.assert_array_equal(means, [0, 2, 1])
                np.testing.assert_array_equal(stds, [5, 2, 1])
                return np.array([4, 3, 2])
        self.assertEqual(select_alternative([0, 2, 1], [25, 4, 1], FixedRng()), 0)

    def test_invalid_inputs(self):
        for means, sigma, rounds, seed in [
            ([1, 2], 1, 10, 0), ([1, 2, np.nan], 1, 10, 0),
            ([1, 2, 3], 0, 10, 0), ([1, 2, 3], -1, 10, 0),
            ([1, 2, 3], np.inf, 10, 0), ([1, 2, 3], 1, 0, 0),
            ([1, 2, 3], 1, 1.5, 0), ([1, 2, 3], 1, True, 0),
            ([1, 2, 3], 1, 10, -1),
        ]:
            with self.subTest(means=means, sigma=sigma, rounds=rounds, seed=seed):
                with self.assertRaises(ValueError):
                    simulate(means, sigma, rounds, seed)


if __name__ == "__main__":
    unittest.main()
