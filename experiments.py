"""Regenerate the required experiment CSVs using one fixed seed and prior."""

from pathlib import Path

import pandas as pd

from gaussian_bandit import simulate

SEED = 42
CONFIGURATIONS = [("Short run", 1, 100), ("Longer run", 1, 1000), ("Higher noise", 3, 1000)]


def main():
    output = Path(__file__).resolve().parent / "results"
    output.mkdir(exist_ok=True)
    tables = []
    for name, sigma, rounds in CONFIGURATIONS:
        result = simulate([1, 2, 3], sigma, rounds, SEED)
        repeated = simulate([1, 2, 3], sigma, rounds, SEED)
        pd.testing.assert_frame_equal(result.summary, repeated.summary)
        pd.testing.assert_frame_equal(result.history, repeated.history)
        assert result.summary.selection_count.sum() == rounds
        table = result.summary.assign(experiment=name, sigma=sigma, rounds=rounds,
                                      seed=SEED, total_reward=result.total_reward,
                                      recommendation=result.recommendation)
        tables.append(table)
        result.history.to_csv(output / f"{name.lower().replace(' ', '_')}_history.csv", index=False)
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(output / "experiments.csv", index=False)
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
