# DS223 Gaussian Thompson Sampling

A Streamlit application comparing the original click-based carousel bandit with a
three-alternative Gaussian reward simulation. The runnable frontend is at the
repository root. In the supplied course checkout, `app/`, `docs/`, and `tutorial/`
retain the starter material; the standalone export excludes those folders.
The frontend needs no database, backend, Docker, credentials, or external API.

## Credit

Adapted from [hovhannisyan91/DS223_Group_Project](https://github.com/hovhannisyan91/DS223_Group_Project),
the starter named in the assignment. This working copy was supplied through
[DS223_Bayesian_AB_Testing](https://github.com/hovhannisyan91/DS223_Bayesian_AB_Testing).
The original Streamlit carousel pages remain available.

## Setup and run

Install Python **3.12** and [uv](https://docs.astral.sh/uv/getting-started/installation/).
From this project's root:

```sh
uv sync --locked
uv run streamlit run main.py
```

Open the local URL printed by Streamlit. Choose **Gaussian Bandit**, set three true
means, known noise standard deviation, rounds, and seed, then press **Run**.
The table shows each true mean, posterior mean, posterior standard deviation, and
selection count. The page also shows total observed reward, the final recommendation,
learning curves, and a downloadable history. Results stay in session state across
page navigation; the displayed configuration always identifies the last completed run.
Closing the session or restarting the server may clear interactive results.

```sh
uv run python -m unittest discover -s tests -v
uv run python experiments.py
```

The second command regenerates `results/experiments.csv` and the three full histories.
The UI accepts 1–100,000 integer rounds, nonnegative integer seeds, finite true means
between -1,000,000 and 1,000,000, and noise up to 1,000,000. Zero noise is rejected.

## Original Beta-Bernoulli carousel

Create a carousel, choose the number of alternatives and visible slots, and open
Interaction. Each alternative begins with a Beta(1, 1) prior for its click
probability. Each round samples a probability from every alternative's posterior
and displays the alternatives with the highest samples. This is Thompson sampling:
uncertain alternatives can still be explored while strong alternatives tend to appear.

Clicking an alternative records reward 1 for that round; choosing no click records 0.
Every displayed alternative gets an impression. The clicked alternative receives
`alpha += 1` and a click; every other displayed alternative receives `beta += 1`.
If nothing is clicked, all displayed alternatives receive `beta += 1`. Alternatives
that were not displayed are unchanged. Posterior mean click probability is
`alpha / (alpha + beta)`. Analytics shows these posteriors, clicks, impressions,
and cumulative round reward. Its legacy “Average Reward” metric is clicks divided
by item impressions, which differs from average reward per round when multiple
alternatives are shown. “Resample without update” chooses a new display without
recording feedback.

`bandit_utils.py` stores carousels, posterior parameters, the active carousel,
current display, and event histories in `st.session_state`. All three original
pages read and modify that shared state. No frontend data is saved to PostgreSQL.
The delete button now removes the selected carousel from session state.

## Gaussian adaptation and posterior update

There are exactly three alternatives. Simulator rewards obey
`R | A=i ~ Normal(true_mean_i, sigma²)`, with shared **known** noise standard
deviation `sigma > 0`. The learner does not know the true means. Every mean starts
with the same `Normal(0, 25)` prior: variance 25, standard deviation 5.

On each round:

1. Sample one possible mean from each current Normal posterior.
2. Select the alternative with the largest sampled mean.
3. Generate one real-valued Gaussian reward from the selected alternative.
4. Update only that alternative's posterior and increment its selection count.

For old posterior mean `m`, variance `v`, and reward `r`:

```text
new_variance = 1 / (1/v + 1/sigma²)
new_mean = new_variance * (m/v + r/sigma²)
```

The implementation uses the equivalent numerically convenient weighted update:
`w = v/(v+sigma²)`, `new_mean = m+w*(r-m)`, `new_variance = sigma²*w`.
The known reward precision is `1/sigma²`; it is not the posterior precision of the
unknown mean. The mean's uncertainty shrinks as observations accumulate.
Rewards are never clipped, rounded, converted to clicks, or forced positive.
True means enter reward generation and evaluation only; selection receives just
learned means and variances. The final recommendation maximizes posterior mean,
not the most recent sampled value or selection count. Exact ties choose the lowest
alternative number.

## Required experiments

All experiments use true means `[1, 2, 3]`, prior `Normal(0, 25)`, and **seed 42**.
Each run resets both the prior and NumPy's local random generator.

| Experiment | σ | Rounds | Posterior means (1, 2, 3) | Selection counts (1, 2, 3) | Total reward | Recommendation |
|---|---:|---:|---|---|---:|---|
| Short run | 1 | 100 | 1.570217, 1.728703, 2.994983 | 4, 2, 94 | 291.518390 | Alternative 3 |
| Longer run | 1 | 1,000 | 1.306247, 2.142116, 2.979191 | 5, 13, 982 | 2960.201477 | Alternative 3 |
| Higher noise | 3 | 1,000 | 1.153692, 1.483757, 2.954223 | 21, 23, 956 | 2884.604431 | Alternative 3 |

The longer low-noise run allocates 98.2% of rounds to the best alternative, compared
with 94% in the short run. Its posterior standard deviation for alternative 3 falls
from 0.103120 to 0.031911. More observations provide more certainty, although they
do not guarantee an estimate closer to truth in every realized run: alternative 3's
short-run estimate happens to be closer to 3 than its longer-run estimate.
Inferior alternatives receive few samples, so their estimates can remain inaccurate.

Increasing σ from 1 to 3 reduces each observation's precision by a factor of nine.
At 1,000 rounds the algorithm explores alternatives 1 and 2 more often (44 total
selections versus 18). Alternative 3's posterior standard deviation is 0.097009,
larger than 0.031911 in the low-noise run. Alternative 2's estimate, 1.483757, is
noticeably below its true mean of 2; noisy, adaptively collected finite samples can
produce such results. All three recommendations are correct for this seed, but
finite-run success is not guaranteed for every seed or configuration. These are
three illustrative runs, not a statistical comparison over many seeds.

Tests verify identical configurations and seeds reproduce the entire history,
counts sum to rounds, negative rewards survive, only the selected posterior changes,
and sequential updates match an independent batch formula. The experiment script
also checks reproducibility and counts before writing its output. Reproduction
assumes the same code and locked dependency environment.

## Repository and deployment

- Own submission repository URL: **pending publication**.
- Deployed Streamlit URL: **pending deployment and verification**.
- Local application: `http://localhost:8501` when the server is running.
- Presentation notes: [PRESENTATION.md](PRESENTATION.md).

The supplied course checkout's origin belongs to the teacher. Publish the frontend to a new
repository under your own GitHub account. `python tools/export_frontend.py` builds
an independent frontend directory at `submission/gaussian-bandit/` and a zip file
next to it. It includes code, pages, setup files, README, tests, experiments, results,
presentation notes, and CI; it excludes the old backend, ETL, local environments,
caches, credentials, and the starter's Git history. Initialize a new Git repository
inside that exported directory on branch `main` and push it to your own repository.

On [Streamlit Community Cloud](https://share.streamlit.io/), choose that repository,
branch **main**, entrypoint **main.py**, and Python **3.12** in Advanced settings.
No secrets are required. Follow the official
[deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
After deployment, run the default configuration, check the results against the
longer-run row above, test zero-noise rejection, and navigate through the original
pages. Add the actual repository and deployed URLs above only after publication
and verification. The assignment submission consists of those two URLs.
