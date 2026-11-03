# Gaussian bandit presentation — 3–5 minutes

## 0:00–0:45 — Problem and starter

We want to learn which alternative gives the largest expected reward while still
exploring uncertain alternatives. The original carousel uses click/no-click
feedback and Beta distributions for click probabilities. Show Create Carousels,
submit a click or no click in Interaction, then open Analytics.

## 0:45–1:45 — Gaussian adaptation

Our new page uses three alternatives with continuous Normal rewards. Each unknown
mean starts with a Normal prior of mean zero and variance 25. Each round samples a
possible mean from each learned posterior and chooses the largest. Only the chosen
alternative gets a reward and posterior update. The true means belong to the
simulator; selection never sees them. Negative rewards remain valid observations.

Show the update in the page's explanation panel. Explain that known σ describes
observation noise, and observation precision is 1/σ². Posterior precision adds to
the previous precision. The updated mean is a precision-weighted average, and
posterior uncertainty decreases with observations. Normal(0, 25) uses variance 25,
not standard deviation 25.

## 1:45–2:45 — Live demo

Open the deployed Gaussian Bandit page. Enter means 1, 2, 3; σ=1; 1,000 rounds;
seed 42. Run and point out counts 5, 13, 982, total reward about 2960.201, and the
recommendation of alternative 3. Explain the recommendation is the largest final
posterior mean. Rerun to show reproducibility. Show the learning curve and history
download. Demonstrate that σ=0 produces an error.

## 2:45–3:45 — Experiments and interpretation

Compare the three README rows. More rounds concentrate choices on alternative 3
and reduce its uncertainty. Greater noise reduces precision per observation,
encourages more exploration, and leaves more uncertainty. More observations do not
guarantee every realized estimate is closer to truth: the short run's estimate for
alternative 3 happens to be closer. Rarely selected alternatives can retain poor
estimates. All recommendations work for this seed; this is not a universal guarantee.

## 3:45–4:15 — Verification and submission

Mention closed-form posterior checks, reproducibility, negative rewards, selected-only
updates, count totals, and UI checks. Show the actual published repository and app
URLs. For a team, divide these sections so every member presents part of the work.
Replace the local demo with the deployed app only after cloud verification succeeds.
