"""Interactive Gaussian Thompson sampling simulation."""

import streamlit as st

from gaussian_bandit import simulate

st.title("Gaussian Bandit")
st.write("Learn which of three alternatives has the highest mean continuous reward.")
st.caption("Each mean starts with a Normal prior: mean 0, variance 25 (standard deviation 5).")

with st.form("gaussian_configuration"):
    columns = st.columns(3)
    true_means = [column.number_input(f"Alternative {i + 1} true mean", value=float(i + 1),
                                      min_value=-1_000_000.0, max_value=1_000_000.0)
                  for i, column in enumerate(columns)]
    sigma = st.number_input("Known noise standard deviation σ", value=1.0, step=0.1,
                            min_value=0.0, max_value=1_000_000.0)
    rounds = st.number_input("Number of rounds", min_value=1, max_value=100_000,
                             value=1000, step=1)
    seed = st.number_input("Random seed", min_value=0, max_value=2**32 - 1, value=42, step=1)
    run = st.form_submit_button("Run", type="primary")

if run:
    try:
        result = simulate(true_means, sigma, rounds, seed)
    except ValueError as error:
        st.error(str(error))
    else:
        st.session_state.gaussian_result = result
        st.session_state.gaussian_config = {
            "true_means": true_means, "sigma": sigma, "rounds": rounds, "seed": seed,
        }

if "gaussian_result" in st.session_state:
    result = st.session_state.gaussian_result
    config = st.session_state.gaussian_config
    st.subheader("Last completed run")
    st.caption(f"True means: {config['true_means']} · σ: {config['sigma']} · "
               f"Rounds: {config['rounds']:,} · Seed: {config['seed']}")
    c1, c2 = st.columns(2)
    c1.metric("Total observed reward", f"{result.total_reward:,.3f}")
    c2.metric("Final recommendation", f"Alternative {result.recommendation}")
    st.caption("Recommendation uses the largest posterior mean; ties use the lowest alternative number.")
    st.dataframe(result.summary, hide_index=True, width="stretch")
    st.subheader("Estimated means over time")
    st.line_chart(result.history.set_index("round")[[f"estimated_mean_{i}" for i in (1, 2, 3)]])
    st.subheader("Cumulative observed reward")
    st.line_chart(result.history.set_index("round")["cumulative_reward"])
    st.download_button("Download round history", result.history.to_csv(index=False),
                       "gaussian_history.csv", "text/csv")

with st.expander("How the posterior update works"):
    st.write("Sample a possible mean from each posterior and select the largest. Generate one "
             "reward using that alternative's true mean and known σ. Update only its posterior. "
             "Rewards remain continuous and may be negative. True means never enter selection.")
    st.latex(r"v_{new}=\left(1/v_{old}+1/\sigma^2\right)^{-1}")
    st.latex(r"m_{new}=v_{new}\left(m_{old}/v_{old}+r/\sigma^2\right)")
