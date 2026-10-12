
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ==========================================================
# CLAIM COST ANALYSIS MODULE
# ==========================================================

def show_claim_cost_analysis(df: pd.DataFrame) -> None:
    """
    Detailed statistical analysis of annual motor insurance claim costs.

    This module supplements, rather than repeats, the basic claim-cost
    summaries in app.py and exploratory_analysis.py.
    """

    st.header("Claim Cost Deep Dive")

    if df is None or df.empty:
        st.warning("No data available. Upload the dataset first.")
        return

    if "Cost_claims_year" not in df.columns:
        st.error("The required Cost_claims_year column is missing.")
        return

    data = df.copy()
    cost = pd.to_numeric(data["Cost_claims_year"], errors="coerce")

    valid = cost.dropna()
    positive = valid[valid > 0]
    negative = valid[valid < 0]
    zero = valid[valid == 0]

    st.write(
        "This module investigates the distribution, severity and "
        "concentration of annual claim costs. It does not automatically "
        "remove extreme observations or fit the final insurance model."
    )

    # ------------------------------------------------------
    # 1. POSITIVE ANNUAL CLAIM-COST SEVERITY
    # ------------------------------------------------------

    st.subheader("1. Positive annual claim-cost severity")

    if positive.empty:
        st.warning("No positive annual claim costs are available.")
        return

    severity = positive.astype(float)

    q25 = severity.quantile(0.25)
    q50 = severity.quantile(0.50)
    q75 = severity.quantile(0.75)
    q90 = severity.quantile(0.90)
    q95 = severity.quantile(0.95)
    q99 = severity.quantile(0.99)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Positive-cost records", f"{len(severity):,}")
    c2.metric("Mean positive cost", f"{severity.mean():,.2f}")
    c3.metric("Median positive cost", f"{q50:,.2f}")
    c4.metric("Maximum positive cost", f"{severity.max():,.2f}")

    severity_summary = pd.DataFrame(
        {
            "Statistic": [
                "Count",
                "Mean",
                "Standard deviation",
                "Minimum",
                "25th percentile",
                "Median",
                "75th percentile",
                "90th percentile",
                "95th percentile",
                "99th percentile",
                "Maximum",
                "Skewness",
                "Coefficient of variation",
            ],
            "Value": [
                len(severity),
                severity.mean(),
                severity.std(),
                severity.min(),
                q25,
                q50,
                q75,
                q90,
                q95,
                q99,
                severity.max(),
                severity.skew(),
                (
                    severity.std() / severity.mean()
                    if severity.mean() != 0
                    else np.nan
                ),
            ],
        }
    )

    st.dataframe(
        severity_summary,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "These statistics describe positive annual aggregate costs, "
        "not the severity of individual claims. The coefficient of "
        "variation is standard deviation divided by mean."
    )

    # ------------------------------------------------------
    # 2. TAIL BEHAVIOUR
    # ------------------------------------------------------

    st.divider()
    st.subheader("2. Tail behaviour and loss concentration")

    st.write(
        "A small number of large annual-loss records can account for "
        "a substantial proportion of total recorded claim costs."
    )

    total_positive_cost = float(severity.sum())

    if total_positive_cost > 0:
        tail_rows = []

        for percentage in [1, 5, 10, 20]:
            n_top = max(
                1,
                int(np.ceil(len(severity) * percentage / 100)),
            )
            top_costs = severity.nlargest(n_top)

            tail_rows.append(
                {
                    "Largest records": f"Top {percentage}%",
                    "Number of records": n_top,
                    "Cost represented": top_costs.sum(),
                    "Share of positive recorded costs (%)": (
                        100 * top_costs.sum() / total_positive_cost
                    ),
                }
            )

        tail_table = pd.DataFrame(tail_rows)

        st.dataframe(
            tail_table,
            hide_index=True,
            use_container_width=True,
        )

        fig, ax = plt.subplots()
        ax.bar(
            tail_table["Largest records"],
            tail_table["Share of positive recorded costs (%)"],
        )
        ax.set_xlabel("Largest annual-loss records")
        ax.set_ylabel("Share of positive recorded costs (%)")
        ax.set_title("Concentration of recorded annual claim costs")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Concentration is calculated across records, not across "
            "individual claims. It can help identify the importance "
            "of the upper tail in the observed portfolio."
        )
    else:
        st.info("The total of positive claim costs is zero.")

    # ------------------------------------------------------
    # 3. DISTRIBUTION SHAPE
    # ------------------------------------------------------

    st.divider()
    st.subheader("3. Distribution shape")

    left, right = st.columns(2)

    with left:
        st.write("**Positive costs on the original scale**")

        fig, ax = plt.subplots()
        ax.hist(severity, bins=50)
        ax.set_xlabel("Positive annual claim cost")
        ax.set_ylabel("Number of records")
        ax.set_title("Positive-cost distribution")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.write("**Positive costs on a logarithmic scale**")

        fig, ax = plt.subplots()
        ax.hist(np.log10(severity), bins=50)
        ax.set_xlabel("log10(positive annual claim cost)")
        ax.set_ylabel("Number of records")
        ax.set_title("Log-transformed positive-cost distribution")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.info(
        "The logarithmic display helps reveal structure when costs vary "
        "widely. It is a visualization only; it does not mean that the "
        "final claim-cost model should automatically use log-transformed "
        "ordinary least squares."
    )

    # ------------------------------------------------------
    # 4. EXTREME-OBSERVATION REVIEW
    # ------------------------------------------------------

    st.divider()
    st.subheader("4. Investigate unusually large annual costs")

    percentile_threshold = st.slider(
        "Flag records above this positive-cost percentile",
        min_value=90,
        max_value=99,
        value=95,
        step=1,
        key="claim_cost_tail_percentile",
    )

    threshold = float(severity.quantile(percentile_threshold / 100))

    st.metric(
        f"{percentile_threshold}th percentile threshold",
        f"{threshold:,.2f}",
    )

    extreme_mask = cost > threshold
    extreme = data.loc[extreme_mask].copy()
    extreme["Cost_claims_year"] = cost.loc[extreme_mask]

    sort_columns = ["Cost_claims_year"]
    extreme = extreme.sort_values(
        sort_columns,
        ascending=False,
    )

    display_candidates = [
        "ID",
        "Date_last_renewal",
        "Date_next_renewal",
        "Type_risk",
        "Area",
        "Premium",
        "N_claims_year",
        "Cost_claims_year",
    ]

    display_columns = [
        col for col in display_candidates
        if col in extreme.columns
    ]

    st.write(
        f"**{len(extreme):,} records** exceed the selected threshold."
    )

    if display_columns:
        st.dataframe(
            extreme[display_columns].head(100),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.dataframe(extreme.head(100), use_container_width=True)

    extreme_csv = extreme.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download flagged annual-loss records",
        data=extreme_csv,
        file_name="motor_insurance_flagged_annual_costs.csv",
        mime="text/csv",
        key="download_extreme_claim_costs",
    )

    st.warning(
        "A statistical flag is not proof of a data error. Check source "
        "records, exposure, currency units, claim development and possible "
        "large genuine losses before deciding how to handle an observation."
    )

    # ------------------------------------------------------
    # 5. COST-DATA VALIDATION
    # ------------------------------------------------------

    st.divider()
    st.subheader("5. Claim-cost validation checks")

    validation = pd.DataFrame(
        [
            {
                "Check": "Missing claim costs",
                "Count": int(cost.isna().sum()),
                "Interpretation": "Cost is unavailable or could not be converted.",
            },
            {
                "Check": "Zero annual costs",
                "Count": int((cost == 0).sum()),
                "Interpretation": "No positive annual cost recorded.",
            },
            {
                "Check": "Positive annual costs",
                "Count": int((cost > 0).sum()),
                "Interpretation": "Positive aggregate annual cost.",
            },
            {
                "Check": "Negative annual costs",
                "Count": int((cost < 0).sum()),
                "Interpretation": "Investigate adjustments, recoveries or data issues.",
            },
            {
                "Check": "Infinite values",
                "Count": int(np.isinf(cost.dropna()).sum()),
                "Interpretation": "Check invalid numerical values.",
            },
        ]
    )

    st.dataframe(
        validation,
        hide_index=True,
        use_container_width=True,
    )

    if not negative.empty:
        st.warning(
            "Negative values are present. Investigate their meaning before "
            "fitting a standard Tweedie model with power between 1 and 2, "
            "because that model assumes non-negative outcomes."
        )

    # ------------------------------------------------------
    # 6. EMPIRICAL FEATURES RELEVANT TO TWEEDIE MODELLING
    # ------------------------------------------------------

    st.divider()
    st.subheader("6. Empirical features relevant to a Tweedie model")

    n_valid = len(valid)
    zero_share = float((valid == 0).mean()) if n_valid else np.nan
    positive_share = float((valid > 0).mean()) if n_valid else np.nan

    a, b, c = st.columns(3)
    a.metric(
        "Zero-cost share",
        f"{100 * zero_share:.2f}%" if pd.notna(zero_share) else "N/A",
    )
    b.metric(
        "Positive-cost share",
        f"{100 * positive_share:.2f}%" if pd.notna(positive_share) else "N/A",
    )
    c.metric(
        "Negative-cost records",
        f"{len(negative):,}",
    )

    if len(negative) > 0:
        st.error(
            "Investigate negative outcomes before considering a standard "
            "compound Poisson-Gamma Tweedie model."
        )
    elif len(zero) > 0 and len(positive) > 0:
        st.success(
            "The observed outcome contains both zero and positive annual "
            "costs, a pattern compatible with considering a compound "
            "Poisson-Gamma Tweedie model."
        )
    else:
        st.info(
            "The observed outcomes do not show both zero and positive "
            "values in this sample. Check the data and modelling assumptions."
        )

    st.markdown(
        """
        **What these diagnostics can tell us**

        - A point mass at zero and positive continuous costs are features
          a compound Poisson-Gamma Tweedie model can accommodate.
        - Strong right skew and high tail concentration show why the mean
          alone may not adequately describe annual costs.
        - Negative values, invalid observations, exposure differences and
          repeated policy-product records need investigation.

        **What they cannot tell us**

        These summaries do not estimate the Tweedie power parameter, prove
        the distribution is correct, establish independence between records,
        or demonstrate that shared random effects improve prediction.
        Those questions require model fitting and out-of-sample evaluation.
        """
    )
