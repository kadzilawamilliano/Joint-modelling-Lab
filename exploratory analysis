
from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ==========================================================
# EXPLORATORY ANALYSIS MODULE
# ==========================================================

def show_exploratory_analysis(df: pd.DataFrame) -> None:
    """
    Display exploratory analysis for the motor insurance dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        The cleaned dataframe supplied by the main Streamlit app.
    """

    st.header("Exploratory Data Analysis")

    if df is None or df.empty:
        st.warning("No data available. Please load the dataset first.")
        return

    data = df.copy()

    st.write(
        "Explore claim costs, claim frequency, renewal outcomes, "
        "risk groups and relationships between numerical variables."
    )

    # ------------------------------------------------------
    # 1. BASIC DATASET SUMMARY
    # ------------------------------------------------------

    st.subheader("1. Dataset overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Records", f"{len(data):,}")
    col2.metric("Variables", f"{data.shape[1]:,}")

    if "ID" in data.columns:
        col3.metric("Unique IDs", f"{data['ID'].nunique():,}")
    else:
        col3.metric("Unique IDs", "Not available")

    if "Cost_claims_year" in data.columns:
        costs = pd.to_numeric(
            data["Cost_claims_year"], errors="coerce"
        )
        col4.metric(
            "Positive claim-cost records",
            f"{(costs > 0).sum():,}"
        )
    else:
        col4.metric("Positive claim-cost records", "Not available")

    with st.expander("Preview data"):
        st.dataframe(data.head(20), use_container_width=True)

    # ------------------------------------------------------
    # 2. CLAIM COST ANALYSIS
    # ------------------------------------------------------

    st.divider()
    st.subheader("2. Annual claim-cost analysis")

    if "Cost_claims_year" not in data.columns:
        st.info("The Cost_claims_year column is not available.")
    else:
        cost = pd.to_numeric(
            data["Cost_claims_year"], errors="coerce"
        )

        valid_cost = cost.dropna()
        positive_cost = valid_cost[valid_cost > 0]

        if not valid_cost.empty:
            zero_pct = (valid_cost == 0).mean() * 100
            positive_pct = (valid_cost > 0).mean() * 100

            a, b, c = st.columns(3)

            a.metric("Zero-cost records", f"{zero_pct:.2f}%")
            b.metric("Positive-cost records", f"{positive_pct:.2f}%")
            c.metric(
                "Mean annual claim cost",
                f"{valid_cost.mean():,.2f}"
            )

            cost_summary = valid_cost.describe().to_frame(
                name="Claim cost"
            )

            st.write("**Claim-cost summary statistics**")
            st.dataframe(cost_summary, use_container_width=True)

            left, right = st.columns(2)

            with left:
                st.write("**Zero versus positive claim costs**")

                fig, ax = plt.subplots()
                ax.bar(
                    ["Zero cost", "Positive cost"],
                    [
                        (valid_cost == 0).sum(),
                        (valid_cost > 0).sum(),
                    ],
                )
                ax.set_ylabel("Number of records")
                ax.set_title("Annual claim-cost status")
                ax.tick_params(axis="x", rotation=10)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with right:
                st.write("**Distribution of positive claim costs**")

                if not positive_cost.empty:
                    upper = positive_cost.quantile(0.99)
                    plotted = positive_cost[
                        positive_cost <= upper
                    ]

                    fig, ax = plt.subplots()
                    ax.hist(plotted, bins=40)
                    ax.set_xlabel("Annual claim cost")
                    ax.set_ylabel("Number of records")
                    ax.set_title(
                        "Positive costs (up to the 99th percentile)"
                    )
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.caption(
                        "The histogram excludes positive costs above "
                        "the 99th percentile for readability. Summary "
                        "statistics above use all valid costs."
                    )
                else:
                    st.info("No positive claim costs were found.")

    # ------------------------------------------------------
    # 3. CLAIM FREQUENCY
    # ------------------------------------------------------

    st.divider()
    st.subheader("3. Annual claim frequency")

    if "N_claims_year" not in data.columns:
        st.info("The N_claims_year column is not available.")
    else:
        frequency = pd.to_numeric(
            data["N_claims_year"], errors="coerce"
        ).dropna()

        if not frequency.empty:
            left, right = st.columns(2)

            with left:
                st.write("**Claim-count distribution**")

                freq_counts = (
                    frequency.value_counts()
                    .sort_index()
                    .head(30)
                )

                fig, ax = plt.subplots()
                ax.bar(
                    freq_counts.index.astype(str),
                    freq_counts.values,
                )
                ax.set_xlabel("Number of claims in the year")
                ax.set_ylabel("Number of records")
                ax.set_title("Annual claim frequency")
                ax.tick_params(axis="x", rotation=45)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with right:
                st.write("**Frequency summary**")

                st.dataframe(
                    frequency.describe().to_frame(
                        name="Annual claim count"
                    ),
                    use_container_width=True,
                )

                st.write(
                    "Proportion with no claims: "
                    f"**{(frequency == 0).mean() * 100:.2f}%**"
                )

            st.warning(
                "For prospective prediction, same-year claim count "
                "should not be used as a predictor of same-year claim "
                "cost unless it is genuinely available at prediction time."
            )

    # ------------------------------------------------------
    # 4. POLICY NON-RENEWAL ANALYSIS
    # ------------------------------------------------------

    st.divider()
    st.subheader("4. Policy non-renewal analysis")

    target_candidates = [
        "policy_nonrenewal",
        "Policy_nonrenewal",
        "nonrenewal",
        "Nonrenewal",
    ]

    target_col = next(
        (col for col in target_candidates if col in data.columns),
        None,
    )

    if target_col is None:
        st.info(
            "A constructed non-renewal target was not found. "
            "This section will become available when your main app "
            "passes a dataframe containing policy_nonrenewal."
        )
    else:
        target = pd.to_numeric(
            data[target_col], errors="coerce"
        )

        valid_target = target[target.isin([0, 1])]

        if valid_target.empty:
            st.warning(
                "The non-renewal target contains no valid 0/1 outcomes."
            )
        else:
            counts = valid_target.value_counts().reindex(
                [0, 1], fill_value=0
            )

            outcome_summary = pd.DataFrame(
                {
                    "Outcome": [
                        "Observed renewal",
                        "Not observed to renew",
                    ],
                    "Records": [
                        int(counts.loc[0]),
                        int(counts.loc[1]),
                    ],
                    "Percentage": [
                        (counts.loc[0] / len(valid_target)) * 100,
                        (counts.loc[1] / len(valid_target)) * 100,
                    ],
                }
            )

            left, right = st.columns(2)

            with left:
                st.write("**Outcome distribution**")

                fig, ax = plt.subplots()
                ax.bar(
                    outcome_summary["Outcome"],
                    outcome_summary["Records"],
                )
                ax.set_ylabel("Number of records")
                ax.set_title("Observed renewal outcomes")
                ax.tick_params(axis="x", rotation=10)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with right:
                st.write("**Outcome summary**")
                st.dataframe(
                    outcome_summary,
                    hide_index=True,
                    use_container_width=True,
                )

            st.warning(
                "The provisional outcome 'not observed to renew' "
                "does not necessarily mean a confirmed cancellation. "
                "Interpret it using the outcome-construction rules "
                "and the observation cutoff."
            )

    # ------------------------------------------------------
    # 5. COMPARISONS BY RISK GROUP
    # ------------------------------------------------------

    st.divider()
    st.subheader("5. Compare outcomes across risk groups")

    category_options = [
        col
        for col in [
            "Type_risk",
            "Area",
            "Distribution_channel",
            "Payment",
            "Second_driver",
            "Type_fuel",
        ]
        if col in data.columns
    ]

    if not category_options:
        st.info("No suitable grouping variables were found.")
    else:
        selected_category = st.selectbox(
            "Choose a grouping variable",
            options=category_options,
            key="exploration_group_variable",
        )

        group_data = data[[selected_category]].copy()

        if "Cost_claims_year" in data.columns:
            group_data["claim_cost"] = pd.to_numeric(
                data["Cost_claims_year"], errors="coerce"
            )

        if target_col is not None:
            group_data["nonrenewal"] = pd.to_numeric(
                data[target_col], errors="coerce"
            )

        group_data = group_data.dropna(
            subset=[selected_category]
        )

        aggregations = {}

        if "claim_cost" in group_data.columns:
            aggregations["claim_cost"] = ["mean", "median", "count"]

        if "nonrenewal" in group_data.columns:
            aggregations["nonrenewal"] = "mean"

        if aggregations:
            grouped = group_data.groupby(
                selected_category,
                observed=True,
            ).agg(aggregations)

            grouped.columns = [
                "_".join(
                    str(part) for part in col if str(part)
                ).strip("_")
                if isinstance(col, tuple)
                else str(col)
                for col in grouped.columns
            ]

            grouped = grouped.reset_index()

            st.write("**Group comparison table**")
            st.dataframe(
                grouped,
                hide_index=True,
                use_container_width=True,
            )

            if "claim_cost_mean" in grouped.columns:
                plot_data = grouped.dropna(
                    subset=["claim_cost_mean"]
                )

                if not plot_data.empty:
                    fig, ax = plt.subplots()
                    ax.bar(
                        plot_data[selected_category].astype(str),
                        plot_data["claim_cost_mean"],
                    )
                    ax.set_xlabel(selected_category)
                    ax.set_ylabel("Mean annual claim cost")
                    ax.set_title(
                        "Mean claim cost by " + selected_category
                    )
                    ax.tick_params(axis="x", rotation=45)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

            if "nonrenewal_mean" in grouped.columns:
                plot_data = grouped.dropna(
                    subset=["nonrenewal_mean"]
                )

                if not plot_data.empty:
                    fig, ax = plt.subplots()
                    ax.bar(
                        plot_data[selected_category].astype(str),
                        plot_data["nonrenewal_mean"] * 100,
                    )
                    ax.set_xlabel(selected_category)
                    ax.set_ylabel("Non-renewal outcome rate (%)")
                    ax.set_title(
                        "Non-renewal outcome rate by "
                        + selected_category
                    )
                    ax.tick_params(axis="x", rotation=45)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.caption(
                        "These are unadjusted group comparisons, "
                        "not estimates of causal effects."
                    )

    # ------------------------------------------------------
    # 6. NUMERICAL CORRELATIONS
    # ------------------------------------------------------

    st.divider()
    st.subheader("6. Relationships between numerical variables")

    excluded_columns = {
        "ID",
        "policy_nonrenewal",
        "Policy_nonrenewal",
        "nonrenewal",
        "Nonrenewal",
    }

    numeric_data = data.select_dtypes(
        include=[np.number]
    ).drop(
        columns=[
            col for col in excluded_columns
            if col in data.columns
        ],
        errors="ignore",
    )

    if numeric_data.shape[1] < 2:
        st.info(
            "At least two numerical variables are needed for "
            "a correlation analysis."
        )
    else:
        selected_numeric = st.multiselect(
            "Choose numerical variables",
            options=list(numeric_data.columns),
            default=list(numeric_data.columns[:min(8, numeric_data.shape[1])]),
            key="exploration_numeric_variables",
        )

        if len(selected_numeric) >= 2:
            correlation = numeric_data[selected_numeric].corr()

            fig, ax = plt.subplots(
                figsize=(
                    max(7, len(selected_numeric) * 0.8),
                    max(5, len(selected_numeric) * 0.7),
                )
            )

            image = ax.imshow(
                correlation,
                vmin=-1,
                vmax=1,
                aspect="auto",
            )

            ax.set_xticks(range(len(selected_numeric)))
            ax.set_yticks(range(len(selected_numeric)))

            ax.set_xticklabels(
                selected_numeric,
                rotation=45,
                ha="right",
            )
            ax.set_yticklabels(selected_numeric)

            ax.set_title("Pearson correlation matrix")
            fig.colorbar(image, ax=ax, label="Correlation")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.dataframe(
                correlation,
                use_container_width=True,
            )

            st.caption(
                "Correlation measures linear association. It does not "
                "establish causality, and numeric category codes may "
                "not have a meaningful linear interpretation."
            )
        else:
            st.info("Select at least two numerical variables.")

    # ------------------------------------------------------
    # 7. DOWNLOAD EXPLORATORY RESULTS
    # ------------------------------------------------------

    st.divider()
    st.subheader("7. Export data for research")

    summary_rows = []

    for col in data.columns:
        summary_rows.append(
            {
                "Variable": col,
                "Data_type": str(data[col].dtype),
                "Non_missing": int(data[col].notna().sum()),
                "Missing": int(data[col].isna().sum()),
                "Missing_percentage": (
                    data[col].isna().mean() * 100
                ),
                "Unique_values": int(data[col].nunique()),
            }
        )

    variable_summary = pd.DataFrame(summary_rows)

    st.write("**Variable-level data summary**")
    st.dataframe(
        variable_summary,
        hide_index=True,
        use_container_width=True,
    )

    csv_bytes = variable_summary.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download variable summary (CSV)",
        data=csv_bytes,
        file_name="motor_insurance_variable_summary.csv",
        mime="text/csv",
    )
