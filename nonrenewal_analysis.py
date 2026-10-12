
import io

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


TARGET_COL = "policy_nonrenewal"

DATE_CANDIDATES = [
    "Date_last_renewal",
    "Date_next_renewal",
]

CATEGORICAL_CANDIDATES = [
    "Type_risk",
    "Area",
    "Distribution_channel",
    "Payment",
    "Second_driver",
    "Type_fuel",
]


def _to_numeric_target(series):
    """Convert the target to numeric without treating missing values as zero."""
    return pd.to_numeric(series, errors="coerce")


def _find_date_column(df):
    """Choose the most useful available renewal-date column."""
    for column in DATE_CANDIDATES:
        if column in df.columns:
            dates = pd.to_datetime(df[column], errors="coerce")
            if dates.notna().any():
                return column, dates

    return None, None


def _rate_table(data, group_col, target_col=TARGET_COL, min_records=1):
    """Summarise non-renewal rates using only classified outcomes."""
    work = data[[group_col, target_col]].copy()
    work[target_col] = _to_numeric_target(work[target_col])
    work = work[work[target_col].isin([0, 1])]

    if work.empty:
        return pd.DataFrame()

    work[group_col] = work[group_col].fillna("Missing").astype(str)

    result = (
        work.groupby(group_col, dropna=False)[target_col]
        .agg(records="count", non_renewals="sum", rate="mean")
        .reset_index()
    )

    result = result[result["records"] >= min_records].copy()
    result["non_renewal_rate_pct"] = result["rate"] * 100

    return result.sort_values(
        "non_renewal_rate_pct", ascending=False
    )


def show_nonrenewal_analysis(target_df: pd.DataFrame) -> None:
    """Outcome-focused diagnostics for the provisional non-renewal target."""

    st.title("Policy Non-Renewal Analysis")

    st.write(
        "This module examines the constructed policy non-renewal outcome, "
        "its uncertainty, renewal-time patterns, and its relationship with "
        "candidate predictors. It does not independently redefine the target."
    )

    if target_df is None or target_df.empty:
        st.warning(
            "No target dataset is available. Visit the Outcome Construction "
            "tab first and check that target_df has been created."
        )
        return

    if TARGET_COL not in target_df.columns:
        st.error(
            f"The required target column '{TARGET_COL}' was not found. "
            "Check the Outcome Construction tab."
        )
        st.write("Columns currently available:", list(target_df.columns))
        return

    data = target_df.copy()
    data[TARGET_COL] = _to_numeric_target(data[TARGET_COL])

    # Only 0 and 1 are treated as classified outcomes.
    classified_mask = data[TARGET_COL].isin([0, 1])
    classified = data.loc[classified_mask].copy()
    unresolved = data.loc[~classified_mask].copy()

    total_records = len(data)
    n_classified = len(classified)
    n_unresolved = len(unresolved)
    n_renewed = int((classified[TARGET_COL] == 0).sum())
    n_nonrenewed = int((classified[TARGET_COL] == 1).sum())

    # ---------------------------------------------------------
    # 1. Outcome balance and unresolved records
    # ---------------------------------------------------------
    st.header("1. Outcome balance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total records", f"{total_records:,}")
    col2.metric("Classified outcomes", f"{n_classified:,}")
    col3.metric("Provisional non-renewals", f"{n_nonrenewed:,}")
    col4.metric("Unresolved outcomes", f"{n_unresolved:,}")

    if total_records:
        st.caption(
            f"Classified: {n_classified / total_records:.1%} of all records. "
            f"Unresolved: {n_unresolved / total_records:.1%} of all records."
        )

    if n_classified:
        renewed_pct = n_renewed / n_classified * 100
        nonrenewed_pct = n_nonrenewed / n_classified * 100

        balance = pd.DataFrame(
            {
                "Outcome": ["Observed renewal", "Provisional non-renewal"],
                "Records": [n_renewed, n_nonrenewed],
                "Percentage of classified records": [
                    renewed_pct,
                    nonrenewed_pct,
                ],
            }
        )

        st.dataframe(balance, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots()
        ax.bar(
            balance["Outcome"],
            balance["Records"],
        )
        ax.set_ylabel("Number of records")
        ax.set_title("Distribution of classified outcomes")
        ax.tick_params(axis="x", rotation=10)
        st.pyplot(fig)
        plt.close(fig)

        if n_nonrenewed == 0 or n_renewed == 0:
            st.warning(
                "Only one outcome class is present among classified records. "
                "A standard binary logistic regression cannot learn to "
                "distinguish both classes from this sample."
            )
        elif min(n_renewed, n_nonrenewed) / n_classified < 0.10:
            st.warning(
                "The outcome classes are imbalanced. Later modelling should "
                "report class-specific performance and avoid relying on "
                "accuracy alone."
            )
    else:
        st.warning(
            "There are no records with a classified 0/1 outcome. "
            "Investigate the target construction before modelling."
        )

    st.info(
        "Interpretation: a provisional non-renewal means the same ID was not "
        "observed renewing under the construction rule by the chosen cutoff. "
        "It does not automatically establish that the policyholder deliberately "
        "cancelled or switched insurer."
    )

    # ---------------------------------------------------------
    # 2. Investigate unresolved outcomes
    # ---------------------------------------------------------
    st.header("2. Unresolved-outcome diagnostics")

    st.write(
        "Unresolved records should not automatically be converted to 0 or 1. "
        "They may include records whose renewal date falls after the cutoff, "
        "incomplete dates, or records that cannot be matched confidently."
    )

    if n_unresolved:
        unresolved_summary = []

        for column in [
            "Date_last_renewal",
            "Date_next_renewal",
            "ID",
            "Lapse",
        ]:
            if column in unresolved.columns:
                unresolved_summary.append(
                    {
                        "Field": column,
                        "Missing among unresolved": int(
                            unresolved[column].isna().sum()
                        ),
                        "Percentage missing": (
                            unresolved[column].isna().mean() * 100
                        ),
                    }
                )

        if unresolved_summary:
            st.dataframe(
                pd.DataFrame(unresolved_summary),
                use_container_width=True,
                hide_index=True,
            )

        # Use a saved reason column if the target-construction code created one.
        reason_candidates = [
            "nonrenewal_reason",
            "policy_nonrenewal_reason",
            "_nonrenewal_reason",
            "outcome_reason",
        ]
        reason_col = next(
            (c for c in reason_candidates if c in unresolved.columns),
            None,
        )

        if reason_col:
            st.subheader("Saved reasons for unresolved outcomes")
            reason_counts = (
                unresolved[reason_col]
                .fillna("Reason not recorded")
                .astype(str)
                .value_counts()
                .rename_axis("Reason")
                .reset_index(name="Records")
            )
            st.dataframe(
                reason_counts,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption(
                "No explicit outcome-reason column was found. The table above "
                "summarises missing fields, but does not claim to identify the "
                "exact reason for every unresolved record."
            )

        preview_cols = [
            c for c in [
                "ID",
                "Date_last_renewal",
                "Date_next_renewal",
                "Lapse",
                TARGET_COL,
            ]
            if c in unresolved.columns
        ]

        with st.expander("Preview unresolved records"):
            st.dataframe(
                unresolved[preview_cols].head(200),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "The preview is limited to the first 200 unresolved records."
            )
    else:
        st.success("Every record has a classified 0/1 outcome.")

    # ---------------------------------------------------------
    # 3. Non-renewal patterns over time
    # ---------------------------------------------------------
    st.header("3. Non-renewal patterns over time")

    date_col, parsed_dates = _find_date_column(data)

    if date_col is None:
        st.warning(
            "No usable renewal-date column was found, so a time-based "
            "non-renewal analysis cannot be produced."
        )
    else:
        st.caption(f"Time grouping uses `{date_col}`.")

        time_data = data.loc[classified_mask].copy()
        time_data["_analysis_date"] = parsed_dates.loc[classified_mask]
        time_data = time_data.dropna(subset=["_analysis_date"])

        if not time_data.empty:
            time_data["Renewal year"] = (
                time_data["_analysis_date"].dt.year.astype(str)
            )

            yearly = (
                time_data.groupby("Renewal year")[TARGET_COL]
                .agg(records="count", non_renewals="sum", rate="mean")
                .reset_index()
            )
            yearly["non_renewal_rate_pct"] = yearly["rate"] * 100

            st.dataframe(
                yearly[
                    [
                        "Renewal year",
                        "records",
                        "non_renewals",
                        "non_renewal_rate_pct",
                    ]
                ].rename(
                    columns={
                        "records": "Classified records",
                        "non_renewals": "Provisional non-renewals",
                        "non_renewal_rate_pct": "Non-renewal rate (%)",
                    }
                ).round(2),
                use_container_width=True,
                hide_index=True,
            )

            fig, ax = plt.subplots()
            ax.plot(
                yearly["Renewal year"],
                yearly["non_renewal_rate_pct"],
                marker="o",
            )
            ax.set_xlabel("Renewal year")
            ax.set_ylabel("Provisional non-renewal rate (%)")
            ax.set_title("Non-renewal rate by renewal year")
            ax.tick_params(axis="x", rotation=30)
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Annual changes may reflect changing customer mix, premiums, "
                "business conditions, or observation-window limitations. "
                "They should not be interpreted as causal effects."
            )
        else:
            st.warning(
                "No classified records have usable renewal dates for the "
                "time-based analysis."
            )

    # ---------------------------------------------------------
    # 4. Non-renewal by categorical predictors
    # ---------------------------------------------------------
    st.header("4. Non-renewal rates across candidate predictors")

    st.write(
        "These are descriptive comparisons among classified records. "
        "They do not establish that a predictor causes non-renewal."
    )

    available_categories = [
        c for c in CATEGORICAL_CANDIDATES if c in classified.columns
    ]

    if not available_categories:
        st.warning(
            "None of the expected categorical predictor columns were found."
        )
    else:
        chosen_category = st.selectbox(
            "Choose a predictor",
            options=available_categories,
            key="nonrenewal_category",
        )

        min_group_size = st.slider(
            "Minimum classified records per group",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            key="nonrenewal_min_group",
        )

        rates = _rate_table(
            classified,
            chosen_category,
            min_records=min_group_size,
        )

        if rates.empty:
            st.warning(
                "No groups meet the selected minimum record count."
            )
        else:
            display_rates = rates[
                [
                    chosen_category,
                    "records",
                    "non_renewals",
                    "non_renewal_rate_pct",
                ]
            ].rename(
                columns={
                    "records": "Classified records",
                    "non_renewals": "Provisional non-renewals",
                    "non_renewal_rate_pct": "Non-renewal rate (%)",
                }
            )

            st.dataframe(
                display_rates.round(2),
                use_container_width=True,
                hide_index=True,
            )

            plot_rates = rates.sort_values("non_renewal_rate_pct")
            fig, ax = plt.subplots()
            ax.barh(
                plot_rates[chosen_category].astype(str),
                plot_rates["non_renewal_rate_pct"],
            )
            ax.set_xlabel("Provisional non-renewal rate (%)")
            ax.set_ylabel(chosen_category)
            ax.set_title(f"Non-renewal rate by {chosen_category}")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ---------------------------------------------------------
    # 5. Premium bands
    # ---------------------------------------------------------
    st.header("5. Non-renewal by premium band")

    if "Premium" not in classified.columns:
        st.info(
            "The Premium column is not available. Premium-band analysis "
            "cannot be calculated."
        )
    else:
        premium_data = classified.copy()
        premium_data["Premium"] = pd.to_numeric(
            premium_data["Premium"], errors="coerce"
        )
        premium_data = premium_data.dropna(subset=["Premium"])
        premium_data = premium_data[premium_data["Premium"] >= 0]

        if premium_data["Premium"].nunique() < 2:
            st.warning(
                "There are too few distinct non-negative premium values "
                "to create meaningful premium bands."
            )
        else:
            try:
                premium_data["Premium band"] = pd.qcut(
                    premium_data["Premium"],
                    q=4,
                    duplicates="drop",
                )

                premium_rates = (
                    premium_data.groupby(
                        "Premium band",
                        observed=True,
                    )[TARGET_COL]
                    .agg(
                        records="count",
                        non_renewals="sum",
                        rate="mean",
                    )
                    .reset_index()
                )
                premium_rates["non_renewal_rate_pct"] = (
                    premium_rates["rate"] * 100
                )

                st.dataframe(
                    premium_rates[
                        [
                            "Premium band",
                            "records",
                            "non_renewals",
                            "non_renewal_rate_pct",
                        ]
                    ].rename(
                        columns={
                            "records": "Classified records",
                            "non_renewals": "Provisional non-renewals",
                            "non_renewal_rate_pct": "Non-renewal rate (%)",
                        }
                    ).round(2),
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "Premium bands are descriptive quartile-based groups. "
                    "Premium may reflect underlying risk and coverage, so "
                    "this comparison is not a causal price-sensitivity estimate."
                )
            except ValueError:
                st.warning(
                    "Premium bands could not be created from the available "
                    "premium values."
                )

    # ---------------------------------------------------------
    # 6. Readiness for logistic regression
    # ---------------------------------------------------------
    st.header("6. Readiness for non-renewal modelling")

    readiness = []

    if n_classified == 0:
        readiness.append(
            ("Outcome availability", "Not ready", "No classified outcomes.")
        )
    else:
        readiness.append(
            (
                "Outcome availability",
                "Review" if n_unresolved > 0 else "Available",
                f"{n_classified:,} classified and {n_unresolved:,} unresolved.",
            )
        )

    if n_renewed > 0 and n_nonrenewed > 0:
        readiness.append(
            (
                "Both outcome classes",
                "Available",
                f"{n_renewed:,} zeros and {n_nonrenewed:,} ones.",
            )
        )
    else:
        readiness.append(
            (
                "Both outcome classes",
                "Not ready",
                "Both 0 and 1 outcomes are needed for standard binary logistic regression.",
            )
        )

    if "ID" in data.columns:
        repeated_id_share = (
            data["ID"].astype("string").duplicated(keep=False).mean() * 100
        )
        readiness.append(
            (
                "Repeated IDs",
                "Investigate",
                f"{repeated_id_share:.1f}% of rows have an ID that appears more than once. "
                "This alone does not establish customer identity.",
            )
        )

    readiness.append(
        (
            "Outcome interpretation",
            "Caution",
            "The target is provisional non-renewal by the cutoff, not verified customer cancellation.",
        )
    )

    st.dataframe(
        pd.DataFrame(
            readiness,
            columns=["Check", "Status", "Interpretation"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        **Before fitting the model, remember:**

        - Unresolved outcomes should not be automatically coded as renewals.
        - Candidate predictors must be available at the intended prediction time.
        - Avoid using post-outcome fields such as termination dates or same-year
          claim counts as predictors when they would not be known in advance.
        - Because the same ID may appear across multiple annual records, a later
          train/test split should respect time and repeated-record structure.
        - The target's meaning and cutoff should be documented in the research
          methodology.
        """
    )

    # ---------------------------------------------------------
    # 7. Export outcome review files
    # ---------------------------------------------------------
    st.header("7. Export outcome review data")

    export_cols = [
        c for c in [
            "ID",
            "Date_last_renewal",
            "Date_next_renewal",
            "Premium",
            "Type_risk",
            "Area",
            "Payment",
            "Lapse",
            TARGET_COL,
        ]
        if c in data.columns
    ]

    if export_cols:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            data[export_cols].to_excel(
                writer,
                index=False,
                sheet_name="Outcome review",
            )
            unresolved[export_cols].to_excel(
                writer,
                index=False,
                sheet_name="Unresolved records",
            )

        st.download_button(
            label="Download non-renewal review workbook",
            data=output.getvalue(),
            file_name="nonrenewal_outcome_review.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
        )

    st.caption(
        "This module is intended for outcome assessment and exploratory "
        "diagnostics. It does not fit the final logistic or joint shared-random-effects model."
    )
