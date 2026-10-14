import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# 1. DATA PREPARATION
# ============================================================

def construct_nonrenewal_outcome(df, cutoff_date):
    """
    Construct a provisional same-ID policy non-renewal outcome.

    Outcome definition:
        0 = A subsequent record with the same ID confirms renewal
            through an exact renewal-date match.

        1 = No subsequent record exists for the same ID, and the
            scheduled renewal date is on or before the cutoff.

        NA = The outcome cannot be classified under these rules.

    Important:
        Outcome 1 is provisional. Absence of a subsequent record
        does not prove that the policy genuinely was not renewed.
    """

    required_columns = {
        "ID",
        "Date_last_renewal",
        "Date_next_renewal",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    data = df.copy()

    # Preserve original row order for returning results.
    if "_row_order" not in data.columns:
        data["_row_order"] = np.arange(len(data))

    # Standardise identifier and date fields.
    data["ID"] = data["ID"].astype("string").str.strip()

    for col in ["Date_last_renewal", "Date_next_renewal"]:
        data[col] = pd.to_datetime(
            data[col],
            errors="coerce",
            dayfirst=True,
        )

    cutoff_ts = pd.Timestamp(cutoff_date)

    # Missing IDs must not be linked to each other.
    valid_id = (
        data["ID"].notna()
        & data["ID"].str.strip().ne("")
    )

    # Sort chronologically within each valid ID.
    ordered = data.sort_values(
        ["ID", "Date_last_renewal", "_row_order"],
        na_position="last",
    ).copy()

    # Detect a subsequent record independently of its date.
    next_row_order = (
        ordered.groupby("ID", dropna=True)["_row_order"].shift(-1)
    )

    ordered["_has_next_record"] = (
        next_row_order.notna()
        & ordered["ID"].notna()
        & ordered["ID"].str.strip().ne("")
    )

    # Get the next record's last-renewal date.
    ordered["_next_record_last_renewal"] = (
        ordered.groupby("ID", dropna=True)[
            "Date_last_renewal"
        ].shift(-1)
    )

    scheduled = ordered["Date_next_renewal"]
    next_dates = ordered["_next_record_last_renewal"]

    # Eligibility is determined by the scheduled renewal date.
    ordered["renewal_outcome_eligible"] = (
        scheduled.notna()
        & (scheduled <= cutoff_ts)
    )

    # A matching subsequent record confirms renewal under this
    # operational definition.
    ordered["observed_renewal_match"] = (
        ordered["_has_next_record"]
        & scheduled.notna()
        & next_dates.notna()
        & (scheduled == next_dates)
    )

    # A subsequent record exists but cannot confirm the expected
    # renewal. These records remain unresolved.
    ordered["renewal_date_mismatch_flag"] = (
        ordered["renewal_outcome_eligible"]
        & ordered["_has_next_record"]
        & ~ordered["observed_renewal_match"]
    )

    # Initialise all outcomes as unresolved.
    ordered["policy_nonrenewal"] = pd.Series(
        pd.NA,
        index=ordered.index,
        dtype="Int64",
    )

    reliable_current_record = (
        ordered["ID"].notna()
        & ordered["ID"].str.strip().ne("")
        & ordered["Date_last_renewal"].notna()
        & scheduled.notna()
    )

    eligible = ordered["renewal_outcome_eligible"]

    # 0 = observed renewal.
    confirmed_renewal = (
        eligible
        & reliable_current_record
        & ordered["observed_renewal_match"]
    )

    ordered.loc[
        confirmed_renewal,
        "policy_nonrenewal",
    ] = 0

    # 1 = provisional non-renewal.
    # No later same-ID record is observed for an eligible record.
    provisional_nonrenewal = (
        eligible
        & reliable_current_record
        & ~ordered["_has_next_record"]
    )

    ordered.loc[
        provisional_nonrenewal,
        "policy_nonrenewal",
    ] = 1

    # Return records in the original upload order.
    result = ordered.sort_values("_row_order").copy()

    # Explicit audit categories.
    result["outcome_status"] = "Unresolved"

    result.loc[
        ~result["renewal_outcome_eligible"],
        "outcome_status",
    ] = "Outside cutoff / not eligible"

    result.loc[
        result["policy_nonrenewal"].eq(0).fillna(False),
        "outcome_status",
    ] = "Observed renewal"

    result.loc[
        result["policy_nonrenewal"].eq(1).fillna(False),
        "outcome_status",
    ] = "Provisional non-renewal"

    missing_id = (
        result["ID"].isna()
        | result["ID"].str.strip().eq("")
    )

    missing_last_date = result["Date_last_renewal"].isna()
    missing_next_date = result["Date_next_renewal"].isna()

    result.loc[
        missing_id,
        "outcome_status",
    ] = "Missing/invalid ID"

    result.loc[
        ~missing_id & missing_next_date,
        "outcome_status",
    ] = "Missing scheduled renewal date"

    result.loc[
        ~missing_id
        & ~missing_last_date
        & result["renewal_date_mismatch_flag"],
        "outcome_status",
    ] = "Subsequent record with unmatched date"

    result.loc[
        ~missing_id
        & missing_last_date
        & ~missing_next_date,
        "outcome_status",
    ] = "Missing last-renewal date"

    return result


# ============================================================
# 2. DIAGNOSTIC SUMMARY
# ============================================================

def build_outcome_diagnostics(target_df, cutoff_date):
    """
    Produce explicit outcome counts and data-quality diagnostics.
    """

    data = target_df.copy()
    cutoff_ts = pd.Timestamp(cutoff_date)

    total = len(data)

    valid_id = (
        data["ID"].notna()
        & data["ID"].astype("string").str.strip().ne("")
    )

    eligible = data["renewal_outcome_eligible"].fillna(False)

    renewed = data["policy_nonrenewal"].eq(0).fillna(False)
    nonrenewed = data["policy_nonrenewal"].eq(1).fillna(False)

    unresolved_eligible = (
        eligible
        & data["policy_nonrenewal"].isna()
    )

    after_cutoff = (
        data["Date_next_renewal"].notna()
        & (data["Date_next_renewal"] > cutoff_ts)
    )

    measures = [
        ("Total records", total),
        ("Valid IDs", int(valid_id.sum())),
        ("Missing or invalid IDs", int((~valid_id).sum())),
        (
            "Valid scheduled renewal dates",
            int(data["Date_next_renewal"].notna().sum()),
        ),
        (
            "Missing scheduled renewal dates",
            int(data["Date_next_renewal"].isna().sum()),
        ),
        ("Eligible by cutoff", int(eligible.sum())),
        ("Observed renewals", int(renewed.sum())),
        ("Provisional non-renewals", int(nonrenewed.sum())),
        ("Eligible but unresolved", int(unresolved_eligible.sum())),
        ("Scheduled after cutoff", int(after_cutoff.sum())),
        (
            "Subsequent same-ID records",
            int(data["_has_next_record"].sum()),
        ),
        (
            "Exact renewal-date matches",
            int(data["observed_renewal_match"].sum()),
        ),
        (
            "Eligible date mismatches",
            int(data["renewal_date_mismatch_flag"].sum()),
        ),
    ]

    summary = pd.DataFrame(
        measures,
        columns=["Measure", "Count"],
    )

    return summary


# ============================================================
# 3. STREAMLIT USER INTERFACE
# ============================================================

def show_outcome_construction(df, cutoff_date):
    """
    Display claim-cost checks, outcome construction, diagnostics,
    and download options.
    """

    st.header("Outcome Construction and Validation")

    st.write(
        "This section constructs a provisional same-ID non-renewal "
        "indicator and reports the limitations of the classification."
    )

    # --------------------------------------------------------
    # Claim-cost outcome
    # --------------------------------------------------------

    st.subheader("A. Annual claim-cost outcome")

    if "Cost_claims_year" in df.columns:

        costs = pd.to_numeric(
            df["Cost_claims_year"],
            errors="coerce",
        )

        valid_costs = costs.notna()
        zero_costs = costs.eq(0)
        positive_costs = costs.gt(0)
        negative_costs = costs.lt(0)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Valid cost records",
            f"{int(valid_costs.sum()):,}",
        )

        c2.metric(
            "Zero costs",
            f"{int(zero_costs.sum()):,}",
        )

        c3.metric(
            "Positive costs",
            f"{int(positive_costs.sum()):,}",
        )

        c4.metric(
            "Negative costs",
            f"{int(negative_costs.sum()):,}",
        )

        if negative_costs.any():
            st.warning(
                "Negative annual claim costs exist. Investigate "
                "adjustments or recoveries before fitting a Tweedie model."
            )

        if valid_costs.any():

            fig, ax = plt.subplots()

            costs.loc[valid_costs].hist(
                bins=50,
                ax=ax,
            )

            ax.set_title("Annual claim-cost distribution")
            ax.set_xlabel("Cost_claims_year")
            ax.set_ylabel("Frequency")

            st.pyplot(fig)
            plt.close(fig)

    else:
        st.warning("Cost_claims_year was not found.")

    st.divider()

    # --------------------------------------------------------
    # Construct outcome
    # --------------------------------------------------------

    st.subheader("B. Construct the provisional non-renewal outcome")

    try:
        target_df = construct_nonrenewal_outcome(
            df,
            cutoff_date,
        )

    except ValueError as exc:
        st.error(str(exc))
        return None

    st.session_state["target_df"] = target_df

    # --------------------------------------------------------
    # Headline counts
    # --------------------------------------------------------

    eligible = target_df["renewal_outcome_eligible"].fillna(False)

    renewed = target_df["policy_nonrenewal"].eq(0).fillna(False)

    nonrenewed = target_df["policy_nonrenewal"].eq(1).fillna(False)

    unresolved_eligible = (
        eligible
        & target_df["policy_nonrenewal"].isna()
    )

    after_cutoff = (
        target_df["Date_next_renewal"].notna()
        & (
            target_df["Date_next_renewal"]
            > pd.Timestamp(cutoff_date)
        )
    )

    st.subheader("C. Outcome counts")

    a, b, c = st.columns(3)

    a.metric(
        "Observed renewals",
        f"{int(renewed.sum()):,}",
    )

    b.metric(
        "Provisional non-renewals",
        f"{int(nonrenewed.sum()):,}",
    )

    c.metric(
        "Eligible but unresolved",
        f"{int(unresolved_eligible.sum()):,}",
    )

    d, e, f = st.columns(3)

    d.metric(
        "Total records",
        f"{len(target_df):,}",
    )

    e.metric(
        "Eligible by cutoff",
        f"{int(eligible.sum()):,}",
    )

    f.metric(
        "Scheduled after cutoff",
        f"{int(after_cutoff.sum()):,}",
    )

    classified_count = int((renewed | nonrenewed).sum())

    st.write(
        f"**Classified outcomes:** {classified_count:,} of "
        f"{len(target_df):,} records "
        f"({100 * classified_count / len(target_df):.1f}%)."
        if len(target_df)
        else "No records available."
    )

    st.write(
        f"**Eligible but unresolved:** {int(unresolved_eligible.sum()):,}."
    )

    st.caption(
        "Records scheduled after the cutoff are not classified. "
        "Provisional non-renewals are based on the absence of a "
        "subsequent same-ID record and are not confirmed cancellations."
    )

    # --------------------------------------------------------
    # Detailed status table
    # --------------------------------------------------------

    st.subheader("D. Detailed outcome-status breakdown")

    status_counts = (
        target_df["outcome_status"]
        .value_counts(dropna=False)
        .rename_axis("Outcome status")
        .reset_index(name="Number of records")
    )

    status_counts["Percentage of all records"] = (
        100 * status_counts["Number of records"] / len(target_df)
        if len(target_df)
        else 0
    )

    status_counts["Percentage of all records"] = (
        status_counts["Percentage of all records"].round(2)
    )

    st.dataframe(
        status_counts,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Diagnostic table
    # --------------------------------------------------------

    st.subheader("E. Data-quality diagnostic")

    diagnostics = build_outcome_diagnostics(
        target_df,
        cutoff_date,
    )

    st.dataframe(
        diagnostics,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Lapse diagnostic
    # --------------------------------------------------------

    if "Lapse" in target_df.columns:

        st.subheader("F. Comparison with the source Lapse variable")

        st.caption(
            "This is a diagnostic comparison only. Lapse is not used "
            "to construct the binary target."
        )

        labelled = target_df["policy_nonrenewal"].notna()

        if labelled.any():

            comparison = pd.crosstab(
                target_df.loc[
                    labelled,
                    "policy_nonrenewal",
                ].astype(int),
                target_df.loc[labelled, "Lapse"],
                margins=True,
            )

            st.dataframe(
                comparison,
                use_container_width=True,
            )

        else:
            st.info(
                "There are no classified observations available "
                "for the Lapse comparison."
            )

    # --------------------------------------------------------
    # Review and export
    # --------------------------------------------------------

    st.subheader("G. Review unresolved observations")

    unresolved_rows = target_df.loc[
        target_df["policy_nonrenewal"].isna()
    ].copy()

    st.write(
        f"Number of rows without a binary outcome: "
        f"{len(unresolved_rows):,}"
    )

    if not unresolved_rows.empty:

        st.dataframe(
            unresolved_rows[
                [
                    col for col in [
                        "ID",
                        "Date_last_renewal",
                        "Date_next_renewal",
                        "_next_record_last_renewal",
                        "_has_next_record",
                        "renewal_outcome_eligible",
                        "renewal_date_mismatch_flag",
                        "outcome_status",
                    ]
                    if col in unresolved_rows.columns
                ]
            ].head(100),
            use_container_width=True,
            hide_index=True,
        )

    st.download_button(
        "Download complete outcome dataset",
        data=target_df.to_csv(index=False).encode("utf-8"),
        file_name="motor_insurance_outcome_audit.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download outcome diagnostics",
        data=diagnostics.to_csv(index=False).encode("utf-8"),
        file_name="outcome_diagnostics.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download outcome-status counts",
        data=status_counts.to_csv(index=False).encode("utf-8"),
        file_name="outcome_status_counts.csv",
        mime="text/csv",
    )

    st.subheader("H. Research interpretation")

    st.info(
        """
        The constructed target is a provisional same-ID non-renewal indicator.

        A matching subsequent renewal record is classified as an observed
        renewal. The absence of a subsequent record by the cutoff is
        classified as provisional non-renewal only when the required
        identifier and dates are available.

        An absent subsequent record does not prove that the policyholder
        genuinely did not renew. Right-censoring, missing records and
        incomplete follow-up must be considered before fitting a model.

        The ID must also be interpreted according to its documented
        meaning. Repeated policy-product records do not automatically
        identify the same individual across all products.
        """
    )

    return target_df
