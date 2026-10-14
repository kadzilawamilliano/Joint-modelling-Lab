import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


def show_repeated_observation_analysis(df, target_df=None):
    """
    Examine repeated policy identifiers and whether their records
    provide a defensible structure for longitudinal modelling.

    This module does not assume that an ID represents a unique person.
    """

    st.subheader("Repeated Observation and ID Structure")
    st.write(
        """
        This analysis examines how frequently policy identifiers appear,
        whether their records span multiple dates, and whether repeated
        observations have usable renewal outcomes.

        The results help assess the data structure required for the
        proposed joint claim-cost and non-renewal model.
        """
    )

    required = {"ID"}

    if not required.issubset(df.columns):
        st.error("The dataset must contain an ID column.")
        return

    data = df.copy()

    # Remove missing and blank identifiers from grouping analysis.
    data["ID"] = data["ID"].astype("string").str.strip()

    valid_id = data["ID"].notna() & data["ID"].ne("")
    valid_data = data.loc[valid_id].copy()

    if valid_data.empty:
        st.error("No valid policy identifiers were found.")
        return

    # ---------------------------------------------------------
    # 1. Basic identifier summary
    # ---------------------------------------------------------
    st.markdown("### 1. Identifier summary")

    total_rows = len(data)
    valid_rows = int(valid_id.sum())
    missing_id_rows = total_rows - valid_rows

    id_counts = valid_data.groupby("ID", dropna=True).size()

    unique_ids = int(id_counts.size)
    singleton_ids = int((id_counts == 1).sum())
    repeated_ids = int((id_counts >= 2).sum())
    ids_three_plus = int((id_counts >= 3).sum())

    rows_in_repeated_groups = int(
        id_counts[id_counts >= 2].sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Dataset rows", f"{total_rows:,}")
    c2.metric("Valid IDs", f"{unique_ids:,}")
    c3.metric("IDs appearing once", f"{singleton_ids:,}")
    c4.metric("IDs appearing 2+ times", f"{repeated_ids:,}")

    c5, c6, c7 = st.columns(3)

    c5.metric("IDs appearing 3+ times", f"{ids_three_plus:,}")
    c6.metric(
        "Rows in repeated-ID groups",
        f"{rows_in_repeated_groups:,}",
    )
    c7.metric("Rows with missing/blank ID", f"{missing_id_rows:,}")

    st.caption(
        "An identifier appearing repeatedly establishes repeated records "
        "under that identifier, but does not by itself prove that the "
        "records represent one individual policyholder."
    )

    # ---------------------------------------------------------
    # 2. Distribution of records per ID
    # ---------------------------------------------------------
    st.markdown("### 2. Distribution of records per identifier")

    distribution = (
        id_counts.value_counts()
        .sort_index()
        .rename_axis("Records per ID")
        .reset_index(name="Number of IDs")
    )

    distribution["Percentage of IDs (%)"] = (
        100 * distribution["Number of IDs"] / unique_ids
    ).round(2)

    st.dataframe(
        distribution,
        use_container_width=True,
        hide_index=True,
    )

    fig, ax = plt.subplots()

    plot_distribution = distribution.copy()
    max_categories = 15

    if len(plot_distribution) > max_categories:
        first = plot_distribution.iloc[:max_categories - 1].copy()

        other = pd.DataFrame({
            "Records per ID": [
                f">{max_categories - 1}"
            ],
            "Number of IDs": [
                int(
                    plot_distribution.iloc[
                        max_categories - 1:
                    ]["Number of IDs"].sum()
                )
            ],
            "Percentage of IDs (%)": [
                round(
                    100
                    * plot_distribution.iloc[
                        max_categories - 1:
                    ]["Number of IDs"].sum()
                    / unique_ids,
                    2,
                )
            ],
        })

        plot_distribution = pd.concat(
            [first, other],
            ignore_index=True,
        )

    ax.bar(
        plot_distribution["Records per ID"].astype(str),
        plot_distribution["Number of IDs"],
    )
    ax.set_title("Number of records per policy identifier")
    ax.set_xlabel("Records per ID")
    ax.set_ylabel("Number of IDs")
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "Download ID-frequency distribution",
        data=distribution.to_csv(index=False).encode("utf-8"),
        file_name="id_frequency_distribution.csv",
        mime="text/csv",
    )

    # ---------------------------------------------------------
    # 3. Repeated observations over time
    # ---------------------------------------------------------
    st.markdown("### 3. Repeated observations over time")

    date_columns = [
        col for col in [
            "Date_start_contract",
            "Date_last_renewal",
            "Date_next_renewal",
        ]
        if col in valid_data.columns
    ]

    if "Date_last_renewal" in valid_data.columns:
        valid_data["Date_last_renewal"] = pd.to_datetime(
            valid_data["Date_last_renewal"],
            errors="coerce",
        )

        date_counts = (
            valid_data.dropna(subset=["Date_last_renewal"])
            .groupby("ID")["Date_last_renewal"]
            .nunique()
        )

        ids_with_multiple_dates = int(
            (date_counts >= 2).sum()
        )

        st.metric(
            "IDs with 2+ distinct last-renewal dates",
            f"{ids_with_multiple_dates:,}",
        )

        st.caption(
            "Distinct renewal dates are evidence of observations at "
            "different times. They do not guarantee that every interval "
            "is complete or that each ID refers to a person."
        )
    else:
        st.warning(
            "`Date_last_renewal` is unavailable, so the temporal "
            "sequence cannot be examined."
        )

    # ---------------------------------------------------------
    # 4. Check date sequences within each ID
    # ---------------------------------------------------------
    st.markdown("### 4. Renewal-date sequence checks")

    sequence_summary = None

    if {
        "Date_last_renewal",
        "Date_next_renewal",
    }.issubset(valid_data.columns):

        sequence_data = valid_data.copy()

        for col in [
            "Date_last_renewal",
            "Date_next_renewal",
        ]:
            sequence_data[col] = pd.to_datetime(
                sequence_data[col],
                errors="coerce",
            )

        sequence_data = sequence_data.sort_values(
            ["ID", "Date_last_renewal"]
        )

        sequence_data["Next_record_last_renewal"] = (
            sequence_data.groupby("ID")[
                "Date_last_renewal"
            ].shift(-1)
        )

        sequence_data["Date_sequence_difference_days"] = (
            sequence_data["Next_record_last_renewal"]
            - sequence_data["Date_next_renewal"]
        ).dt.days

        has_comparison = (
            sequence_data["Date_next_renewal"].notna()
            & sequence_data["Next_record_last_renewal"].notna()
        )

        matches = (
            has_comparison
            & (
                sequence_data["Date_next_renewal"]
                == sequence_data["Next_record_last_renewal"]
            )
        )

        mismatches = has_comparison & ~matches

        n_comparable = int(has_comparison.sum())
        n_matches = int(matches.sum())
        n_mismatches = int(mismatches.sum())

        match_rate = (
            100 * n_matches / n_comparable
            if n_comparable > 0
            else np.nan
        )

        sequence_summary = pd.DataFrame({
            "Measure": [
                "Consecutive records with comparable dates",
                "Exact renewal-date matches",
                "Date mismatches",
                "Exact match percentage",
            ],
            "Value": [
                n_comparable,
                n_matches,
                n_mismatches,
                round(match_rate, 2)
                if not np.isnan(match_rate)
                else np.nan,
            ],
        })

        st.dataframe(
            sequence_summary,
            use_container_width=True,
            hide_index=True,
        )

        if n_mismatches > 0:
            st.warning(
                "Some consecutive records do not match the previous "
                "record's scheduled renewal date. Investigate these "
                "cases before using the provisional outcome."
            )

            mismatch_columns = [
                "ID",
                "Date_last_renewal",
                "Date_next_renewal",
                "Next_record_last_renewal",
                "Date_sequence_difference_days",
            ]

            mismatch_records = sequence_data.loc[
                mismatches,
                mismatch_columns,
            ]

            st.dataframe(
                mismatch_records.head(100),
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download renewal-date mismatches",
                data=mismatch_records.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="renewal_date_mismatches.csv",
                mime="text/csv",
            )

    else:
        st.info(
            "Both Date_last_renewal and Date_next_renewal are needed "
            "for this sequence check."
        )

    # ---------------------------------------------------------
    # 5. Outcome availability in repeated groups
    # ---------------------------------------------------------
    st.markdown("### 5. Outcome availability in repeated groups")

    if target_df is not None and len(target_df) == len(data):

        # Align by original row position, not by ID.
        # target_df is expected to preserve the source row order.
        outcome_data = valid_data.copy()

        target_outcome = target_df[
            "policy_nonrenewal"
        ].reset_index(drop=True)

        # Map the original row positions back to the matching target.
        if "_row_order" in data.columns:
            positions = data.loc[
                valid_id, "_row_order"
            ].astype(int).to_numpy()
        else:
            positions = np.flatnonzero(
                valid_id.to_numpy()
            )

        outcome_data["policy_nonrenewal"] = (
            target_outcome.iloc[positions].to_numpy()
        )

        outcome_data["_labelled"] = (
            outcome_data["policy_nonrenewal"].notna()
        )

        outcome_data["_renewed"] = (
            outcome_data["policy_nonrenewal"] == 0
        )

        outcome_data["_nonrenewed"] = (
            outcome_data["policy_nonrenewal"] == 1
        )

        outcome_by_id = outcome_data.groupby("ID").agg(
            records=("ID", "size"),
            labelled_records=("_labelled", "sum"),
            observed_renewals=("_renewed", "sum"),
            observed_nonrenewals=("_nonrenewed", "sum"),
        )

        outcome_by_id["has_repeated_records"] = (
            outcome_by_id["records"] >= 2
        )

        outcome_by_id["has_two_or_more_labelled_records"] = (
            outcome_by_id["labelled_records"] >= 2
        )

        repeated_groups = outcome_by_id[
            outcome_by_id["has_repeated_records"]
        ]

        outcome_metrics = pd.DataFrame({
            "Measure": [
                "IDs with repeated records",
                "Repeated IDs with 2+ labelled records",
                "Repeated IDs with no labelled records",
            ],
            "Value": [
                len(repeated_groups),
                int(
                    repeated_groups[
                        "has_two_or_more_labelled_records"
                    ].sum()
                ),
                int(
                    (
                        repeated_groups["labelled_records"] == 0
                    ).sum()
                ),
            ],
        })

        st.dataframe(
            outcome_metrics,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "A labelled record has a provisional outcome of 0 or 1. "
            "This check describes outcome availability; it does not "
            "confirm that the provisional labels are correct."
        )

        st.download_button(
            "Download per-ID outcome summary",
            data=outcome_by_id.reset_index().to_csv(
                index=False
            ).encode("utf-8"),
            file_name="per_id_outcome_summary.csv",
            mime="text/csv",
        )

    else:
        st.info(
            "Construct the provisional outcome first. The outcome "
            "summary will appear when a target dataframe with the same "
            "number of rows is available."
        )

    # ---------------------------------------------------------
    # 6. Research interpretation
    # ---------------------------------------------------------
    st.markdown("### 6. Implications for the proposed joint model")

    st.write(
        f"""
        The uploaded dataset contains **{unique_ids:,} valid policy
        identifiers**. Of these, **{repeated_ids:,}** appear in at least
        two records, while **{singleton_ids:,}** appear only once.

        The repeated records are potentially useful for investigating
        longitudinal structure. However, the identifier must be
        interpreted according to its documented meaning: repeated
        annual policy-product records do not automatically establish
        repeated observations for the same individual customer.
        """
    )

    st.warning(
        """
        These diagnostics do not, by themselves, establish that the
        shared-random-effects model is identifiable or statistically
        suitable. The definition of the grouping unit, the availability
        of both outcomes, the observation window, and the validity of
        the non-renewal labels must be assessed before fitting the model.
        """
    )
