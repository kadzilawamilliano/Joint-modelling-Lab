
import io
import re

import numpy as np
import pandas as pd
import streamlit as st


TARGET = "policy_nonrenewal"
CLAIM_TARGET = "Cost_claims_year"

# These fields are not ordinary predictors for the reasons shown below.
SPECIAL_RULES = {
    "ID": (
        "Identifier",
        "Do not use as an ordinary predictor. Also, do not assume it "
        "identifies a unique person."
    ),
    "policy_nonrenewal": (
        "Non-renewal outcome",
        "Target variable; never use as its own predictor."
    ),
    "Cost_claims_year": (
        "Claim-cost outcome",
        "Target for the claim-cost model; exclude as a predictor."
    ),
    "N_claims_year": (
        "Potential target leakage",
        "Same-year claim count may not be available at prediction time."
    ),
    "Lapse": (
        "Diagnostic only",
        "Do not use as a non-renewal predictor because it may reflect "
        "the outcome or a related termination event."
    ),
    "Date_lapse": (
        "Potential target leakage",
        "Termination date may reveal information about the outcome."
    ),
    "Value_vehicle": (
        "Timing concern",
        "The dataset documentation describes this as market value at "
        "31 December 2019. Check whether it was known at the prediction date."
    ),
}


def _missing_pct(series):
    return round(series.isna().mean() * 100, 2) if len(series) else 0.0


def _variable_type(series):
    """Give a practical screening type, not a definitive data dictionary."""
    name = str(series.name).lower()

    if pd.api.types.is_datetime64_any_dtype(series):
        return "Date"

    if name.startswith("date_") or name.endswith("_date"):
        return "Date-like (check parsing)"

    if pd.api.types.is_bool_dtype(series):
        return "Binary"

    non_missing = series.dropna()

    if non_missing.empty:
        return "Empty"

    if pd.api.types.is_numeric_dtype(series):
        unique_count = non_missing.nunique()

        if unique_count == 2:
            return "Binary numeric"

        if unique_count <= 10:
            return "Low-cardinality numeric"

        return "Continuous numeric"

    unique_count = non_missing.nunique()

    if unique_count <= 10:
        return "Categorical"

    return "Text / high-cardinality"


def _screen_variable(name, series):
    """Produce cautious, name-based screening recommendations."""
    type_label = _variable_type(series)
    missing = _missing_pct(series)
    unique_count = int(series.nunique(dropna=True))
    lower = str(name).lower()

    role = "Candidate predictor"
    claim_cost = "Review"
    nonrenewal = "Review"
    caution = "Confirm meaning and when this variable becomes available."

    if name in SPECIAL_RULES:
        role, caution = SPECIAL_RULES[name]

    if name == CLAIM_TARGET:
        claim_cost = "Outcome — exclude"
        nonrenewal = "Exclude"
    elif name == TARGET:
        claim_cost = "Exclude"
        nonrenewal = "Outcome — exclude"
    elif name == "ID":
        claim_cost = "Exclude"
        nonrenewal = "Exclude"
    elif name in {"Lapse", "Date_lapse"}:
        claim_cost = "Usually exclude"
        nonrenewal = "Exclude"
    elif name == "N_claims_year":
        claim_cost = "Exclude unless available beforehand"
        nonrenewal = "Exclude unless available beforehand"
    elif name == "Value_vehicle":
        claim_cost = "Check timing first"
        nonrenewal = "Check timing first"
    else:
        # Dates need to be converted into meaningful historical features.
        if type_label.startswith("Date") or lower.startswith("date_"):
            role = "Date / feature engineering"
            claim_cost = "Use derived features if known beforehand"
            nonrenewal = "Use derived features if known beforehand"
            caution = (
                "Check the date's timing; derive age, tenure, or calendar "
                "features only when available at the prediction date."
            )

        # Other variables that often describe claims over a period.
        if any(term in lower for term in [
            "claims_year", "claims_history", "claim_history",
            "date_lapse", "lapse"
        ]):
            role = "Claims / termination history"
            claim_cost = "Check observation timing"
            nonrenewal = "Check observation timing"
            caution = (
                "May describe the outcome period or a later period. "
                "Use only information known at the prediction date."
            )

        # Potential identifiers and near-identifiers.
        if lower in {"id", "policy_id", "customer_id", "client_id"}:
            role = "Identifier"
            claim_cost = "Exclude"
            nonrenewal = "Exclude"
            caution = (
                "Use for grouping or record linkage only after validating "
                "what the identifier represents."
            )

        # Row-level fields whose timing requires particular care.
        if lower in {"premium", "seniority", "policies_in_force",
                     "max_policies", "max_products"}:
            caution = (
                "Potentially useful, but verify that its value is known "
                "at the start of the prediction period."
            )

    if missing >= 50:
        caution += f" High missingness ({missing:.1f}%)."
    elif missing > 0:
        caution += f" Missing values: {missing:.1f}%."

    if unique_count <= 1:
        role = "Constant / unusable"
        claim_cost = "Exclude"
        nonrenewal = "Exclude"
        caution = "Only one observed non-missing value; cannot distinguish records."

    return {
        "Variable": str(name),
        "Data type": type_label,
        "Missing (%)": missing,
        "Unique values": unique_count,
        "Screening role": role,
        "Claim-cost model": claim_cost,
        "Non-renewal model": nonrenewal,
        "Notes": caution,
    }


def build_predictor_screening_table(df, target_df=None):
    """
    Screen all variables from the original dataframe.
    If target_df contains the constructed target, attach it by row position
    when the row counts match.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    data = df.copy()

    if (
        target_df is not None
        and isinstance(target_df, pd.DataFrame)
        and TARGET in target_df.columns
        and len(target_df) == len(data)
    ):
        # Row-position alignment is intentional: target construction preserves
        # the original row order. Do not merge by ID, which may repeat.
        data[TARGET] = target_df[TARGET].to_numpy()

    rows = [
        _screen_variable(column, data[column])
        for column in data.columns
    ]

    return pd.DataFrame(rows)


def _outcome_summary(data):
    """Summarise the constructed outcome if it is available."""
    if TARGET not in data.columns:
        return None

    outcome = pd.to_numeric(data[TARGET], errors="coerce")
    classified = outcome.isin([0, 1])

    if not classified.any():
        return None

    counts = outcome[classified].value_counts().reindex([0, 1], fill_value=0)
    total = int(counts.sum())

    return pd.DataFrame({
        "Outcome": ["Renewal (0)", "Non-renewal (1)"],
        "Records": [int(counts.loc[0]), int(counts.loc[1])],
        "Percentage of classified records": [
            round(counts.loc[0] / total * 100, 2),
            round(counts.loc[1] / total * 100, 2),
        ],
    })


def show_predictor_analysis(df, target_df=None):
    """Render predictor screening in the Streamlit app."""
    st.header("Predictor screening")

    st.write(
        "This section screens the available variables before model fitting. "
        "The recommendations are preliminary: variable definitions and "
        "availability at the prediction date must be verified."
    )

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.warning("Upload and prepare the original dataset first.")
        return

    screening = build_predictor_screening_table(df, target_df)

    if screening.empty:
        st.warning("No variables are available to screen.")
        return

    # Reconstruct the displayed data consistently with the screening table.
    data = df.copy()
    if (
        target_df is not None
        and isinstance(target_df, pd.DataFrame)
        and TARGET in target_df.columns
        and len(target_df) == len(data)
    ):
        data[TARGET] = target_df[TARGET].to_numpy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Variables screened", len(screening))
    c2.metric(
        "Variables with missing values",
        int((screening["Missing (%)"] > 0).sum())
    )
    c3.metric(
        "Variables flagged for review",
        int(screening["Screening role"].isin([
            "Potential target leakage",
            "Timing concern",
            "Claims / termination history",
            "Date / feature engineering",
            "Identifier",
        ]).sum())
    )

    st.subheader("Full screening table")

    search = st.text_input(
        "Search variable names or screening notes",
        placeholder="e.g. Premium, date, claims, leakage"
    )

    view = screening.copy()
    if search.strip():
        mask = view.astype(str).apply(
            lambda column: column.str.contains(
                re.escape(search.strip()), case=False, na=False
            )
        ).any(axis=1)
        view = view.loc[mask]

    st.dataframe(view, use_container_width=True, hide_index=True)

    csv_bytes = screening.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download screening table (CSV)",
        data=csv_bytes,
        file_name="predictor_screening.csv",
        mime="text/csv",
    )

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        screening.to_excel(
            writer, sheet_name="Predictor screening", index=False
        )

        if TARGET in data.columns:
            summary = _outcome_summary(data)
            if summary is not None:
                summary.to_excel(
                    writer, sheet_name="Outcome summary", index=False
                )

    st.download_button(
        "Download screening workbook (Excel)",
        data=excel_buffer.getvalue(),
        file_name="predictor_screening.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
    )

    st.subheader("Outcome availability")

    if TARGET in data.columns:
        outcome = pd.to_numeric(data[TARGET], errors="coerce")
        n_classified = int(outcome.isin([0, 1]).sum())
        n_unresolved = int(outcome.isna().sum())

        left, right = st.columns(2)
        left.metric("Classified outcomes", f"{n_classified:,}")
        right.metric("Unresolved outcomes", f"{n_unresolved:,}")

        summary = _outcome_summary(data)
        if summary is not None:
            st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        st.info(
            "The constructed non-renewal outcome is not attached yet. "
            "You can still screen the original variables; outcome-specific "
            "summaries will appear once policy_nonrenewal is available."
        )

    st.subheader("How to interpret the screening")

    st.markdown(
        """
        - **Candidate predictor** means a variable worth investigating, not
          automatic approval for the final model.
        - **Exclude** means it should not be used as an ordinary predictor
          in the indicated model.
        - **Check timing first** means verify that the information was known
          at the beginning of the period being predicted.
        - A high number of unique values does not automatically make a
          variable unsuitable; it may instead require careful encoding.
        """
    )

    st.warning(
        "Do not fit the final models from this screening table alone. "
        "Confirm the data dictionary, prediction timing, repeated-record "
        "structure, and train/test split strategy first."
    )
