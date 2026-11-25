
import streamlit as st
import pandas as pd


RISK_COL = "Type_risk"
ID_COL = "ID"
PASSENGER_CAR_CODE = 3

RISK_LABELS = {
    1: "Motorbikes",
    2: "Vans",
    3: "Passenger cars",
    4: "Agricultural vehicles",
}

# Possible names for the non-renewal outcome column.
OUTCOME_CANDIDATES = [
    "policy_nonrenewal",
    "Policy_nonrenewal",
    "nonrenewal",
    "Nonrenewal",
    "policy_lapse",
    "Lapse_outcome",
]


def find_outcome_column(df, preferred=None):
    """Find the existing outcome column without creating a new one."""
    if preferred and preferred in df.columns:
        return preferred

    lookup = {str(col).lower(): col for col in df.columns}

    for candidate in OUTCOME_CANDIDATES:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def standardise_outcome(series):
    """
    Standardise an existing binary outcome:
        0 = observed renewal
        1 = provisional non-renewal
        missing/unrecognised = unresolved

    This function does not construct or infer renewal outcomes.
    """
    result = pd.Series(pd.NA, index=series.index, dtype="Int64")

    numeric = pd.to_numeric(series, errors="coerce")
    result.loc[numeric.isin([0, 1])] = (
        numeric.loc[numeric.isin([0, 1])].astype(int)
    )

    text = series.astype("string").str.strip().str.lower()

    renewal_labels = {
        "renewed",
        "renewal",
        "observed renewal",
        "0",
    }

    nonrenewal_labels = {
        "non-renewed",
        "nonrenewed",
        "non renewal",
        "non-renewal",
        "provisional non-renewal",
        "provisional nonrenewal",
        "1",
    }

    result.loc[text.isin(renewal_labels)] = 0
    result.loc[text.isin(nonrenewal_labels)] = 1

    return result


def prepare_passenger_car_data(
    df,
    risk_col=RISK_COL,
):
    """Filter the original data to passenger-car observations."""
    if risk_col not in df.columns:
        raise KeyError(
            f"Required risk-category column '{risk_col}' was not found."
        )

    risk_values = pd.to_numeric(df[risk_col], errors="coerce")

    passenger_cars = df.loc[
        risk_values == PASSENGER_CAR_CODE
    ].copy()

    return passenger_cars


def calculate_identifier_statistics(df, id_col=ID_COL):
    """Calculate unique, repeated and singleton identifier statistics."""
    if id_col not in df.columns:
        raise KeyError(
            f"Required identifier column '{id_col}' was not found."
        )

    valid_ids = df[id_col].dropna()
    id_counts = valid_ids.value_counts()

    unique_ids = int(id_counts.size)
    repeated_ids = int((id_counts >= 2).sum())
    singleton_ids = int((id_counts == 1).sum())

    return {
        "observations": int(len(df)),
        "missing_ids": int(df[id_col].isna().sum()),
        "unique_ids": unique_ids,
        "repeated_ids": repeated_ids,
        "singleton_ids": singleton_ids,
        "rows_with_repeated_ids": int(
            id_counts[id_counts >= 2].sum()
        ),
    }


def calculate_outcome_statistics(df, outcome_col):
    """Summarise an already constructed renewal outcome."""
    outcome = standardise_outcome(df[outcome_col])

    renewals = int((outcome == 0).sum())
    nonrenewals = int((outcome == 1).sum())
    unresolved = int(outcome.isna().sum())

    return {
        "observed_renewals": renewals,
        "provisional_nonrenewals": nonrenewals,
        "unresolved": unresolved,
        "classified": renewals + nonrenewals,
        "outcome": outcome,
    }


def show_passenger_car_audit(
    df,
    outcome_col=None,
    id_col=ID_COL,
    risk_col=RISK_COL,
    audited_df=None,
):
    """
    Display the passenger-car preliminary data audit in Streamlit.

    Returns:
        passenger_cars: filtered dataframe
        audit_summary: dictionary containing audit statistics
    """

    st.subheader("Passenger-Car Preliminary Data Audit")

    st.write(
        "This audit restricts the analysis to passenger-car observations "
        "identified by Type_risk = 3."
    )

    try:
        passenger_cars = (audited_df.copy()
           if audited_df is not None
           else prepare_passenger_car_data(df, risk_col=risk_col)
        )
        

        identifier_stats = calculate_identifier_statistics(
            passenger_cars,
            id_col=id_col,
        )

    except (KeyError, ValueError) as error:
        st.error(str(error))
        return None, None

    if passenger_cars.empty:
        st.warning("No passenger-car observations were found.")
        return passenger_cars, None

    # Check whether an identifier appears under more than one risk class
    # in the original dataset.
    risk_consistency = None

    if id_col in df.columns and risk_col in df.columns:
        valid = df.dropna(subset=[id_col, risk_col]).copy()

        risk_counts = (
            valid.groupby(id_col)[risk_col]
            .nunique()
        )

        risk_consistency = int((risk_counts > 1).sum())

    st.markdown("#### 1. Passenger-car dataset")

    total_rows = identifier_stats["observations"]
    original_rows = len(df)

    percentage = (
        total_rows / original_rows * 100
        if original_rows > 0 else 0
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Passenger-car observations",
        f"{total_rows:,}",
    )

    col2.metric(
        "Share of original dataset",
        f"{percentage:.2f}%",
    )

    col3.metric(
        "Missing identifiers",
        f"{identifier_stats['missing_ids']:,}",
    )

    st.markdown("#### 2. Identifier structure")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Unique IDs",
        f"{identifier_stats['unique_ids']:,}",
    )

    col2.metric(
        "IDs with repeated observations",
        f"{identifier_stats['repeated_ids']:,}",
    )

    col3.metric(
        "IDs with one observation",
        f"{identifier_stats['singleton_ids']:,}",
    )

    st.metric(
        "Observations belonging to repeated IDs",
        f"{identifier_stats['rows_with_repeated_ids']:,}",
    )

    if risk_consistency is not None:
        st.metric(
            "IDs appearing in multiple risk categories "
            "in the original dataset",
            f"{risk_consistency:,}",
        )

        if risk_consistency > 0:
            st.warning(
                "Some IDs appear under more than one risk category. "
                "Investigate these IDs before relying on the "
                "passenger-car-only longitudinal structure."
            )
        else:
            st.success(
                "No identifier was found across multiple risk categories."
            )

    st.markdown("#### 3. Renewal outcome audit")

    selected_outcome = find_outcome_column(
        passenger_cars,
        preferred=outcome_col,
    )

    if selected_outcome is None:
        st.warning(
            "No existing non-renewal outcome column was found. "
            "The renewal outcome statistics have not been calculated. "
            "Construct the outcome using the scheduled renewal-date "
            "rule before interpreting these figures."
        )

        outcome_stats = None

    else:
        outcome_stats = calculate_outcome_statistics(
            passenger_cars,
            selected_outcome,
        )

        st.caption(
            f"Outcome column used: {selected_outcome}"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Observed renewals",
            f"{outcome_stats['observed_renewals']:,}",
        )

        col2.metric(
            "Provisional non-renewals",
            f"{outcome_stats['provisional_nonrenewals']:,}",
        )

        col3.metric(
            "Unresolved observations",
            f"{outcome_stats['unresolved']:,}",
        )

        st.metric(
            "Observations with classified outcomes",
            f"{outcome_stats['classified']:,}",
        )

        # Show the outcome distribution.
        distribution = pd.DataFrame({
            "Outcome": [
                "Observed renewal",
                "Provisional non-renewal",
                "Unresolved",
            ],
            "Observations": [
                outcome_stats["observed_renewals"],
                outcome_stats["provisional_nonrenewals"],
                outcome_stats["unresolved"],
            ],
        })

        st.dataframe(
            distribution,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### 4. Audit checks")

    checks = {
        "Passenger-car filter applied": True,
        "Missing IDs": identifier_stats["missing_ids"] == 0,
        "Repeated and singleton IDs reconcile": (
            identifier_stats["repeated_ids"]
            + identifier_stats["singleton_ids"]
            == identifier_stats["unique_ids"]
        ),
    }

    if outcome_stats is not None:
        checks["Outcome counts reconcile"] = (
            outcome_stats["observed_renewals"]
            + outcome_stats["provisional_nonrenewals"]
            + outcome_stats["unresolved"]
            == total_rows
        )

    for description, passed in checks.items():
        if passed:
            st.success(f"✓ {description}")
        else:
            st.warning(f"Check required: {description}")

    st.markdown("#### 5. Passenger-car data preview")

    st.dataframe(
        passenger_cars.head(20),
        use_container_width=True,
    )

    csv = passenger_cars.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download passenger-car dataset",
        data=csv,
        file_name="passenger_car_dataset.csv",
        mime="text/csv",
    )

    audit_summary = {
        **identifier_stats,
        "risk_consistency_issues": risk_consistency,
        "outcome_column": selected_outcome,
        "outcome_statistics": outcome_stats,
    }

    return passenger_cars, audit_summary
  
