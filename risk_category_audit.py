
import pandas as pd
import streamlit as st


def risk_category_audit(
    df,
    risk_col="Type_risk",
    id_col="ID"
):
    """
    Summarise observations and policy-product identifiers
    across the four motor insurance risk categories.
    """

    risk_labels = {
        1: "Motorbikes",
        2: "Vans",
        3: "Passenger cars",
        4: "Agricultural vehicles"
    }

    # Check required columns
    if risk_col not in df.columns:
        st.error(f"Column '{risk_col}' was not found.")
        return None

    if id_col not in df.columns:
        st.error(f"Column '{id_col}' was not found.")
        return None

    # Work on a copy
    audit_df = df[[risk_col, id_col]].copy()

    # Convert risk codes to numeric
    audit_df[risk_col] = pd.to_numeric(
        audit_df[risk_col], errors="coerce"
    )

    # Keep recognised risk categories
    audit_df = audit_df[
        audit_df[risk_col].isin(risk_labels.keys())
    ].copy()

    # Group by vehicle category
    summary = (
        audit_df.groupby(risk_col)
        .agg(
            observations=(id_col, "size"),
            unique_ids=(id_col, "nunique")
        )
        .reindex(risk_labels.keys(), fill_value=0)
    )

    # Calculate repeated and singleton IDs
    repeated_ids = (
        audit_df.groupby(risk_col)[id_col]
        .apply(lambda x: (x.value_counts() >= 2).sum())
    )

    singleton_ids = (
        audit_df.groupby(risk_col)[id_col]
        .apply(lambda x: (x.value_counts() == 1).sum())
    )

    summary["repeated_ids"] = repeated_ids
    summary["singleton_ids"] = singleton_ids

    # Add category names
    summary.insert(
        0,
        "vehicle_category",
        [risk_labels[i] for i in summary.index]
    )

    # Calculate observation percentages
    total_observations = summary["observations"].sum()

    summary["percentage"] = (
        summary["observations"] /
        total_observations * 100
    ).round(2)

    summary = summary.reset_index(drop=True)

    # Display results
    st.subheader("Vehicle Risk Category Audit")

    st.write(
        "Distribution of observations and policy-product "
        "identifiers across the vehicle categories."
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    # Visual comparison
    st.subheader("Observations by Vehicle Category")

    chart_data = summary.set_index("vehicle_category")[
        "observations"
    ]

    st.bar_chart(chart_data)

    # Download results
    csv = summary.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Risk Category Audit",
        data=csv,
        file_name="risk_category_audit.csv",
        mime="text/csv"
    )

    return summary
