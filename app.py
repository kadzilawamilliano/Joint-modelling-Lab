import io
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from exploratory_analysis import show_exploratory_analysis
from claim_cost_analysis import show_claim_cost_analysis
from nonrenewal_analysis import show_nonrenewal_analysis
from predictor_analysis import show_predictor_analysis
from repeated_observation_analysis import (
    show_repeated_observation_analysis
)
from outcome_construction import show_outcome_construction

st.set_page_config(
    page_title="Motor Insurance Research Lab",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Motor Insurance Research Lab")
st.caption("Data audit • Policy non-renewal construction • Predictor screening")
st.markdown(
    """
    This app supports the exploratory and data-preparation stage of the research:
    **Joint Modelling of Motor Insurance Claim Costs and Policy Non-central Using Shared Random Effects.**

    Upload the source CSV, inspect data quality, construct a provisional policy-level
    non-renewal indicator, and review candidate predictors before fitting models.
    """
)

with st.sidebar:
    st.header("1. Upload data")
    uploaded = st.file_uploader(
        "Upload the motor insurance CSV",
        type=["csv", "txt"],
        help="The published dataset is semicolon-separated. The app also tries comma-separated files.",
    )
    st.divider()
    st.header("2. Outcome settings")
    cutoff = st.date_input(
        "Observation-window cutoff",
        value=date(2018, 12, 31),
        help="Provisional cutoff. Confirm this against the dataset documentation before final modelling.",
    )
    st.caption("Non-renewal is defined for the same ID/policy-product record, not necessarily for the whole customer.")

if uploaded is None:
    st.info("Upload your dataset in the left panel to begin.")
    st.subheader("What you can investigate")
    c1, c2, c3 = st.columns(3)
    c1.metric("Data audit", "Missingness, duplicates, dates")
    c2.metric("Outcome audit", "Claim costs + non-renewal")
    c3.metric("Predictor screening", "Candidate variables + warnings")
    st.markdown(
        """
        **Expected source columns include:** `ID`, `Date_last_renewal`, `Date_next_renewal`,
        `Cost_claims_year`, `Lapse`, `Date_lapse`, and vehicle/customer characteristics.

        **Important:** this app screens variables; it does not automatically prove that a variable
        is statistically significant, causally relevant, or available at the intended prediction time.
        """
    )
    st.stop()


@st.cache_data(show_spinner=False)
def read_uploaded_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    # Try the source's likely semicolon delimiter first, then comma.
    errors = []
    for sep in [";", ",", "\t"]:
        try:
            frame = pd.read_csv(io.BytesIO(file_bytes), sep=sep, low_memory=False)
            if frame.shape[1] > 1:
                return frame
        except Exception as exc:
            errors.append(str(exc))
    raise ValueError("Could not parse the file. Please check that it is a valid CSV.")


try:
    raw = read_uploaded_file(uploaded.getvalue(), uploaded.name)
except Exception as exc:
    st.error(f"Could not read the uploaded file: {exc}")
    st.stop()

df = raw.copy()
df.columns = [str(c).strip() for c in df.columns]

date_cols = [
    "Date_start_contract", "Date_last_renewal", "Date_next_renewal",
    "Date_birth", "Date_driving_licence", "Date_lapse"
]
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

# Convert likely numeric fields without forcing identifier/text fields.
numeric_cols = [
    "Seniority", "Policies_in_force", "Max_policies", "Max_products",
    "Lapse", "Payment", "Premium", "Cost_claims_year", "N_claims_year",
    "N_claims_history", "R_Claims_history", "Type_risk", "Area",
    "Second_driver", "Year_matriculation", "Power", "Cylinder_capacity",
    "Value_vehicle", "N_doors", "Length", "Weight", "Distribution_channel"
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

required_for_target = {"ID", "Date_last_renewal", "Date_next_renewal"}
has_target_fields = required_for_target.issubset(df.columns)

# Stable source order is retained in _row_order; sorting is used only to inspect same-ID sequence.
df["_row_order"] = np.arange(len(df))
if "ID" in df.columns:
    df["ID"] = df["ID"].astype("string").str.strip()

st.success(f"Loaded {len(df):,} rows and {len(raw.columns):,} original columns.")
tab_overview, tab_quality, tab_outcomes, tab_predictors, tab_export, tab_exploratory, tab_claim, tab_nonrenewal,tab_repeated= st.tabs(
    ["Overview", "Data quality", "Outcome construction", "Predictor screening", "Export","Exploratory analysis","Claim cost analysis", "Nonrenewal analysis", "Repeated observations"]
)

with tab_overview:
    st.subheader("Dataset overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{len(df):,}")
    m2.metric("Original columns", f"{len(raw.columns):,}")
    m3.metric("Duplicate full rows", f"{int(raw.duplicated().sum()):,}")
    m4.metric("Unique IDs", f"{df['ID'].nunique(dropna=True):,}" if "ID" in df else "ID not found")
    st.markdown("**Column names and inferred types**")
    schema = pd.DataFrame({
        "column": raw.columns,
        "inferred type": [str(df[c].dtype) for c in raw.columns],
        "missing count": [int(df[c].isna().sum()) for c in raw.columns],
        "missing %": [round(100 * df[c].isna().mean(), 2) for c in raw.columns],
        "distinct non-missing": [int(df[c].nunique(dropna=True)) for c in raw.columns],
    })
    st.dataframe(schema, use_container_width=True, hide_index=True)
    st.markdown("**Preview of uploaded data**")
    st.dataframe(df.drop(columns=["_row_order"], errors="ignore").head(30), use_container_width=True)

with tab_quality:
    st.subheader("Missingness and duplicate checks")
    miss = pd.DataFrame({
        "variable": raw.columns,
        "missing_n": [int(df[c].isna().sum()) for c in raw.columns],
        "missing_pct": [round(100 * df[c].isna().mean(), 2) for c in raw.columns],
    }).sort_values(["missing_pct", "missing_n"], ascending=False)
    st.dataframe(miss, use_container_width=True, hide_index=True)
    st.download_button(
        "Download missingness audit CSV",
        data=miss.to_csv(index=False).encode("utf-8"),
        file_name="missingness_audit.csv",
        mime="text/csv",
    )

    st.markdown("**Numeric summaries**")
    numeric = df.select_dtypes(include="number").drop(columns=["_row_order"], errors="ignore")
    if not numeric.empty:
        st.dataframe(numeric.describe().T, use_container_width=True)
    else:
        st.write("No numeric columns detected.")

    st.markdown("**Date range checks**")
    available_dates = [c for c in date_cols if c in df.columns]
    if available_dates:
        date_summary = []
        for col in available_dates:
            date_summary.append({
                "variable": col,
                "valid dates": int(df[col].notna().sum()),
                "earliest": df[col].min(),
                "latest": df[col].max(),
                "missing": int(df[col].isna().sum()),
            })
        st.dataframe(pd.DataFrame(date_summary), use_container_width=True, hide_index=True)
    else:
        st.warning("No recognized date columns found.")

    if {"Date_last_renewal", "Date_next_renewal"}.issubset(df.columns):
        bad_order = df[
            df["Date_last_renewal"].notna()
            & df["Date_next_renewal"].notna()
            & (df["Date_next_renewal"] <= df["Date_last_renewal"])
        ]
        st.metric("Rows where next renewal is not after last renewal", f"{len(bad_order):,}")
        if len(bad_order):
            st.warning("Investigate these date-order cases before modelling.")
            st.dataframe(bad_order.head(50), use_container_width=True)

    if "N_doors" in df.columns:
        zero_doors = int((df["N_doors"] == 0).sum())
        if zero_doors:
            st.warning(
                f"`N_doors` contains {zero_doors:,} zero values. Do not automatically recode them as missing; "
                "check whether zero is a valid code for some vehicle types."
            )

    if "Value_vehicle" in df.columns:
        st.warning(
            "`Value_vehicle` is documented as vehicle market value on 31 December 2019. "
            "It may leak future information for earlier prediction dates, so exclude it from the main "
            "predictor set unless its timing is defensible."
        )

#with tab_outcomes:
with tab_outcomes:
    target_df = show_outcome_construction(
        df=df,
        cutoff_date=cutoff,
    )

    if target_df is not None:
        st.session_state["target_df"] = target_df

#with tab_predictors:
with tab_predictors:
    target_df = st.session_state.get("target_df")
    show_predictor_analysis(df, target_df)


with tab_export:
    st.subheader("Export cleaned data and audit outputs")
    st.write("Download the parsed dataset with standardized dates and numeric conversions. Original values are not overwritten outside this app.")
    export_df = df.drop(columns=["_row_order"], errors="ignore").copy()
    # Convert datetimes to ISO format for portable CSV output.
    for col in export_df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        export_df[col] = export_df[col].dt.strftime("%Y-%m-%d")
    st.download_button(
        "Download parsed data CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="motor_insurance_parsed_data.csv",
        mime="text/csv",
    )
    if "target_df" in st.session_state:
        target_export = st.session_state["target_df"].copy()
        for col in target_export.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
            target_export[col] = target_export[col].dt.strftime("%Y-%m-%d")
        target_export = target_export.drop(columns=["_row_order"], errors="ignore")
        st.download_button(
            "Download data with provisional non-renewal target and audit flags",
            data=target_export.to_csv(index=False).encode("utf-8"),
            file_name="motor_insurance_with_provisional_nonrenewal.csv",
            mime="text/csv",
        )
    st.markdown("**Recommended next steps before modelling**")
    st.markdown(
        """
        1. Confirm the observation-window cutoff and prediction time.
        2. Resolve date mismatches and invalid/implausible values.
        3. Verify the time definition of historical claims and policy-count variables.
        4. Decide whether evaluation targets future observations for known IDs or entirely unseen IDs.
        5. Use a time-based test set and compare Tweedie, logistic, and shared-random-effect models on the same eligible records.
        """
    )

with tab_exploratory:
    show_exploratory_analysis(df)
with tab_claim:
     show_claim_cost_analysis(df)
with tab_nonrenewal:
    target_df = st.session_state.get("target_df")
    show_nonrenewal_analysis(target_df)
with tab_repeated:
    target_df = st.session_state.get("target_df")
    show_repeated_observation_analysis(df, target_df)

st.divider()
st.caption("Research support tool. All outcome definitions and predictor choices require researcher review and documentation.")
