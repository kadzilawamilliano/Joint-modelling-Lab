import io
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from exploratory_analysis import show_exploratory analysis

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
    **Joint Modelling of Motor Insurance Claim Costs and Policy Non-Renewal Using Shared Random Effects.**

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
tab_overview, tab_quality, tab_outcomes, tab_predictors, tab_export, tab_exploratory= st.tabs(
    ["Overview", "Data quality", "Outcome construction", "Predictor screening", "Export","Exploratory analysis",
    ]
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

with tab_outcomes:
    st.subheader("Claim-cost outcome")
    if "Cost_claims_year" in df.columns:
        costs = pd.to_numeric(df["Cost_claims_year"], errors="coerce")
        n_valid = int(costs.notna().sum())
        n_zero = int((costs == 0).sum())
        n_positive = int((costs > 0).sum())
        n_negative = int((costs < 0).sum())
        a, b, c, d = st.columns(4)
        a.metric("Valid claim-cost rows", f"{n_valid:,}")
        b.metric("Zero costs", f"{n_zero:,}", f"{(100*n_zero/n_valid if n_valid else 0):.1f}%")
        c.metric("Positive costs", f"{n_positive:,}", f"{(100*n_positive/n_valid if n_valid else 0):.1f}%")
        d.metric("Negative costs", f"{n_negative:,}")
        if n_negative:
            st.warning("Negative annual claim costs found. Investigate adjustments/refunds before fitting a Tweedie model.")
        if n_valid:
            fig, ax = plt.subplots()
            costs.dropna().clip(lower=0).hist(bins=50, ax=ax)
            ax.set_title("Annual claim-cost distribution (display clipped at zero)")
            ax.set_xlabel("Cost_claims_year")
            ax.set_ylabel("Rows")
            st.pyplot(fig)
            plt.close(fig)
            st.caption("A large mass at zero with positive continuous costs can be compatible with a Tweedie compound Poisson-Gamma model, but suitability still needs checking.")

    st.divider()
    st.subheader("Construct provisional policy non-renewal outcome")
    if not has_target_fields:
        st.warning("Cannot construct this outcome: the dataset needs ID, Date_last_renewal, and Date_next_renewal.")
    else:
        cutoff_ts = pd.Timestamp(cutoff)
        ordered = df.sort_values(["ID", "Date_last_renewal", "_row_order"], na_position="last").copy()
        ordered["_next_record_last_renewal"] = ordered.groupby("ID", dropna=False)["Date_last_renewal"].shift(-1)
        ordered["_has_next_record"] = ordered.groupby("ID", dropna=False)["Date_last_renewal"].shift(-1).notna()
        next_dates = ordered["_next_record_last_renewal"]
        scheduled = ordered["Date_next_renewal"]
        observed_match = (
            ordered["_has_next_record"]
            & scheduled.notna()
            & next_dates.notna()
            & (next_dates == scheduled)
        )
        # A following same-ID row that fails to align is ambiguous, not automatically a non-renewal.
        has_following_but_mismatch = ordered["_has_next_record"] & scheduled.notna() & ~observed_match
        eligible = scheduled.notna() & (scheduled <= cutoff_ts)
        outcome = pd.Series(pd.NA, index=ordered.index, dtype="Int64")
        outcome.loc[eligible & observed_match] = 0
        outcome.loc[eligible & ~ordered["_has_next_record"] & scheduled.notna()] = 1
        outcome.loc[eligible & has_following_but_mismatch] = pd.NA
        ordered["policy_nonrenewal"] = outcome
        ordered["renewal_outcome_eligible"] = eligible
        ordered["renewal_date_mismatch_flag"] = has_following_but_mismatch
        # Restore to source row order for export/display.
        target_df = ordered.sort_values("_row_order").copy()
        n_eligible = int(target_df["renewal_outcome_eligible"].sum())
        n_renewed = int((target_df["policy_nonrenewal"] == 0).sum())
        n_nonrenewed = int((target_df["policy_nonrenewal"] == 1).sum())
        n_mismatch = int(target_df["renewal_date_mismatch_flag"].sum())
        n_unlabelled_eligible = int(
            (target_df["renewal_outcome_eligible"] & target_df["policy_nonrenewal"].isna()).sum()
        )
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Rows scheduled by cutoff", f"{n_eligible:,}")
        q2.metric("Observed renewals (0)", f"{n_renewed:,}")
        q3.metric("No same-ID renewal observed (1)", f"{n_nonrenewed:,}")
        q4.metric("Ambiguous date mismatches", f"{n_mismatch:,}")
        st.caption(
            f"Eligible rows without a usable label: {n_unlabelled_eligible:,}. "
            "Rows whose scheduled renewal is after the cutoff are left unlabelled."
        )
        outcome_counts = pd.DataFrame({
            "outcome": ["Renewed under same ID (0)", "No same-ID renewal observed (1)", "Eligible but unresolved"],
            "count": [
                n_renewed,
                n_nonrenewed,
                n_unlabelled_eligible,
            ],
        })
        st.dataframe(outcome_counts, use_container_width=True, hide_index=True)
        if n_mismatch:
            st.warning("Some IDs have a following record but its last-renewal date does not match the scheduled renewal. These cases are flagged and not labelled automatically.")
            st.dataframe(
                target_df.loc[target_df["renewal_date_mismatch_flag"],
                              [c for c in ["ID", "Date_last_renewal", "Date_next_renewal", "_next_record_last_renewal"] if c in target_df]].head(100),
                use_container_width=True,
            )
        if "Lapse" in target_df.columns:
            st.markdown("**Diagnostic comparison with `Lapse` (not the target definition)**")
            lapse_diag = pd.crosstab(
                target_df.loc[target_df["policy_nonrenewal"].notna(), "policy_nonrenewal"],
                target_df.loc[target_df["policy_nonrenewal"].notna(), "Lapse"],
                margins=True,
            )
            st.dataframe(lapse_diag, use_container_width=True)
            st.caption("`Lapse` is an aggregate count and may describe cancellations other than the specific same-ID renewal event. Use it for diagnostics, not as a substitute for this binary outcome.")
        st.session_state["target_df"] = target_df

with tab_predictors:
    st.subheader("Candidate predictor screening")
    st.markdown(
        """
        The lists below are **starting candidates based on subject-matter reasoning**, not a final statistical selection.
        Confirm each variable's availability at the prediction date. Fit models using training data only, and evaluate
        on a later period to reduce optimistic performance estimates.
        """
    )
    claim_candidates = [
        "Type_risk", "Area", "Second_driver", "Distribution_channel", "Payment",
        "Seniority", "Premium", "Power", "Cylinder_capacity", "Year_matriculation",
        "N_doors", "Type_fuel", "Length", "Weight", "Date_birth",
        "Date_driving_licence",
    ]
    nonrenew_candidates = [
        "Distribution_channel", "Payment", "Seniority", "Premium", "Type_risk",
        "Area", "Second_driver", "Policies_in_force", "Max_policies", "Max_products",
        "Power", "Cylinder_capacity", "Year_matriculation", "N_doors", "Type_fuel",
        "Length", "Weight", "Date_birth", "Date_driving_licence",
    ]
    history_candidates = ["N_claims_history", "R_Claims_history"]
    risky_or_target_cols = [
        "ID", "Date_lapse", "Lapse", "Date_next_renewal", "Cost_claims_year",
        "N_claims_year", "Value_vehicle"
    ]

    def present(items):
        return [x for x in items if x in df.columns]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Claim-cost model: initial candidates**")
        st.write(present(claim_candidates))
    with c2:
        st.markdown("**Non-renewal model: initial candidates**")
        st.write(present(nonrenew_candidates))

    st.markdown("**Variables requiring special treatment**")
    special = pd.DataFrame([
        {"variable": "ID", "recommendation": "Use to identify repeated policy-product records / group random effect, not as an ordinary numeric predictor."},
        {"variable": "N_claims_year", "recommendation": "Usually exclude when predicting same-year claim cost because it is only known after the year unfolds."},
        {"variable": "Lapse / Date_lapse", "recommendation": "Do not use to predict the same non-renewal event; may directly encode or occur after the outcome."},
        {"variable": "Value_vehicle", "recommendation": "Documented 31-Dec-2019 valuation; potential future-information leakage for earlier years."},
        {"variable": "N_claims_history / R_Claims_history", "recommendation": "Include only after verifying they are calculated using information available by the prediction date, not the full future policy history."},
        {"variable": "Policies_in_force / Max_policies / Max_products", "recommendation": "Potentially useful for non-renewal, but verify their reference period and whether they contain future/aggregate information."},
        {"variable": "Premium", "recommendation": "May be useful, but confirm that the premium is known at the actual prediction point and consider whether pricing and renewal decisions are jointly determined."},
    ])
    st.dataframe(special, use_container_width=True, hide_index=True)

    st.markdown("**Derived variables (calculated at last-renewal date)**")
    derived = df.copy()
    if "Date_last_renewal" in derived and "Date_birth" in derived:
        derived["age_at_renewal"] = (
            (derived["Date_last_renewal"] - derived["Date_birth"]).dt.days / 365.2425
        )
    if "Date_last_renewal" in derived and "Date_driving_licence" in derived:
        derived["driving_experience_years"] = (
            (derived["Date_last_renewal"] - derived["Date_driving_licence"]).dt.days / 365.2425
        )
    if "Date_last_renewal" in derived and "Year_matriculation" in derived:
        derived["vehicle_age_years"] = (
            derived["Date_last_renewal"].dt.year - derived["Year_matriculation"]
        )
    derived_cols = [c for c in ["age_at_renewal", "driving_experience_years", "vehicle_age_years"] if c in derived]
    if derived_cols:
        st.dataframe(derived[derived_cols].describe().T, use_container_width=True)
        st.caption("Check negative and implausibly large values before retaining derived variables.")

    st.markdown("**High-correlation / redundancy screen for numeric variables**")
    numeric = df.select_dtypes(include="number").drop(columns=["_row_order"], errors="ignore")
    if numeric.shape[1] >= 2:
        corr = numeric.corr(numeric_only=True)
        pairs = []
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                value = corr.iloc[i, j]
                if pd.notna(value) and abs(value) >= 0.8:
                    pairs.append({"variable_1": cols[i], "variable_2": cols[j], "correlation": round(float(value), 3)})
        if pairs:
            st.dataframe(pd.DataFrame(pairs).sort_values("correlation", key=lambda s: s.abs(), ascending=False), use_container_width=True, hide_index=True)
        else:
            st.write("No numeric variable pairs with absolute correlation at least 0.80 were detected.")
    else:
        st.write("Not enough numeric variables for correlation screening.")

    st.warning(
        "Do not choose variables solely because they correlate with the outcome in the full dataset. "
        "Use timing, actuarial rationale, missingness, redundancy, and training-only validation together."
    )

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

with tab_explorattory:
    show_exploratory_analysis(df)

st.divider()
st.caption("Research support tool. All outcome definitions and predictor choices require researcher review and documentation.")
