import os
import io
import json
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Dengue Forecast Chatbot (LSTM)", layout="wide")

FEATURES_CSV = "data/features_monthly.csv"
FEATURE_COLS_JSON = "data/feature_cols.json"
MODEL_H5 = "models/lstm_model.h5"
CALIBRATION_CSV = "data/CALIBRATION_PARAMS_2021.csv"  # optional

# =========================
# HELPERS
# =========================
def month_start(x):
    return pd.Timestamp(x).to_period("M").to_timestamp()

def parse_12_cases(text: str):
    raw = text.replace(",", "\n").splitlines()
    vals = [r.strip() for r in raw if r.strip() != ""]
    nums = [float(v) for v in vals]
    return nums

def rmse(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def safe_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def load_feature_cols(path_json):
    if not os.path.exists(path_json):
        raise FileNotFoundError(
            f"Missing {path_json}. You must export feature_cols.json from Colab."
        )
    with open(path_json, "r") as f:
        cols = json.load(f)
    if not isinstance(cols, list) or len(cols) == 0:
        raise ValueError("feature_cols.json is empty or invalid.")
    return cols

def load_features_table(path_csv):
    df = safe_read_csv(path_csv)

    # accept DATE or DATE_TARGET
    if "DATE" in df.columns:
        date_col = "DATE"
    elif "DATE_TARGET" in df.columns:
        date_col = "DATE_TARGET"
    else:
        raise ValueError("features_monthly.csv must have DATE or DATE_TARGET column.")

    df[date_col] = pd.to_datetime(df[date_col])
    df[date_col] = df[date_col].apply(month_start)
    df = df.sort_values(date_col).reset_index(drop=True)

    # normalize to a single name "DATE"
    if date_col != "DATE":
        df = df.rename(columns={date_col: "DATE"})

    return df

def load_calibration(path_csv):
    if not os.path.exists(path_csv):
        return None
    cal = pd.read_csv(path_csv)
    # expected columns: Model, a_intercept, b_slope
    if not set(["Model", "a_intercept", "b_slope"]).issubset(cal.columns):
        return None
    row = cal.loc[cal["Model"].str.lower() == "lstm"]
    if row.empty:
        return None
    a = float(row.iloc[0]["a_intercept"])
    b = float(row.iloc[0]["b_slope"])
    return (a, b)

def compute_dengue_minmax_from_table(features_df):
    """
    Uses TRAIN rows if SET exists; else uses full DENGUE_CASES_TARGET column.
    This matches your 'min/max from training' logic without needing extra files.
    """
    if "DENGUE_CASES_TARGET" not in features_df.columns:
        # fallback defaults (won't be as correct)
        return (0.0, float(features_df.select_dtypes(include=[np.number]).max().max()))

    if "SET" in features_df.columns:
        train_rows = features_df.loc[features_df["SET"].astype(str).str.upper() == "TRAIN"]
        if not train_rows.empty:
            dmin = float(train_rows["DENGUE_CASES_TARGET"].min())
            dmax = float(train_rows["DENGUE_CASES_TARGET"].max())
            return dmin, dmax

    # fallback: full column
    dmin = float(features_df["DENGUE_CASES_TARGET"].min())
    dmax = float(features_df["DENGUE_CASES_TARGET"].max())
    return dmin, dmax

def scale_cases(cases, dmin, dmax):
    denom = (dmax - dmin) if (dmax - dmin) != 0 else 1.0
    x = (np.array(cases, dtype=float) - dmin) / denom
    return np.clip(x, 0.0, 1.0)

def unscale_cases(x_scaled, dmin, dmax):
    denom = (dmax - dmin) if (dmax - dmin) != 0 else 1.0
    return (np.array(x_scaled, dtype=float) * denom) + dmin

def build_X_for_month(features_df, feature_cols, target_month, last12_cases, dmin, dmax):
    """
    Builds ONE feature row for the model for a given target_month using:
    - climate & seasonal features from features_monthly.csv for that month
    - dengue lag features computed from user last12_cases (scaled)
    """
    row = features_df.loc[features_df["DATE"] == target_month]
    if row.empty:
        raise ValueError(f"No row found for month {target_month.date()} in features_monthly.csv")

    base = row.iloc[0].to_dict()

    # Compute scaled dengue lags from last12_cases (cases -> scaled)
    last12_scaled = scale_cases(last12_cases, dmin, dmax)

    # Map: lag_1 = last month, lag_12 = 12 months ago
    dengue_lag_map = {}
    for k in range(1, 13):
        dengue_lag_map[f"DENGUE_CASES_LAG_{k}_SCALED"] = float(last12_scaled[-k])

    # Build final vector in the exact order used in training
    missing = []
    feat_vals = []
    for c in feature_cols:
        # Override dengue lags if they appear in feature columns
        if c in dengue_lag_map:
            feat_vals.append(dengue_lag_map[c])
        else:
            if c not in base:
                missing.append(c)
            else:
                feat_vals.append(base[c])

    if missing:
        raise ValueError(
            "Some model features are missing in features_monthly.csv.\n"
            f"Missing columns: {missing}\n\n"
            "Fix: export features_monthly.csv from the SAME sheet used for training "
            "(Win12_ClimCases_Scaled_Values) so all scaled lag columns exist."
        )

    X = np.array(feat_vals, dtype=np.float32).reshape(1, 1, len(feature_cols))  # (batch, timesteps=1, features)
    return X

def apply_calibration(pred_cases, cal_params):
    if cal_params is None:
        return pred_cases
    a, b = cal_params
    pred_cal = a + b * np.array(pred_cases, dtype=float)
    return np.clip(pred_cal, 0.0, None)

def recursive_forecast(model, features_df, feature_cols, last_observed_month, last12_cases, horizon, dmin, dmax, cal_params):
    """
    Predict next horizon months, recursively updating last12_cases with predictions.
    Model outputs scaled dengue (as trained), we unscale to cases, optionally calibrate.
    """
    history_cases = list(map(float, last12_cases))
    preds = []

    start_month = (last_observed_month.to_period("M") + 1).to_timestamp()

    for i in range(horizon):
        target_month = (start_month.to_period("M") + i).to_timestamp()

        X = build_X_for_month(features_df, feature_cols, target_month, history_cases[-12:], dmin, dmax)

        # Predict (scaled)
        pred_scaled = float(model.predict(X, verbose=0).reshape(-1)[0])

        # Unscale to cases
        pred_cases = float(unscale_cases([pred_scaled], dmin, dmax)[0])
        pred_cases = max(0.0, pred_cases)

        # Optional calibration on case scale
        pred_cases_cal = float(apply_calibration([pred_cases], cal_params)[0])

        preds.append({
            "DATE": target_month,
            "PRED_CASES_RAW": pred_cases,
            "PRED_CASES_CAL": pred_cases_cal
        })

        # Update history with CALIBRATED (recommended) so multi-step is stable
        history_cases.append(pred_cases_cal)

    return pd.DataFrame(preds)

# =========================
# LOAD ASSETS
# =========================
st.sidebar.header("Files Status")

try:
    features_df = load_features_table(FEATURES_CSV)
    st.sidebar.success("✅ Loaded data/features_monthly.csv")
except Exception as e:
    st.sidebar.error("❌ features_monthly.csv not found or invalid")
    st.sidebar.exception(e)
    st.stop()

try:
    feature_cols = load_feature_cols(FEATURE_COLS_JSON)
    st.sidebar.success("✅ Loaded data/feature_cols.json")
except Exception as e:
    st.sidebar.error("❌ feature_cols.json not found or invalid")
    st.sidebar.exception(e)
    st.stop()

try:
    if not os.path.exists(MODEL_H5):
        raise FileNotFoundError(f"Missing {MODEL_H5}")
    model = load_model(MODEL_H5)
    st.sidebar.success("✅ Loaded models/lstm_model.h5")
except Exception as e:
    st.sidebar.error("❌ lstm_model.h5 not found or invalid")
    st.sidebar.exception(e)
    st.stop()

cal_params = load_calibration(CALIBRATION_CSV)
use_calibration = st.sidebar.checkbox("Apply calibration (if available)", value=True)
if cal_params and use_calibration:
    st.sidebar.info(f"Calibration loaded for LSTM: a={cal_params[0]:.4f}, b={cal_params[1]:.4f}")
elif os.path.exists(CALIBRATION_CSV) and not cal_params:
    st.sidebar.warning("Calibration file found, but LSTM row/columns not detected.")
else:
    st.sidebar.caption("Calibration optional. (data/CALIBRATION_PARAMS_2021.csv)")

# Dengue scaling min/max
dmin, dmax = compute_dengue_minmax_from_table(features_df)
st.sidebar.write("**Dengue scaling (from TRAIN if available):**")
st.sidebar.write(f"Min_train = {dmin:.2f}")
st.sidebar.write(f"Max_train = {dmax:.2f}")

# =========================
# CHATBOT UI
# =========================
st.subheader("💬 LGU Chatbot Input (Climate-integrated)")

available_months = features_df["DATE"].drop_duplicates().sort_values().tolist()
default_last = available_months[-1] if len(available_months) else pd.Timestamp("2022-12-01")

col1, col2 = st.columns([1, 1])

with col1:
    last_month = st.selectbox(
        "Select the LAST month you have actual dengue cases for:",
        options=available_months,
        index=max(0, len(available_months)-13)  # roughly last year
    )
with col2:
    horizon = st.slider("How many months ahead to predict?", min_value=1, max_value=12, value=3)

st.markdown("**Enter the last 12 monthly dengue cases (oldest → newest, most recent last).**")
cases_text = st.text_area(
    "Paste 12 numbers separated by commas or new lines",
    height=120,
    placeholder="Example:\n12, 18, 25, 40, 55, 60, 45, 30, 22, 19, 15, 20"
)

run = st.button("🔮 Predict")

if run:
    try:
        last12_cases = parse_12_cases(cases_text)
        if len(last12_cases) != 12:
            st.error(f"You entered {len(last12_cases)} values. Please enter exactly 12 months.")
            st.stop()

        # Check the forecast months exist in features table
        start_month = (pd.Timestamp(last_month).to_period("M") + 1).to_timestamp()
        needed = [(start_month.to_period("M") + i).to_timestamp() for i in range(horizon)]
        missing = [m for m in needed if (features_df["DATE"] == m).sum() == 0]
        if missing:
            st.error(
                "Your features table is missing some forecast months:\n"
                + "\n".join([str(m.date()) for m in missing])
                + "\n\nFix: ensure features_monthly.csv includes rows up to your forecast horizon."
            )
            st.stop()

        with st.spinner("Forecasting..."):
            pred_df = recursive_forecast(
                model=model,
                features_df=features_df,
                feature_cols=feature_cols,
                last_observed_month=pd.Timestamp(last_month),
                last12_cases=last12_cases,
                horizon=horizon,
                dmin=dmin,
                dmax=dmax,
                cal_params=(cal_params if (cal_params and use_calibration) else None)
            )

        st.success("Done ✅")

        # Show results
        out = pred_df.copy()
        out["DATE"] = out["DATE"].dt.strftime("%Y-%m-%d")
        st.subheader("Predictions")
        st.dataframe(out, use_container_width=True)

        # Chart
        st.subheader("Forecast Chart")
        fig = plt.figure()
        x = pd.to_datetime(pred_df["DATE"])
        plt.plot(x, pred_df["PRED_CASES_RAW"], marker="o", label="Pred (raw)")
        plt.plot(x, pred_df["PRED_CASES_CAL"], marker="o", label="Pred (calibrated)")
        plt.xticks(rotation=45)
        plt.xlabel("Month")
        plt.ylabel("Predicted dengue cases")
        plt.legend()
        st.pyplot(fig, clear_figure=True)

        # Download
        st.subheader("Download")
        csv_bytes = pred_df.assign(DATE=pred_df["DATE"].dt.strftime("%Y-%m-%d")).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download predictions as CSV",
            data=csv_bytes,
            file_name="dengue_predictions_chatbot.csv",
            mime="text/csv"
        )

        st.caption(
            "Note: This tool uses NASA POWER climate/features from your dataset and predicts dengue cases using your trained LSTM. "
            "Outputs are decision-support estimates, not diagnostic."
        )

    except Exception as e:
        st.error("Something went wrong:")
        st.exception(e)

with st.expander("🔎 Debug: show loaded columns"):
    st.write("Feature columns expected by the model (from feature_cols.json):")
    st.write(feature_cols)
    st.write("Columns available in features_monthly.csv:")
    st.write(list(features_df.columns))
    st.write("First rows:")
    st.dataframe(features_df.head(5), use_container_width=True)
