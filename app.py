# app.py — Dengue Forecast Chatbot (LSTM) + 10-Year Forecast Viewer (with Risk Levels + Uncertainty)
# Replace your old app.py with this file

import os
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
CALIBRATION_CSV = "data/CALIBRATION_PARAMS_2021.csv"     # optional
FORECAST_10Y_CSV = "data/FORECAST_10Y.csv"               # optional (precomputed 10y forecast)


# =========================
# HELPERS
# =========================
def month_start(x):
    return pd.Timestamp(x).to_period("M").to_timestamp()

def parse_12_cases(text: str):
    raw = text.replace(",", "\n").splitlines()
    vals = [r.strip() for r in raw if r.strip() != ""]
    nums = []
    for v in vals:
        nums.append(float(re.sub(r"[^\d\.\-eE+]", "", v)))
    return nums

def safe_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def load_feature_cols(path_json):
    if not os.path.exists(path_json):
        raise FileNotFoundError(f"Missing {path_json}. Export feature_cols.json from Colab.")
    with open(path_json, "r") as f:
        cols = json.load(f)
    if not isinstance(cols, list) or len(cols) == 0:
        raise ValueError("feature_cols.json is empty or invalid.")
    return cols

def load_features_table(path_csv):
    df = safe_read_csv(path_csv)

    if "DATE" in df.columns:
        date_col = "DATE"
    elif "DATE_TARGET" in df.columns:
        date_col = "DATE_TARGET"
    else:
        raise ValueError("features_monthly.csv must have DATE or DATE_TARGET column.")

    df[date_col] = pd.to_datetime(df[date_col])
    df[date_col] = df[date_col].apply(month_start)
    df = df.sort_values(date_col).reset_index(drop=True)

    if date_col != "DATE":
        df = df.rename(columns={date_col: "DATE"})

    return df

def load_calibration(path_csv):
    if not os.path.exists(path_csv):
        return None
    cal = pd.read_csv(path_csv)

    needed = {"Model", "a_intercept", "b_slope"}
    if not needed.issubset(set(cal.columns)):
        return None

    row = cal.loc[cal["Model"].astype(str).str.lower() == "lstm"]
    if row.empty:
        return None

    a = float(row.iloc[0]["a_intercept"])
    b = float(row.iloc[0]["b_slope"])
    return (a, b)

@st.cache_data
def load_forecast_csv(path):
    if not os.path.exists(path):
        return None
    f = pd.read_csv(path)
    if "DATE" in f.columns:
        f["DATE"] = pd.to_datetime(f["DATE"])
    return f

def compute_dengue_minmax_from_table(features_df):
    if "DENGUE_CASES_TARGET" not in features_df.columns:
        return (0.0, float(features_df.select_dtypes(include=[np.number]).max().max()))

    if "SET" in features_df.columns:
        train_rows = features_df.loc[features_df["SET"].astype(str).str.upper() == "TRAIN"]
        if not train_rows.empty:
            dmin = float(train_rows["DENGUE_CASES_TARGET"].min())
            dmax = float(train_rows["DENGUE_CASES_TARGET"].max())
            return dmin, dmax

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

def apply_calibration(pred_cases, cal_params):
    if cal_params is None:
        return np.asarray(pred_cases, dtype=float)
    a, b = cal_params
    pred_cal = a + b * np.asarray(pred_cases, dtype=float)
    return np.clip(pred_cal, 0.0, None)

def build_X_for_month(features_df, feature_cols, target_month, last12_cases, dmin, dmax, timesteps=1):
    row = features_df.loc[features_df["DATE"] == target_month]
    if row.empty:
        raise ValueError(f"No row found for month {target_month.date()} in features_monthly.csv")

    base = row.iloc[0].to_dict()

    last12_scaled = scale_cases(last12_cases, dmin, dmax)
    dengue_lag_map = {f"DENGUE_CASES_LAG_{k}_SCALED": float(last12_scaled[-k]) for k in range(1, 13)}

    missing = []
    feat_vals = []
    for c in feature_cols:
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
            "Fix: regenerate features_monthly.csv from the SAME sheet used for training."
        )

    X = np.array(feat_vals, dtype=np.float32).reshape(1, timesteps, len(feature_cols))
    return X

def recursive_forecast(model, features_df, feature_cols, last_observed_month, last12_cases,
                       horizon, dmin, dmax, cal_params, timesteps=1):
    history_cases = list(map(float, last12_cases))
    preds = []

    start_month = (last_observed_month.to_period("M") + 1).to_timestamp()

    for i in range(horizon):
        target_month = (start_month.to_period("M") + i).to_timestamp()

        X = build_X_for_month(features_df, feature_cols, target_month, history_cases[-12:], dmin, dmax, timesteps)

        pred_scaled = float(model.predict(X, verbose=0).reshape(-1)[0])
        pred_cases = float(unscale_cases([pred_scaled], dmin, dmax)[0])
        pred_cases = max(0.0, pred_cases)

        pred_cases_cal = float(apply_calibration([pred_cases], cal_params)[0])

        preds.append({"DATE": target_month, "PRED_CASES_RAW": pred_cases, "PRED_CASES_CAL": pred_cases_cal})
        history_cases.append(pred_cases_cal)

    return pd.DataFrame(preds)


# =========================
# SIDEBAR: LOAD ASSETS
# =========================
st.sidebar.title("Dengue Forecast Dashboard")

page = st.sidebar.radio("Choose view", ["Predict (Chatbot)", "10-Year Forecast", "Debug / Files"])
st.sidebar.divider()
st.sidebar.header("Files Status")

# Load features
try:
    features_df = load_features_table(FEATURES_CSV)
    st.sidebar.success("✅ Loaded data/features_monthly.csv")
except Exception as e:
    st.sidebar.error("❌ features_monthly.csv not found or invalid")
    st.sidebar.exception(e)
    st.stop()

# Load feature cols
try:
    feature_cols = load_feature_cols(FEATURE_COLS_JSON)
    st.sidebar.success("✅ Loaded data/feature_cols.json")
except Exception as e:
    st.sidebar.error("❌ feature_cols.json not found or invalid")
    st.sidebar.exception(e)
    st.stop()

# Load model safely
try:
    if not os.path.exists(MODEL_H5):
        raise FileNotFoundError(f"Missing {MODEL_H5}")
    model = load_model(MODEL_H5, compile=False)
    st.sidebar.success("✅ Loaded models/lstm_model.h5")
except Exception as e:
    st.sidebar.error("❌ lstm_model.h5 not found or invalid")
    st.sidebar.exception(e)
    st.stop()

# Validate model input shape vs feature_cols
try:
    input_shape = model.input_shape
    timesteps = int(input_shape[1]) if input_shape and len(input_shape) >= 3 else 1
    nfeat_model = int(input_shape[2]) if input_shape and len(input_shape) >= 3 else len(feature_cols)

    if nfeat_model != len(feature_cols):
        st.sidebar.error("❌ Feature mismatch (Model vs feature_cols.json)")
        st.sidebar.write(f"Model expects features = **{nfeat_model}**")
        st.sidebar.write(f"feature_cols.json has = **{len(feature_cols)}**")
        st.sidebar.info("Fix: upload matching feature_cols.json or retrain/resave model.")
        st.stop()
except Exception as e:
    st.sidebar.warning("Could not validate model input shape.")
    st.sidebar.exception(e)
    timesteps = 1

# Calibration
cal_params = load_calibration(CALIBRATION_CSV)
use_calibration = st.sidebar.checkbox("Apply calibration (if available)", value=True)
if cal_params and use_calibration:
    st.sidebar.info(f"Calibration (LSTM): a={cal_params[0]:.4f}, b={cal_params[1]:.4f}")
elif os.path.exists(CALIBRATION_CSV) and not cal_params:
    st.sidebar.warning("Calibration CSV found, but LSTM row/columns not detected.")
else:
    st.sidebar.caption("Calibration optional (data/CALIBRATION_PARAMS_2021.csv)")

# Dengue scaling min/max
dmin, dmax = compute_dengue_minmax_from_table(features_df)
st.sidebar.write("**Dengue scaling (from TRAIN if available):**")
st.sidebar.write(f"Min_train = {dmin:.2f}")
st.sidebar.write(f"Max_train = {dmax:.2f}")


# =========================
# PAGE 1: CHATBOT PREDICT
# =========================
if page == "Predict (Chatbot)":
    st.title("💬 LGU Chatbot Input (Climate-integrated LSTM)")
    st.write("Enter last **12 months** dengue cases → predicts next months using climate-integrated LSTM.")

    available_months = features_df["DATE"].drop_duplicates().sort_values().tolist()
    if not available_months:
        st.error("No dates found in features_monthly.csv.")
        st.stop()

    col1, col2 = st.columns([1, 1])
    with col1:
        last_month = st.selectbox(
            "Select the LAST month you have actual dengue cases for:",
            options=available_months,
            index=max(0, len(available_months) - 13)
        )
    with col2:
        horizon = st.slider("How many months ahead to predict?", 1, 12, 3)

    st.markdown("**Enter last 12 monthly dengue cases (oldest → newest).**")
    cases_text = st.text_area(
        "Paste 12 numbers separated by commas or new lines",
        height=140,
        placeholder="Example:\n12, 18, 25, 40, 55, 60, 45, 30, 22, 19, 15, 20"
    )

    if st.button("🔮 Predict"):
        try:
            last12_cases = parse_12_cases(cases_text)
            if len(last12_cases) != 12:
                st.error(f"You entered {len(last12_cases)} values. Please enter exactly 12.")
                st.stop()

            start_month = (pd.Timestamp(last_month).to_period("M") + 1).to_timestamp()
            needed = [(start_month.to_period("M") + i).to_timestamp() for i in range(horizon)]
            missing = [m for m in needed if (features_df["DATE"] == m).sum() == 0]
            if missing:
                st.error("Missing forecast months in features table:\n" + "\n".join([str(m.date()) for m in missing]))
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
                    cal_params=(cal_params if (cal_params and use_calibration) else None),
                    timesteps=timesteps
                )

            st.success("Done ✅")

            show = pred_df.copy()
            show["DATE"] = show["DATE"].dt.strftime("%Y-%m-%d")
            st.subheader("Predictions")
            st.dataframe(show, use_container_width=True)

            st.subheader("Chart")
            fig = plt.figure()
            x = pd.to_datetime(pred_df["DATE"])
            plt.plot(x, pred_df["PRED_CASES_RAW"], marker="o", label="Pred (raw)")
            plt.plot(x, pred_df["PRED_CASES_CAL"], marker="o", label="Pred (calibrated)")
            plt.xticks(rotation=45)
            plt.xlabel("Month")
            plt.ylabel("Predicted dengue cases")
            plt.legend()
            st.pyplot(fig, clear_figure=True)

            st.download_button(
                "⬇️ Download predictions CSV",
                data=pred_df.assign(DATE=pred_df["DATE"].dt.strftime("%Y-%m-%d")).to_csv(index=False).encode("utf-8"),
                file_name="dengue_predictions_chatbot.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error("Error:")
            st.exception(e)


# =========================
# PAGE 2: 10-YEAR FORECAST
# =========================
elif page == "10-Year Forecast":
    st.title("📈 10-Year Forecast (Precomputed CSV + Risk Levels)")

    st.write("Upload the file as `data/FORECAST_10Y.csv` (should contain DATE + median/p10/p90 + RISK_LEVEL).")

    fc = load_forecast_csv(FORECAST_10Y_CSV)
    if fc is None:
        st.error("Forecast file not found. Add `data/FORECAST_10Y.csv` to your repo.")
        st.stop()

    if "DATE" not in fc.columns:
        st.error("FORECAST_10Y.csv must contain a DATE column.")
        st.dataframe(fc.head(20))
        st.stop()

    # ✅ Option A: detect predicted cases column including "median"
    case_col = None
    for c in ["median", "PRED_CASES", "PRED_CASES_CAL", "PRED_CASES_RAW", "p90", "p10"]:
        if c in fc.columns:
            case_col = c
            break

    if case_col is None:
        st.warning("Forecast loaded but no predicted cases column detected.")
        st.dataframe(fc, use_container_width=True)
        st.stop()

    min_date = fc["DATE"].min()
    max_date = fc["DATE"].max()

    start, end = st.slider(
        "Select date range",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=(min_date.to_pydatetime(), max_date.to_pydatetime())
    )

    fc_view = fc[(fc["DATE"] >= pd.to_datetime(start)) & (fc["DATE"] <= pd.to_datetime(end))].copy()

    st.subheader("Forecast Table")
    st.dataframe(fc_view, use_container_width=True)

    # Risk summary
    if "RISK_LEVEL" in fc_view.columns:
        st.subheader("Risk Summary")
        st.write(fc_view["RISK_LEVEL"].value_counts())

    # Uncertainty plot
    st.subheader("Forecast Trend")
    fig = plt.figure()
    if "median" in fc_view.columns:
        plt.plot(fc_view["DATE"], fc_view["median"], label="Median forecast")
    else:
        plt.plot(fc_view["DATE"], fc_view[case_col], label=case_col)

    if "p10" in fc_view.columns and "p90" in fc_view.columns:
        plt.fill_between(fc_view["DATE"], fc_view["p10"], fc_view["p90"], alpha=0.2, label="10–90% interval")

    plt.xticks(rotation=45)
    plt.xlabel("Month")
    plt.ylabel("Predicted dengue cases")
    plt.legend()
    st.pyplot(fig, clear_figure=True)

    st.download_button(
        "⬇️ Download filtered forecast CSV",
        data=fc_view.to_csv(index=False).encode("utf-8"),
        file_name="forecast_10y_filtered.csv",
        mime="text/csv"
    )

    st.caption("Note: 10-year forecasts are scenario outputs; report MAE/RMSE only on observed years.")


# =========================
# PAGE 3: DEBUG
# =========================
else:
    st.title("🧪 Debug / Files")

    st.subheader("Model input shape")
    st.write(model.input_shape)

    st.subheader("feature_cols.json (expected by model)")
    st.write(f"Count: {len(feature_cols)}")
    st.write(feature_cols[:50])
    if len(feature_cols) > 50:
        st.write("...")

    st.subheader("features_monthly.csv columns")
    st.write(f"Count: {len(features_df.columns)}")
    st.write(list(features_df.columns))

    st.subheader("Preview of features_monthly.csv")
    st.dataframe(features_df.head(10), use_container_width=True)

    st.subheader("Forecast file status")
    st.write(f"Looking for: {FORECAST_10Y_CSV}")
    if os.path.exists(FORECAST_10Y_CSV):
        st.success("✅ Found FORECAST_10Y.csv")
        fc = load_forecast_csv(FORECAST_10Y_CSV)
        st.dataframe(fc.head(10), use_container_width=True)
    else:
        st.warning("FORECAST_10Y.csv not found (optional).")
