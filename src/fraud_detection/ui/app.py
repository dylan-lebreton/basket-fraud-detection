"""Streamlit UI for fraud detection inference."""

import os

import pandas as pd
import requests
import streamlit as st


API_URL = os.environ.get("API_URL", "http://fraud-inference:8000")

st.set_page_config(page_title="Fraud Detection", page_icon="🛒", layout="wide")
st.title("🛒 Basket Fraud Detection")


@st.cache_data(ttl=30)
def get_health() -> dict:
    try:
        return requests.get(f"{API_URL}/health", timeout=5).json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


health = get_health()
col1, col2, col3 = st.columns(3)
col1.metric("API status", health.get("status", "unknown"))
col2.metric("Model version", health.get("model_version") or "—")
if col3.button("🔄 Reload model"):
    try:
        r = requests.post(f"{API_URL}/reload", timeout=30).json()
        st.success(f"Reloaded to v{r.get('model_version')}")
        st.cache_data.clear()
    except Exception as e:
        st.error(f"Reload failed: {e}")

st.divider()

tab_single, tab_batch = st.tabs(["Single basket", "Batch CSV"])


with tab_single:
    st.subheader("Predict a single basket")

    basket_id = st.number_input("Basket ID", value=1, step=1)
    n_items = st.number_input("Number of items", min_value=1, max_value=24, value=1)

    items = []
    for i in range(int(n_items)):
        with st.expander(f"Item {i + 1}", expanded=(i == 0)):
            c1, c2 = st.columns(2)
            item = c1.text_input("Category", value="COMPUTERS", key=f"item_{i}")
            cash_price = c2.number_input("Cash price (€)", value=1500.0, key=f"price_{i}")
            c3, c4 = st.columns(2)
            make = c3.text_input("Make", value="APPLE", key=f"make_{i}")
            model = c4.text_input("Model", value="MACBOOK PRO", key=f"model_{i}")
            c5, c6 = st.columns(2)
            goods_code = c5.text_input("Goods code", value="239283", key=f"code_{i}")
            n_prods = c6.number_input("Nb products", min_value=1, value=1, key=f"nprods_{i}")
            items.append({
                "item": item,
                "cash_price": cash_price,
                "make": make,
                "model": model,
                "goods_code": goods_code,
                "nbr_of_prod_purchas": int(n_prods),
            })

    if st.button("Predict", type="primary"):
        try:
            r = requests.post(
                f"{API_URL}/predict",
                json={"id": int(basket_id), "items": items},
                timeout=30,
            )
            r.raise_for_status()
            result = r.json()
            proba = result["fraud_probability"]

            c1, c2 = st.columns(2)
            c1.metric("Fraud probability", f"{proba:.2%}")
            c2.metric("Model version", result["model_version"])

            if proba > 0.5:
                st.error(f"⚠️ High fraud risk ({proba:.2%})")
            elif proba > 0.1:
                st.warning(f"⚡ Moderate risk ({proba:.2%})")
            else:
                st.success(f"✅ Low risk ({proba:.2%})")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


with tab_batch:
    st.subheader("Predict on a CSV file")
    st.caption("Upload a CSV in the challenge format (X_test.csv)")

    uploaded = st.file_uploader("CSV file", type=["csv"])

    if uploaded is not None:
        if st.button("Run batch prediction", type="primary"):
            try:
                r = requests.post(
                    f"{API_URL}/predict/batch",
                    files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
                    timeout=300,
                )
                r.raise_for_status()
                data = r.json()
                df = pd.DataFrame(data["predictions"])

                st.success(f"Predicted {len(df)} baskets with model v{data['model_version']}")

                c1, c2, c3 = st.columns(3)
                c1.metric("Baskets", len(df))
                c2.metric("High risk (>50%)", int((df["fraud_flag"] > 0.5).sum()))
                c3.metric("Mean proba", f"{df['fraud_flag'].mean():.3f}")

                st.dataframe(df.sort_values("fraud_flag", ascending=False), use_container_width=True)

                st.download_button(
                    "Download predictions CSV",
                    df.to_csv(index=False).encode(),
                    "predictions.csv",
                    "text/csv",
                )
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")