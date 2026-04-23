"""Streamlit UI for fraud detection inference."""

import io
import os

import pandas as pd
import requests
import streamlit as st


API_URL = os.environ.get("API_URL", "http://fraud-inference:8000")

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("Basket Fraud Detection")


def call_predict(basket_id: int, items: list[dict]) -> dict:
    r = requests.post(
        f"{API_URL}/predict",
        json={"id": basket_id, "items": items},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def call_predict_batch(filename: str, content: bytes) -> dict:
    r = requests.post(
        f"{API_URL}/predict/batch",
        files={"file": (filename, content, "text/csv")},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


def show_result(proba: float) -> None:
    c1, c2 = st.columns(2)
    c1.metric("Fraud probability", f"{proba:.2%}")
    c2.metric("Prediction", "FRAUD" if proba > 0.5 else "LEGIT")

    if proba > 0.5:
        st.error(f"High fraud risk ({proba:.2%})")
    elif proba > 0.1:
        st.warning(f"Moderate risk ({proba:.2%})")
    else:
        st.success(f"Low risk ({proba:.2%})")


tab_predict, tab_batch = st.tabs(["Predict", "Predict CSV"])


with tab_predict:
    st.subheader("Predict")

    n_items = st.number_input("Number of items", 1, 24, 1)
    items = []

    for i in range(int(n_items)):
        with st.expander(f"Item {i + 1}", expanded=(i == 0)):
            c1, c2 = st.columns(2)
            category = c1.text_input("Category", "COMPUTERS", key=f"cat_{i}")
            cash_price = c2.number_input("Cash price (€)", 0.0, 50000.0, 1500.0, key=f"p_{i}")

            c3, c4 = st.columns(2)
            make = c3.text_input("Make", "APPLE", key=f"make_{i}")
            model = c4.text_input("Model", "MACBOOK PRO", key=f"mod_{i}")

            n_prods = st.number_input("Nb products", 1, 40, 1, key=f"np_{i}")

            items.append({
                "item": category,
                "cash_price": cash_price,
                "make": make,
                "model": model,
                "goods_code": f"GC{i:04d}",
                "nbr_of_prod_purchas": int(n_prods),
            })

    if st.button("Predict", type="primary"):
        try:
            result = call_predict(1, items)
            show_result(result["fraud_probability"])
        except Exception as e:
            st.error(f"Prediction failed: {e}")


with tab_batch:
    st.subheader("Predict CSV")

    uploaded = st.file_uploader("CSV file", type=["csv"])

    if uploaded is not None:
        content = uploaded.getvalue()

        with st.spinner("Running predictions..."):
            try:
                data = call_predict_batch(uploaded.name, content)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        preds_df = pd.DataFrame(data["predictions"])
        preds_df["ID"] = preds_df["ID"].astype(str)
        preds_df = preds_df.rename(columns={"fraud_flag": "fraud_probability"})
        preds_df["fraud_prediction"] = (preds_df["fraud_probability"] > 0.5).astype(int)

        raw_df = pd.read_csv(io.BytesIO(content))
        raw_df["ID"] = raw_df["ID"].astype(str)

        result_df = raw_df.merge(preds_df, on="ID", how="left")

        c1, c2, c3 = st.columns(3)
        c1.metric("Baskets", len(result_df))
        c2.metric("Predicted frauds", int(result_df["fraud_prediction"].sum()))
        c3.metric("Mean probability", f"{result_df['fraud_probability'].mean():.3f}")

        filter_choice = st.radio(
            "Filter",
            ["All", "Only frauds (1)", "Only legit (0)"],
            horizontal=True,
        )
        if filter_choice == "Only frauds (1)":
            view_df = result_df[result_df["fraud_prediction"] == 1]
        elif filter_choice == "Only legit (0)":
            view_df = result_df[result_df["fraud_prediction"] == 0]
        else:
            view_df = result_df

        view_df = view_df.sort_values("fraud_probability", ascending=False)
        st.dataframe(view_df, use_container_width=True)

        st.download_button(
            "Download results CSV",
            view_df.to_csv(index=False).encode(),
            "predictions.csv",
            "text/csv",
        )