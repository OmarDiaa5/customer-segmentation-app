"""Streamlit app: customer segmentation (RFM + clustering) for an online store."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from segmentation import REQUIRED_COLUMNS, RFM, action_for, clean_transactions, compute_rfm

ROOT = Path(__file__).parent
st.set_page_config(page_title="Customer Segmentation", page_icon="👥", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "model" / "best_model.joblib")


@st.cache_data
def load_meta():
    return json.loads((ROOT / "model" / "metrics.json").read_text())


@st.cache_data
def load_customers():
    return pd.read_csv(ROOT / "data" / "customer_segments.csv", dtype={"CustomerID": str})


bundle, meta, customers = load_model(), load_meta(), load_customers()
pipe, names = bundle["pipeline"], {int(k): v for k, v in bundle["names"].items()}


def segment(rfm: pd.DataFrame) -> pd.Series:
    return pd.Series(pipe.predict(rfm[RFM]), index=rfm.index).map(names)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    s = df.groupby("Segment").agg(Customers=("Recency", "size"),
                                  Median_days_since_order=("Recency", "median"),
                                  Median_orders=("Frequency", "median"),
                                  Median_spend=("Monetary", "median"),
                                  Revenue=("Monetary", "sum"))
    s.insert(1, "Customers_%", (s["Customers"] / s["Customers"].sum() * 100).round(1))
    s.insert(len(s.columns), "Revenue_%", (s["Revenue"] / s["Revenue"].sum() * 100).round(1))
    s["Recommended action"] = [action_for(x) for x in s.index]
    return s.sort_values("Median_spend", ascending=False).drop(columns="Revenue")


st.title("👥 Customer Segmentation")
st.write("Groups customers by **how recently**, **how often** and **how much** they buy (RFM), "
         "and suggests a marketing action for each group.")
st.caption(f"Model built on {meta['customers']:,} customers of a UK online store "
           f"({meta['n_segments']} segments). Segments are a guide for marketing, not a fixed truth.")

tab_overview, tab_one, tab_upload = st.tabs(
    ["Segment overview", "Check one customer", "Segment my customers (CSV)"])

with tab_overview:
    summary = summarise(customers)
    st.dataframe(summary.style.format({"Median_spend": "£{:,.0f}"}), width="stretch")
    plot = customers.assign(Spend_log10=np.log10(customers["Monetary"].clip(lower=1)))
    st.scatter_chart(plot, x="Recency", y="Spend_log10", color="Segment",
                     x_label="Days since last order", y_label="Total spend (log10 £)")

with tab_one:
    with st.form("one"):
        c1, c2, c3 = st.columns(3)
        r = c1.number_input("Days since last order", 0, 3650, 30)
        f = c2.number_input("Number of orders", 1, 10_000, 3)
        m = c3.number_input("Total spend (£)", 1.0, 1e7, 500.0, step=50.0)
        go = st.form_submit_button("Find segment", type="primary")
    if go:
        seg = segment(pd.DataFrame([{"Recency": r, "Frequency": f, "Monetary": m}])).iloc[0]
        st.success(f"Segment: **{seg}**")
        st.write(f"Recommended action: {action_for(seg)}")

with tab_upload:
    st.write("Upload transaction lines (one row per product in an order). Required columns: "
             + ", ".join(f"`{c}`" for c in REQUIRED_COLUMNS) + ".")
    sample = (ROOT / "data" / "sample_transactions.csv").read_text()
    st.download_button("Download a sample file", sample, "sample_transactions.csv", "text/csv")
    file = st.file_uploader("Transactions CSV", type="csv")
    if st.button("Segment customers", type="primary"):
        if file is None:
            st.warning("Please upload a CSV file first.")
            st.stop()
        try:
            tx = pd.read_csv(file, encoding_errors="replace")
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            st.stop()
        missing = [c for c in REQUIRED_COLUMNS if c not in tx.columns]
        if missing:
            st.error("Missing column(s): " + ", ".join(missing))
            st.stop()
        try:
            tx, info = clean_transactions(tx)
        except Exception as e:
            st.error(f"Could not clean the data (check dates and numbers): {e}")
            st.stop()
        if tx.empty:
            st.error("No valid purchases left after cleaning.")
            st.stop()
        rfm = compute_rfm(tx)
        rfm["Segment"] = segment(rfm)
        st.write(f"{info['rows_raw']:,} lines read, {info['rows_clean']:,} valid purchases, "
                 f"**{len(rfm):,} customers** segmented (recency measured from "
                 f"{tx['InvoiceDate'].max():%d %b %Y}).")
        st.dataframe(summarise(rfm).style.format({"Median_spend": "{:,.0f}"}),
                     width="stretch")
        out = rfm.reset_index()
        out["Recommended action"] = out["Segment"].map(action_for)
        st.dataframe(out, hide_index=True, width="stretch")
        st.download_button("Download customer segments (CSV)", out.to_csv(index=False),
                           "customer_segments.csv", "text/csv")

with st.expander("Model comparison and evaluation"):
    st.write(f"Selected model: **{meta['selected_model']}**, {meta['n_segments']} segments")
    st.caption(f"Selected by {meta['selection_rule']}. Silhouette and Calinski-Harabasz: higher is "
               "better; Davies-Bouldin: lower is better.")
    st.dataframe(pd.DataFrame(meta["results"]), hide_index=True, width="stretch")
    for img in ["segments_overview.png", "choose_k.png"]:
        if (ROOT / "reports" / img).exists():
            st.image(str(ROOT / "reports" / img))
