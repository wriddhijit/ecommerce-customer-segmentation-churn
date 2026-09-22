from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.customer_analysis import (
    load_and_clean, compute_rfm, segment_customers,
    build_churn_training, fit_lda, transform_rfm,
)

st.set_page_config(
    page_title="E-Commerce Customer Segmentation & Churn",
    layout="wide",
)
st.title("Multivariate Customer Segmentation and Churn Prediction")
st.caption("CIA 3 | MCAI513B-3 Multivariate Techniques | Wriddhijit Roy")
st.info(
    "Upload a transaction CSV with InvoiceNo, StockCode, Description, Quantity, "
    "InvoiceDate, UnitPrice, CustomerID, and Country."
)

file = st.file_uploader("Upload transaction CSV", type=["csv"])
if file:
    with st.spinner("Cleaning transactions and fitting multivariate models..."):
        df = load_and_clean(file)
        rfm = compute_rfm(df)
        segments, info = segment_customers(rfm)
        training, cutoff, lower, upper = build_churn_training(df)
        lda, _ = fit_lda(training)
        current_x, _, _ = transform_rfm(rfm, lower, upper)
        segments["Churn_Probability"] = lda.predict_proba(current_x)[:, 1]
        segments["Predicted_Churn"] = (
            segments["Churn_Probability"] >= 0.5
        ).astype(int)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clean transactions", f"{len(df):,}")
    c2.metric("Customers", f"{len(segments):,}")
    c3.metric("Selected clusters", len(segments.Cluster.unique()))
    c4.metric("90-day churn training rate", f"{training.Churn.mean():.1%}")

    st.subheader("Customer output")
    st.dataframe(
        segments.sort_values("Churn_Probability", ascending=False),
        use_container_width=True,
        height=430,
    )

    st.subheader("Segment summary")
    summary = (
        segments.groupby("Segment")
        .agg(
            Customers=("CustomerID", "count"),
            Mean_Recency=("Recency", "mean"),
            Mean_Frequency=("Frequency", "mean"),
            Mean_Monetary=("Monetary", "mean"),
            Mean_Churn_Risk=("Churn_Probability", "mean"),
        )
        .reset_index()
    )
    st.dataframe(summary, use_container_width=True)
    st.download_button(
        "Download customer segments + churn probabilities",
        segments.to_csv(index=False).encode(),
        "customer_segments_churn.csv",
        "text/csv",
    )
    st.caption(
        f"Prospective churn model uses a 91-day holdout beginning after {cutoff.date()}."
    )
