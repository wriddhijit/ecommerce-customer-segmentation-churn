"""Core analysis functions for the CIA3 e-commerce multivariate project."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = {
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country"
}


def load_and_clean(source):
    """Load transaction CSV and apply project cleaning rules."""
    df = pd.read_csv(source)
    if "index" in df.columns:
        df = df.drop(columns=["index"])
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df[df["CustomerID"].notna() & df["InvoiceDate"].notna()].copy()
    df = df[~df["InvoiceNo"].astype(str).str.upper().str.startswith("C")].copy()
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    dedupe_cols = [
        "InvoiceNo", "StockCode", "Description", "Quantity",
        "InvoiceDate", "UnitPrice", "CustomerID", "Country"
    ]
    df = df.drop_duplicates(subset=dedupe_cols)
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    return df


def compute_rfm(df, reference_date=None):
    """Aggregate clean transactions into customer-level RFM vectors."""
    if reference_date is None:
        reference_date = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("CustomerID")
        .agg(
            last=("InvoiceDate", "max"),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Revenue", "sum"),
        )
        .reset_index()
    )
    rfm["Recency"] = (reference_date - rfm["last"]).dt.days
    return rfm[["CustomerID", "Recency", "Frequency", "Monetary"]]


def transform_rfm(rfm, lower=None, upper=None):
    """Log-transform and 1%-99% winsorize RFM coordinates."""
    x = np.log1p(rfm[["Recency", "Frequency", "Monetary"]].astype(float))
    if lower is None:
        lower = x.quantile(0.01)
    if upper is None:
        upper = x.quantile(0.99)
    x = x.clip(lower, upper, axis=1)
    x.columns = ["R_log", "F_log", "M_log"]
    return x, lower, upper


def segment_customers(rfm, k_range=range(2, 9), random_state=42):
    """PCA reduction followed by silhouette-selected K-Means clustering."""
    x, lower, upper = transform_rfm(rfm)
    scaler = StandardScaler()
    z = scaler.fit_transform(x)

    pca_full = PCA().fit(z)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cumulative >= 0.90) + 1)
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(z)

    scores, models = {}, {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=50, random_state=random_state).fit(pcs)
        scores[k] = float(silhouette_score(pcs, km.labels_))
        models[k] = km

    best_k = max(scores, key=scores.get)
    labels = models[best_k].labels_
    out = rfm.copy()
    out["Cluster"] = labels

    means = out.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    loyal = int(means["Monetary"].idxmax())
    out["Segment"] = out["Cluster"].map(
        lambda k: "Loyal / High-Value" if k == loyal else "At-Risk / Low-Engagement"
    )
    return out, {
        "scaler": scaler,
        "pca": pca,
        "kmeans": models[best_k],
        "silhouette_scores": scores,
        "lower": lower,
        "upper": upper,
    }


def build_churn_training(df, holdout_days=91):
    """Build prospective churn labels using a temporal outcome window."""
    cutoff = (df["InvoiceDate"].max() - pd.Timedelta(days=holdout_days)).replace(
        hour=23, minute=59, second=59
    )
    history = df[df["InvoiceDate"] <= cutoff].copy()
    future = df[df["InvoiceDate"] > cutoff].copy()
    rfm = compute_rfm(history, cutoff.normalize() + pd.Timedelta(days=1))
    future_ids = set(future["CustomerID"].unique())
    rfm["Churn"] = (~rfm["CustomerID"].isin(future_ids)).astype(int)
    x, lower, upper = transform_rfm(rfm)
    training = pd.concat([rfm.reset_index(drop=True), x.reset_index(drop=True)], axis=1)
    return training, cutoff, lower, upper


def fit_lda(training, random_state=42):
    """Fit course-aligned LDA baseline with a stratified 70/30 split."""
    features = ["R_log", "F_log", "M_log"]
    X = training[features]
    y = training["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_state
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis()),
    ])
    model.fit(X_train, y_train)
    return model, (X_test, y_test)
