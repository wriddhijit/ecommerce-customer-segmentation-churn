from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.customer_analysis import (
    load_and_clean, compute_rfm, segment_customers,
    build_churn_training, fit_lda, transform_rfm,
)

DATA = ROOT / "data/raw/online_retail.csv"
if not DATA.exists():
    raise FileNotFoundError(
        "Place online_retail.csv in data/raw/. See data/raw/README.md"
    )

df = load_and_clean(DATA)
rfm = compute_rfm(df)
segments, seg_model = segment_customers(rfm)
training, cutoff, lower, upper = build_churn_training(df)
lda, _ = fit_lda(training)
current_x, _, _ = transform_rfm(rfm, lower, upper)
segments["Churn_Probability"] = lda.predict_proba(current_x)[:, 1]
segments.to_csv(ROOT / "outputs/customer_predictions.csv", index=False)
print(
    f"Done. Customers={len(segments)}, "
    f"best k={len(set(segments.Cluster))}, cutoff={cutoff.date()}"
)
