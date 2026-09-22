# Multivariate Customer Segmentation and Churn Prediction for E-Commerce

**Student:** Wriddhijit Roy  
**Roll No.:** 2548613  
**Course:** MCAI513B-3 - Multivariate Techniques  
**Program:** MSc Computational Statistics & Applied AI, CHRIST (Deemed to be University)

## Project objective

This project models each customer's purchase behavior as the random vector **X = (Recency, Frequency, Monetary)^T**. It combines PCA, K-Means clustering, MANOVA, Hotelling's T-squared and Linear Discriminant Analysis (LDA) to produce interpretable customer segments and a prospective 90-day churn-risk score.

## Dataset

The supplied file has **541,909 rows** and spans **1 Dec 2010 - 9 Dec 2011**, so it corresponds to the UCI **Online Retail** dataset rather than the two-year Online Retail II file named in the original proposal. Official source: https://archive.ics.uci.edu/dataset/352/online+retail

After excluding transactions without CustomerID, cancellations, non-positive values and duplicates, the analysis uses **392,692 transactions, 4,338 customers and 18,532 invoices**.

## Main results

- PCA: PC1 = 75.35%, PC2 = 18.68%; cumulative first two PCs = 94.03%.
- K-Means: best silhouette over k=2...8 at **k=2**, silhouette = **0.469**.
- Loyal/high-value cluster: 1,678 customers and 85.5% of customer-level monetary value.
- MANOVA cluster effect: Wilks' lambda = 0.3167, F = 3117.28, p < 0.001.
- Prospective churn proxy: 3,370 historical customers; churn rate = 43.0%.
- Hotelling T-squared = 677.84, F = 225.81, p < 0.001.
- LDA test ROC-AUC = **0.725**, balanced accuracy = **0.665**, churn F1 = **0.627**.

## Repository structure

```text
data/raw/                 raw dataset location (not committed)
data/processed/           derived RFM/churn tables
src/                      reusable Python functions
notebooks/                reproducible notebook
outputs/                  statistical results and publication-quality figures
flowchart/                end-to-end workflow
app/                      Streamlit application
evidence_templates/       LinkedIn/SDK/peer evidence guidance
report/                    final report
```

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# Put online_retail.csv in data/raw/
python run_analysis.py
streamlit run streamlit_app.py
```

## GitHub Student Developer Pack integration

The project is designed to use **GitHub Codespaces** from the Student Developer Pack for a reproducible cloud environment. Evidence must be generated from the student's own account: pack-activation screenshot, Codespaces run screenshot, and public port/app demonstration. See `evidence_templates/GitHub_Student_Pack_Evidence.md`.

## Deployment

In GitHub Codespaces:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Set port **8501** to Public for the demonstration and copy that public URL into the LMS submission and LinkedIn posts. A permanent alternative is Streamlit Community Cloud, but if the rubric specifically requires Codespaces/Antigravity, show the Codespaces URL/evidence.

## Statistical caution

The transformed RFM variables still reject normality, and Box's M rejects equal covariance matrices for the churn groups. Therefore Hotelling's pooled-covariance test and LDA are reported as course-aligned classical methods with an explicit assumption caveat. QDA is a reasonable follow-up model because it permits class-specific covariance matrices.

## Academic integrity / AI disclosure

See `AI_DISCLOSURE.md`. AI assistance must be disclosed in the submitted report. All code should be reviewed and understood by the student before viva/submission.

## References

- UCI Machine Learning Repository. Online Retail. https://archive.ics.uci.edu/dataset/352/online+retail
- Chen, D., Sain, S. L., & Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. *Journal of Database Marketing & Customer Strategy Management, 19*(3), 197-208. https://doi.org/10.1057/dbm.2012.17
- scikit-learn documentation: PCA, KMeans, and LinearDiscriminantAnalysis.
