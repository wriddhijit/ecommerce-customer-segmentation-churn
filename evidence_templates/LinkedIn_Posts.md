# LinkedIn Post Drafts - CIA 3

Replace `[GITHUB URL]`, `[APP URL]`, and instructor tag before publishing. Attach `outputs/04_pca_customer_segments.png` to Post 1 and `outputs/07_lda_roc_curve.png` to Post 2.

## Post 1 - Development Chronicle: From Transactions to Multivariate Segments

I am working on my CIA 3 individual project for **MCAI513B-3: Multivariate Techniques** at CHRIST (Deemed to be University: **Multivariate Customer Segmentation and Churn Prediction for E-Commerce**.

Using the UCI Online Retail transaction data, I converted purchase histories into a customer-level RFM random vector - Recency, Frequency, and Monetary value - and treated the three measures jointly rather than as independent univariate scores. After cleaning 541,909 raw transaction rows, the analysis retained 392,692 valid sales records representing 4,338 customers.

For dimensionality reduction, PCA showed that the first two principal components explain 94.0% of the standardized RFM variation. K-Means model selection over k=2...8 selected **k=2** with a silhouette score of **0.469**. The resulting segments show a strong contrast between a high-value, frequent, recent-purchase group and a low-engagement, higher-recency group.

Next, I am connecting the segmentation results to multivariate inference and prospective churn classification using MANOVA, Hotelling's T-squared, and Linear Discriminant Analysis.

GitHub: [GITHUB URL]
App: [APP URL]
Instructor: [TAG DR. DIBU A S]

#MultivariateAnalysis #DataScience #CHRIST #MachineLearning #CustomerAnalytics #MarketingAnalytics #OpenToWork

---

## Post 2 - Results and Deployed Application

I have completed the main analysis for my CIA 3 project, **Multivariate Customer Segmentation and Churn Prediction for E-Commerce**.

A key design choice was to avoid defining churn directly from the same recency value used as a predictor. Instead, I created a **prospective 91-day holdout window**: RFM features were calculated using transactions available before the cutoff, and a customer was labelled churned only when no purchase occurred in the following outcome window. This produced 3,370 historical customers, with an observed churn proxy rate of 43.0%.

Hotelling's T-squared found a substantial multivariate difference between retained and churned customer RFM profiles (T-squared=677.84, F=225.81, p<0.001). The LDA model achieved ROC-AUC **0.725**, balanced accuracy **0.665**, and churn-class F1 **0.627** on the held-out test set.

I also built a Streamlit dashboard that accepts a transaction CSV and returns customer segment labels and churn probabilities. The project helped me connect PCA, clustering, MANOVA, multivariate mean testing, and discriminant analysis in one reproducible marketing-analytics workflow.

GitHub: [GITHUB URL]
Live App: [APP URL]
Instructor: [TAG DR. DIBU A S]

#MultivariateAnalysis #DataScience #CHRIST #Python #CustomerSegmentation #ChurnPrediction #OpenToWork
