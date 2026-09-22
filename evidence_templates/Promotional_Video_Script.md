# Promotional Video Script (about 100-120 seconds)

**0-10 sec - Introduction**
"Hi, I am Wriddhijit Roy. This is my CIA 3 project for MCAI513B-3 Multivariate Techniques at CHRIST University. My project is Multivariate Customer Segmentation and Churn Prediction for E-Commerce."

**10-30 sec - Show app upload screen**
"The application takes transaction-level retail data and first cleans cancellations, missing customer IDs, invalid values, and duplicate rows. It then converts the transactions into the three-dimensional RFM customer vector: recency, frequency, and monetary value."

**30-55 sec - Show PCA/cluster figure**
"Instead of scoring each RFM measure independently, I use PCA and K-Means to capture their joint structure. Two principal components explain 94.0% of the variance. The best silhouette score was 0.469 at k=2, separating loyal high-value customers from a lower-engagement group."

**55-80 sec - Show churn results / ROC figure**
"For churn, I use a prospective 91-day holdout so the target is based on future purchasing rather than being defined directly from recency. Hotelling's T-squared shows that retained and churned mean RFM vectors differ significantly. The LDA test ROC-AUC is 0.725."

**80-105 sec - Show table/output download**
"The deployed Streamlit app returns each customer's segment, recency, frequency, monetary value, and estimated churn probability, and the results can be downloaded as a CSV for retention targeting."

**105-115 sec - Close**
"The full reproducible code and results are available in my GitHub repository. This project integrates PCA, clustering, MANOVA, Hotelling's T-squared, and Linear Discriminant Analysis from the course."

**Before posting:** add captions/subtitles, include `[GITHUB URL]` in the LinkedIn post, and keep the final video between 60 and 180 seconds.
