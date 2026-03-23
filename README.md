# AutoValue Pro: Automotive Intelligence & Predictive Market Analysis
**Author:** Navdeep Singh Tung | **Institution:** Toronto Metropolitan University (CIND820)

## Project Overview
AutoValue Pro is a Prescriptive Analytics and Decision Support System (DSS) designed to eliminate information asymmetry in the used car market. Moving beyond static price prediction, this engine calculates the Total Cost of Ownership (TCO) and generates a definitive Purchase Quality Score (PQS) by bridging machine learning valuations with live mechanical risk data.

## Research Questions Addressed
1. **Predictive Modeling:** Utilizing XGBoost to handle high-cardinality automotive data and accurately forecast Fair Market Value.
2. **Anomaly Detection:** Deploying PySpark for IQR statistical filtering to purge market outliers and $1 listings.
3. **Explainable AI (XAI):** Implementing SHAP (TreeExplainer) to interpret feature importance (e.g., Odometer vs. Age).
4. **Synthesis of Decision Support:** Engineering a multi-pillar Purchase Quality Score (PQS) combining the Financial Deal, Asset Retention, and NHTSA Mechanical Risk.

## Project Stages & Pipeline
* **Phase 1-3 (Data Ingestion & EDA):** Loading the primary dataset, handling missing values, and exploring feature correlations.
* **Phase 4 (PySpark Data Engineering):** Advanced NLP text standardization, N>=50 statistical thresholding, CooperUnion MSRP integration, and CAD localization.
* **Phase 5-6 (XGBoost & SHAP):** Training the gradient boosting regressor and extracting human-readable attributions to prevent black-box bias.
* **Phase 7-9 (Hyperparameter Tuning):** Optimizing tree depth and learning rates to maximize the R² score and minimize MAE.
* **Phase 10 (The DSS Dashboard):** A live Gradio UI integrating a real-time `requests.get` pipeline to the US Government NHTSA Complaints API. Includes a monotonic financial governor to prevent tree-based extrapolation hallucinations.

## Repository Contents
* `AutoValue_Pro_Final.ipynb`: The master PySpark/Python codebase.
* `AutoValue_Pro_Final.pdf`: The compiled, readable technical report.
* **Used Cars Dataset (Kaggle):** https://www.kaggle.com/api/v1/datasets/download/austinreese/craigslist-carstrucks-data

## Technical Architecture
* **Languages:** Python 3, PySpark, SQL
* **Libraries:** XGBoost, SHAP, Gradio, Pandas, Matplotlib
* **External APIs:** US Department of Transportation (NHTSA) Live Complaints API
