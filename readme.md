# Credit Risk Early-Warning System

A full-stack machine learning application that predicts whether a loan applicant is a good or bad credit risk — built end-to-end from raw data to a live, deployed, interactive web app.

**Live app:** https://credit-risk-early-warning-2jg9juc4b7ncjrkjnnazeq.streamlit.app/

## Problem

Banks need to flag high-risk loan applicants before approval. This project builds a model that does exactly that, using the UCI Statlog German Credit dataset (1000 applicants, 20 attributes each, with a 70/30 good/bad risk imbalance).

The key design decision throughout: **optimizing for recall on defaulters, not raw accuracy.** A model that predicts "good" for everyone would score 70% accuracy while catching zero actual defaulters — for a bank, missing a real defaulter is far costlier than a false alarm.

## Tech Stack

- **PostgreSQL** — data storage and business-question SQL analysis
- **Python (pandas, scikit-learn, seaborn/matplotlib)** — cleaning, EDA, modeling
- **Streamlit** — deployed web application
- **Supabase** — free cloud Postgres, powering live submission logging on the deployed app

## SQL Insights

Business-question queries run directly against the dataset:
- Applicants employed **<1 year showed higher default risk (40.7%) than unemployed applicants (37.1%)** — counterintuitive, likely reflecting job-hopping/instability in new roles
- **Education loans were riskiest (44% bad risk)**; used car and retraining loans were safest (16.5%, 11.1%)

## EDA Highlights

- Bad-risk applicants had a higher median loan amount (~2600 DM vs ~2200 DM) and more high-value outliers
- `duration` and `credit_amount` were the numeric features most correlated with risk (0.21, 0.15)

## Model Comparison

Five classifiers were trained and compared, ranked by recall on the "bad risk" class:

| Model | Recall (bad risk) | Precision (bad risk) | Accuracy |
|---|---|---|---|
| **Logistic Regression** | **0.80** | 0.56 | 0.76 |
| Naive Bayes | 0.65 | 0.51 | 0.70 |
| XGBoost | 0.48 | 0.53 | 0.71 |
| Decision Tree | 0.50 | 0.49 | 0.69 |
| Random Forest | 0.33 | 0.67 | 0.75 |

**Logistic Regression was selected as the final model**, despite Random Forest and XGBoost having near-identical or better accuracy — their recall on actual defaulters was far worse, which would mean missing more real risk cases in production. This is a deliberate example of prioritizing the metric that matches the business problem over the model with the most sophisticated reputation.

## Application Features

- **Predict tab** — form-based risk prediction for any new applicant, using the trained Logistic Regression model
- **Why This Result? tab** — personalized explanation showing the top factors (via model coefficients) that pushed each individual prediction toward risk or safety
- **Live database logging** — every submission is saved to a cloud Postgres database, making the app's usage data grow over time
- **Admin dashboard** (private, link-gated) — live analytics across all submissions: risk breakdown, risk by purpose/employment, credit amount distribution

## Business Recommendation

Given the model's 80% recall on defaulters, deploying this as a first-pass screening tool would let a bank flag roughly 4 out of 5 actual high-risk applicants for manual review before approval — at the cost of some false positives requiring extra review. For a lender, this tradeoff is generally favorable, since the cost of an undetected default significantly exceeds the cost of an unnecessary manual check.

## Project Structure

```
├── app.py                     # Streamlit application
├── requirements.txt           # Python dependencies
├── credit_risk_model.pkl      # Trained Logistic Regression model
├── model_columns.pkl          # Feature column structure for encoding new data
└── README.md
```

## Limitations & Future Work

- Dataset is a public benchmark dataset, not live bank data — real-world deployment would require validation on current, institution-specific data
- Free-tier Supabase database pauses after 7 days of inactivity
- Future scope: SHAP-based explainability, hyperparameter tuning, cost-sensitive threshold optimization, MLflow experiment tracking
