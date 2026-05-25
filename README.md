# Credit Card Customer Segmentation & Churn Intelligence

A customer analytics project that segments credit card holders into behavioral profiles and predicts churn risk, framed as a Relationship Manager decision-support tool for a retail bank.

Built as a portfolio project targeting banking data science roles.

---

## What it does

The project runs in two stages. First it groups customers into five behavioral segments using K-Means clustering. Then it trains a Random Forest classifier to predict which customers are likely to churn, with SHAP explanations so the model's reasoning is interpretable at the individual customer level.

The Streamlit app surfaces everything in a dashboard designed for a Relationship Manager; not just "here are clusters" but "here is what to do about this customer today."

---

## The five segments

| Segment | Churn Rate | Profile |
|---|---|---|
| Revolvers | 5.1% | Carry a balance, low credit limit, rate-sensitive |
| Stable Mid-Tier | 7.8% | Moderate engagement, not at immediate risk |
| Disengaged At-Risk | 37.9% | Low transactions, high contact frequency, urgent |
| Dormant High-Value | 18.1% | High credit limit, near-zero activity |
| High-Value Actives | 2.1% | Highest spend and engagement, cross-sell opportunity |

---

## Model performance

- ROC-AUC: 0.96
- Recall on churned class: 82%
- Precision on churned class: 71%

Optimized for recall because missing a churner is more costly than a false alarm in a retention context.

---

## Stack

- **Data:** Kaggle BankChurners (10,127 customers, 20 features)
- **Wrangling:** Pandas, NumPy
- **EDA:** Matplotlib, Seaborn
- **ML:** scikit-learn (StandardScaler, PCA, KMeans, RandomForestClassifier), imbalanced-learn (SMOTE)
- **Explainability:** SHAP
- **Macro data:** yfinance (live BOT rate proxy)
- **App:** Streamlit, Plotly

---

## Project structure

```
credit-card-segmentation/
│
├── data/
│   ├── BankChurners.csv
│   ├── BankChurners_engineered.csv
│   ├── test_predictions.csv
│   ├── shap_sample.csv
│   └── shap_values_sample.npy
│
├── models/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   └── kmeans.pkl
│
├── notebooks/
│   └── customer_segmentation.ipynb
│
├── app/
│   └── app.py
│
└── requirements.txt
```

---

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## A note on the dataset

The dataset is sourced from a US credit card portfolio. The BOT macro adjustment layer in the Streamlit app demonstrates the analytical framework that would apply to Thai retail banking data and is included to show how macro rate environment affects segment-level churn risk. The segmentation methodology, churn prediction framework, and SHAP explainability are all directly transferable to a Thai banking context.

---

## Key findings

- Churners disengage behaviorally before they formally close their account. Transaction count drops, inactivity rises, and contact frequency increases in the months before churn.
- Two structurally different churn types exist in the data: customers who were once active and gradually disengaged, and customers who never really activated their card.
- Month 3 of consecutive inactivity combined with elevated contact frequency is the strongest early warning signal.
- Long-tenure customers are not necessarily engaged. Months on book correlates negatively with engagement score at -0.58.
- Product depth is the strongest retention lever. Customers with one product churn at 25%+. Those with four or more churn at around 11%.
- Platinum card holders churn more than Silver despite being the premium tier, suggesting card prestige is a weaker retention signal than product relationship depth.
