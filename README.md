# Telco Customer Churn Prediction

A machine learning pipeline that predicts customer churn for a telecommunications company using the IBM Telco Customer Churn dataset. The project covers the full ML workflow — from data cleaning and exploratory analysis to model training, evaluation, and feature importance.

---

## Results

| Model | Test Accuracy | Churn Recall | Churn F1 |
|---|---|---|---|
| Logistic Regression | 0.739 | 0.78 | 0.61 |
| Decision Tree | 0.794 | 0.55 | 0.58 |
| Random Forest | 0.737 | **0.81** | **0.62** |
| Gradient Boosting | **0.797** | 0.51 | 0.57 |

> **Best model for this use case: Random Forest** — highest churn recall (0.81), meaning it catches the most customers at risk of leaving. In a real business context, missing a churner is more costly than a false alarm.

---

## Key Findings

- **Contract type, tenure, and monthly charges** are the strongest predictors of churn.
- Customers on **month-to-month contracts** churn at significantly higher rates.
- **Short-tenure customers** with high monthly charges are the highest-risk group.

---

## Project Structure

```
telco-churn/
│
├── telco_churn.py       # Main ML pipeline
├── requirements.txt     # Dependencies
├── README.md
└── .gitignore
```

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/atahancelenk/telco-churn.git
cd telco-churn
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Get the dataset from Kaggle: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Place the CSV file in the project root directory:
```
telco-churn/
└── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

**4. Run the script**
```bash
python telco_churn.py
```

---

## Tech Stack

- **Python 3.x**
- **pandas** — data manipulation
- **scikit-learn** — model training & evaluation
- **matplotlib / seaborn** — visualizations

---

## Workflow

1. **Data Cleaning** — Convert `TotalCharges` to numeric, impute missing values
2. **EDA** — Count plots for categorical features, histograms for numerical, churn heatmap
3. **Preprocessing** — Drop unnecessary columns, one-hot encoding, standard scaling (applied after train/test split to prevent data leakage)
4. **Model Training** — Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
5. **Evaluation** — Accuracy, precision, recall, F1-score
6. **Feature Importance** — Top 10 features from Random Forest

---

## License

This project is open source and available under the [MIT License](LICENSE).
