# Probability of Default Prediction Model

A logistic regression model that predicts the probability of loan default using real-world lending data from LendingClub.

## Overview

This project builds a **Probability of Default (PD) model** — a core credit risk modeling technique used by lenders and financial institutions to estimate the likelihood that a borrower will default on a loan. The model is trained on 1.3M+ historical loan records and evaluated using industry-standard classification metrics.

## Dataset

- **Source:** LendingClub loan data
- **Size:** 1,347,681 rows, 15 columns
- **Target variable:** `Default` (binary: 1 = defaulted, 0 = did not default)
- **Base default rate:** ~20%

**Key features used:**
| Feature | Description |
|---|---|
| `loan_amnt` | Loan amount requested |
| `revenue` | Borrower's annual income |
| `dti_n` | Debt-to-income ratio |
| `fico_n` | FICO credit score |
| `home_ownership_n` | Home ownership status (one-hot encoded: OWN, RENT, OTHER) |

## Methodology

1. **Data preparation** — Loaded and cleaned the raw CSV, handled mixed-type columns, one-hot encoded the categorical `home_ownership_n` feature.
2. **Train-test split** — 70/30 split, stratified on the target to preserve the default rate in both sets.
3. **Model pipeline** — `StandardScaler` + `LogisticRegression` (scikit-learn `Pipeline`), keeping preprocessing and modeling leakage-free and reproducible.
4. **Evaluation** — Assessed using AUC-ROC, ROC curve, and classification report — the standard metrics for credit risk models, since raw accuracy is misleading on imbalanced classes.

## Results

The AUC-ROC (0.65) reflects the model's underlying ability to rank riskier borrowers above safer ones, and is unchanged by the class-weight adjustment. What changes is the *decision threshold behavior*: with balanced class weights, the model correctly flags a much larger share of actual defaults, at the cost of more false positives — a standard precision-recall trade-off in credit risk scoring.

