# Input Data Specification

Dataset: Kaggle *Give Me Some Credit* (2011) — 150,000 borrowers, binary target `SeriousDlqin2yrs`

---

## Input Features

| # | Sheet Column | Internal Name | Type | Range | Typical | Notes |
|---|---|---|---|---|---|---|
| 1 | `RevolvingUtilizationOfUnsecuredLines` | `unsecured_credit` | Float | 0.0 – 1.5+ | 0.0 – 1.0 | Total revolving balance / credit limit. Values >1.0 = over-limit (valid, not capped) |
| 2 | `age` | `age` | Int | 21 – 100 | 25 – 75 | Rows with age=0 dropped in training |
| 3 | `NumberOfTime30-59DaysPastDueNotWorse` | `delinq_30_59` | Int | 0 – 98 | 0 – 3 | Clipped to 10 by model. 96/98 are sentinel codes |
| 4 | `DebtRatio` | `debt_ratio` | Float | 0.0 – 5.0+ | 0.0 – 1.0 | Monthly debt payments / income. >1.0 is valid |
| 5 | `MonthlyIncome` | `monthly_income` | Float | 0 – 100,000+ | 1,000 – 25,000 | Can be blank (imputed to median 5,400) |
| 6 | `NumberOfOpenCreditLinesAndLoans` | `open_credit` | Int | 0 – 40 | 2 – 15 | Count of open credit lines and loans |
| 7 | `NumberOfTimes90DaysLate` | `delinq_90` | Int | 0 – 98 | 0 – 1 | Clipped to 10. Values 96/98 trigger sentinel flag |
| 8 | `NumberRealEstateLoansOrLines` | `real_estate_loans` | Int | 0 – 10 | 0 – 3 | Count of mortgage / real estate loans |
| 9 | `NumberOfTime60-89DaysPastDueNotWorse` | `delinq_60_89` | Int | 0 – 98 | 0 – 1 | Clipped to 10. 96/98 are sentinel codes |
| 10 | `NumberOfDependents` | `dependents` | Int | 0 – 10 | 0 – 3 | Can be blank (imputed to 0) |

---

## Auto-Derived Features

These are computed by the preprocessing pipeline — do not enter them manually.

| Feature | Type | Value | Derivation |
|---|---|---|---|
| `monthly_income_missing` | Binary | 0 or 1 | 1 if `MonthlyIncome` is blank/NaN |
| `dependents_missing` | Binary | 0 or 1 | 1 if `NumberOfDependents` is blank/NaN |
| `delinq_sentinel` | Binary | 0 or 1 | 1 if raw `NumberOfTimes90DaysLate` >= 96 |

---

## Sentinel Codes (96 / 98)

The delinquency columns (`30-59`, `60-89`, `90+`) use values **96** and **98** as data-entry codes — they do not represent actual counts. The pipeline:

1. Flags `delinq_sentinel = 1` if `delinq_90 >= 96`
2. Clips all three delinquency columns to a maximum of **10**

This preserves the information (severe default event) without distorting the count features.

---

## Missingness

| Feature | Missing Rate | Mechanism | Imputation |
|---|---|---|---|
| `MonthlyIncome` | 19.8% | MNAR (higher default rate when missing, chi-square p<0.05) | Median (5,400) + `monthly_income_missing` flag |
| `NumberOfDependents` | 2.6% | Low MI, likely MCAR | Zero + `dependents_missing` flag |

---

## Target Variable

| Column | Values | Meaning |
|---|---|---|
| `SeriousDlqin2yrs` | 0 or 1 | 1 = borrower had 90+ days past-due delinquency within 2 years |

- Default rate: ~6.7% (class imbalance ratio ~13.5:1)
- Blank/NaN in the test set (predictions to be filled)
