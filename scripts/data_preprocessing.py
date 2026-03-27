import pandas as pd
raw = pd.read_csv('../Data/cs-training.csv', index_col=0)

RENAME = {
    'SeriousDlqin2yrs':                        'defaulted',
    'RevolvingUtilizationOfUnsecuredLines':     'unsecured_credit',
    'age':                                      'age',
    'NumberOfTime30-59DaysPastDueNotWorse':     'delinq_30_59',
    'DebtRatio':                                'debt_ratio',
    'MonthlyIncome':                            'monthly_income',
    'NumberOfOpenCreditLinesAndLoans':          'open_credit',
    'NumberOfTimes90DaysLate':                  'delinq_90',
    'NumberRealEstateLoansOrLines':             'real_estate_loans',
    'NumberOfTime60-89DaysPastDueNotWorse':     'delinq_60_89',
    'NumberOfDependents':                       'dependents',
}
raw = raw.rename(columns=RENAME)

df = raw.copy()
df = df[df['age'] > 0]

# Sentinel treatment: flag codes 96/98, cap delinquency at 10
DELINQ_COLS = ['delinq_30_59', 'delinq_60_89', 'delinq_90']
df['delinq_sentinel'] = (df['delinq_90'] >= 96).astype(int)
for col in DELINQ_COLS:
    df[col] = df[col].clip(upper=10)

# Missing indicators (derived from raw, before imputation)
df['monthly_income_missing'] = raw.loc[df.index, 'monthly_income'].isna().astype(int)
df['dependents_missing']     = raw.loc[df.index, 'dependents'].isna().astype(int)

# Imputation
df['monthly_income'] = df['monthly_income'].fillna(df['monthly_income'].median())
df['dependents']     = df['dependents'].fillna(0)

FEATURE_COLS = [
    'unsecured_credit', 'age', 'delinq_30_59', 'debt_ratio',
    'monthly_income', 'open_credit', 'delinq_90', 'real_estate_loans',
    'delinq_60_89', 'dependents', 'monthly_income_missing', 'dependents_missing',
    'delinq_sentinel',
]

print(f"Shape          : {df.shape}")
print(f"Default rate   : {df['defaulted'].mean():.3%}")
print(f"Sentinel rows  : {df['delinq_sentinel'].sum():,}")
