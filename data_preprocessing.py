import pandas as pd

# Replace 'grammar_data.csv' with the actual filename if different
df = pd.read_csv(r'C:\grammer project\archive\Grammar correction.csv')

# Display basic info
print("Dataset Info:")
print(df.info())

# Show first few rows
print("\nFirst 5 Rows:")
print(df.head())

# Check for null values
print("\nMissing Values:")
print(df.isnull().sum())

# Show distribution of target column (if present)
if 'grammar_score' in df.columns:
    print("\nGrammar Score Distribution:")
    print(df['grammar_score'].describe())
