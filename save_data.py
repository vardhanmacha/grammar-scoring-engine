import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

# Load your CSV
df = pd.read_csv(r'C:\grammer project\archive\Grammar correction.csv')

# Print columns to verify
print("Columns in dataset:", df.columns)

# Rename columns to match what we need
df = df.rename(columns={
    "Ungrammatical Statement": "input_text",
    "Standard English": "target_text"
})

# Drop rows with any missing values in these columns
df = df.dropna(subset=["input_text", "target_text"])

# Split into train and validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to required format
train_data = train_df[["input_text", "target_text"]].to_dict(orient="records")
val_data = val_df[["input_text", "target_text"]].to_dict(orient="records")

# Save as pickle
with open("train_data.pkl", "wb") as f:
    pickle.dump(train_data, f)

with open("val_data.pkl", "wb") as f:
    pickle.dump(val_data, f)

print("Data saved as train_data.pkl and val_data.pkl")
