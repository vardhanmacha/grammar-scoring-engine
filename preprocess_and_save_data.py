import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset
df = pd.read_csv(r'C:\grammer project\archive\Grammar correction.csv')  # replace with actual dataset name

# Drop Serial Number column
df = df.drop(columns=['Serial Number'])

# Clean the text
df['Ungrammatical Statement'] = df['Ungrammatical Statement'].str.lower().str.strip()
df['Standard English'] = df['Standard English'].str.lower().str.strip()

# Split into input (X) and output (y)
X = df['Ungrammatical Statement']
y = df['Standard English']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save to pickle files
with open("X_train.pkl", "wb") as f:
    pickle.dump(X_train.tolist(), f)

with open("X_test.pkl", "wb") as f:
    pickle.dump(X_test.tolist(), f)

with open("y_train.pkl", "wb") as f:
    pickle.dump(y_train.tolist(), f)

with open("y_test.pkl", "wb") as f:
    pickle.dump(y_test.tolist(), f)

print("Data saved as pickle files.")
