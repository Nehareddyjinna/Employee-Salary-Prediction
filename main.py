import pandas as pd

# Step 1: Load the dataset with incorrect format
df_raw = pd.read_csv("data/preprocessed_data.csv", header=None)

# Step 2: Split the single string column into multiple columns
df = df_raw[0].str.split(",", expand=True)

# Step 3: Set the first row as the header
df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)

# Step 4: Check columns
print("Columns after fixing:", df.columns.tolist())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Step 1: Split into features and target
X = df.drop(columns=["income", "fnlwgt", "race", "relationship"])
y = df["income"]

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 3: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 5: Save the model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/salary_model.joblib")
print("Model saved to models/salary_model.joblib")
