
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load the built-in Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X, y)

# Save the trained model in this project folder
model_path = Path(__file__).parent / "model.pkl"
joblib.dump(model, model_path)

print("Model trained successfully.")
print(f"Model saved to: {model_path.name}")
print("Number of training samples:", len(X))
print("Number of input features:", X.shape[1])
