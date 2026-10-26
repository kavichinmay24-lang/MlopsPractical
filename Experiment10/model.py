
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load the Iris dataset.
data = load_iris()

X = data.data
y = data.target

# Create and train the model.
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X, y)


def predict(features):
    """Predict the Iris class from four input features."""
    prediction = model.predict([features])
    return int(prediction[0])
