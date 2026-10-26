
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


def create_model():
    """Create and train a Random Forest classifier."""
    dataset = load_iris()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(dataset.data, dataset.target)

    return model


def predict(features):
    """Predict the Iris class from four input features."""
    if len(features) != 4:
        raise ValueError("Exactly four input features are required.")

    model = create_model()
    return int(model.predict([features])[0])
