
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Keep MLflow tracking data inside this experiment folder
PROJECT_DIR = Path(__file__).resolve().parent
MLFLOW_DIR = PROJECT_DIR / "mlruns"

mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
mlflow.set_experiment("DVML_Experiment_5")

# Step 1: Load the dataset
dataset = load_breast_cancer(as_frame=True)
X = dataset.data
y = dataset.target

print("Dataset loaded successfully.")
print("Dataset shape:", X.shape)
print("\nFirst five rows:")
print(X.head())

# Step 2: Check missing values
print("\nMissing values:", int(X.isnull().sum().sum()))

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Step 4: Define model parameters
n_estimators = 100
max_depth = 5

# Step 5: Build the ML pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    ))
])

# Step 6: Train and track the experiment
with mlflow.start_run(run_name="RandomForest_Experiment"):

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Step 7: Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    # Step 8: Log model parameters
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("test_size", 0.2)

    # Step 9: Log evaluation metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Step 10: Log the trained model
    signature = infer_signature(X_train, model.predict(X_train))

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(3)
    )

    print("\nExperiment completed successfully.")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Parameters, metrics, and model logged to MLflow.")
    print("Run ID:", mlflow.active_run().info.run_id)

print("\nMLflow tracking directory:", MLFLOW_DIR)
