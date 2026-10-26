
from pathlib import Path

import joblib

from ml_model import create_model


def main():
    model = create_model()

    output_path = Path(__file__).parent / "model.joblib"
    joblib.dump(model, output_path)

    print("Model trained successfully.")
    print(f"Model saved to: {output_path.name}")


if __name__ == "__main__":
    main()
