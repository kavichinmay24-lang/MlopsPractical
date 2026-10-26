
import pytest

from ml_model import create_model, predict


def test_model_trains_successfully():
    model = create_model()

    assert model is not None
    assert hasattr(model, "predict")


def test_prediction_returns_valid_iris_class():
    result = predict([5.1, 3.5, 1.4, 0.2])

    assert result in [0, 1, 2]


def test_prediction_rejects_wrong_number_of_features():
    with pytest.raises(ValueError):
        predict([5.1, 3.5])
