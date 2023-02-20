import math

import numpy as np
from sklearn.metrics import auccracy_score

from regression_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    # expected_first_prediction_value = 113422
    expected_no_predictions = 131

    # When
    result = make_prediction(input_data=sample_input_data)

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, np.ndarray)
    assert isinstance(predictions[0], np.int64)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    _predictions = list(predictions)
    y_true = sample_input_data["survived"]
    auccracy =  auccracy_score(_predictions,y_true)
    assert auccracy > 0.7

    # assert math.isclose(predictions[0], expected_first_prediction_value, abs_tol=100)
