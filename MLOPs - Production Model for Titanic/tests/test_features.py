from regression_model.config.core import config
from regression_model.processing.features import ExtractLetterTransformer


def test_temporal_variable_transformer(sample_input_data):
    # Given
    transformer = ExtractLetterTransformer(
        variables=config.model_config.cabin,  # YearRemodAdd
        # reference_variable=config.model_config.ref_var,
    )
    assert sample_input_data["cabin"].iat[6] == 'E12'

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    assert subject["cabin"].iat[6] == 'E'
