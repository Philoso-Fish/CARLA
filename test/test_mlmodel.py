import numpy as np
import torch

from carla.data.catalog import DataCatalog
from carla.models.catalog import MLModelCatalog


def test_properties():
    data_name = "adult"
    data_catalog = "adult_catalog.yaml"
    data = DataCatalog(data_name, data_catalog, drop_first_encoding=True)

    feature_input_order = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "workclass_Private",
        "marital-status_Non-Married",
        "occupation_Other",
        "relationship_Non-Husband",
        "race_White",
        "sex_Male",
        "native-country_US",
    ]

    model_tf_adult = MLModelCatalog(data, "ann", feature_input_order)

    exp_backend_tf = "tensorflow"
    exp_feature_order_adult = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "workclass_Private",
        "marital-status_Non-Married",
        "occupation_Other",
        "relationship_Non-Husband",
        "race_White",
        "sex_Male",
        "native-country_US",
    ]

    assert model_tf_adult.backend == exp_backend_tf
    assert model_tf_adult.feature_input_order == exp_feature_order_adult


def test_predictions_tf():
    data_name = "adult"
    data_catalog = "adult_catalog.yaml"
    data = DataCatalog(data_name, data_catalog, drop_first_encoding=True)

    feature_input_order = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "workclass_Private",
        "marital-status_Non-Married",
        "occupation_Other",
        "relationship_Non-Husband",
        "race_White",
        "sex_Male",
        "native-country_US",
    ]

    model_tf_adult = MLModelCatalog(data, "ann", feature_input_order)

    single_sample = data.encoded_normalized.iloc[22]
    single_sample = single_sample[model_tf_adult.feature_input_order].values.reshape(
        (1, -1)
    )
    samples = data.encoded_normalized.iloc[0:22]
    samples = samples[model_tf_adult.feature_input_order].values

    # Test single and bulk non probabilistic predictions
    single_prediction_tf = model_tf_adult.predict(single_sample)
    expected_shape = tuple((1, 1))
    assert single_prediction_tf.shape == expected_shape

    predictions_tf = model_tf_adult.predict(samples)
    expected_shape = tuple((22, 1))
    assert predictions_tf.shape == expected_shape

    # Test single and bulk probabilistic predictions
    single_predict_proba_tf = model_tf_adult.predict_proba(single_sample)
    expected_shape = tuple((1, 2))
    assert single_predict_proba_tf.shape == expected_shape

    predictions_proba_tf = model_tf_adult.predict_proba(samples)
    expected_shape = tuple((22, 2))
    assert predictions_proba_tf.shape == expected_shape

    # Check predictions for pipeline
    samples = data.raw.iloc[0:22]

    predictions_proba_tf = model_tf_adult.predict_proba(samples)
    expected_shape = tuple((22, 2))
    assert predictions_proba_tf.shape == expected_shape

    predictions_tf = model_tf_adult.predict(samples)
    expected_shape = tuple((22, 1))
    assert predictions_tf.shape == expected_shape


def test_predictions_pt():
    data_name = "adult"
    data_catalog = "adult_catalog.yaml"
    data = DataCatalog(data_name, data_catalog, drop_first_encoding=False)

    feature_input_order = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "sex_Female",
        "sex_Male",
        "workclass_Non-Private",
        "workclass_Private",
        "marital-status_Married",
        "marital-status_Non-Married",
        "occupation_Managerial-Specialist",
        "occupation_Other",
        "relationship_Husband",
        "relationship_Non-Husband",
        "race_Non-White",
        "race_White",
        "native-country_Non-US",
        "native-country_US",
    ]
    model = MLModelCatalog(data, "ann", feature_input_order, backend="pytorch")

    single_sample = data.encoded_normalized.iloc[22]
    single_sample = single_sample[model.feature_input_order].values.reshape((1, -1))
    single_sample_torch = torch.Tensor(single_sample)

    samples = data.encoded_normalized.iloc[0:22]
    samples = samples[model.feature_input_order].values
    samples_torch = torch.Tensor(samples)

    # Test single non probabilistic predictions
    single_prediction = model.predict(single_sample)
    expected_shape = tuple((1, 1))
    assert single_prediction.shape == expected_shape
    assert isinstance(single_prediction, np.ndarray)

    single_prediction_torch = model.predict(single_sample_torch)
    expected_shape = tuple((1, 1))
    assert single_prediction_torch.shape == expected_shape
    assert torch.is_tensor(single_prediction_torch)

    # bulk non probabilistic predictions
    predictions = model.predict(samples)
    expected_shape = tuple((22, 1))
    assert predictions.shape == expected_shape
    assert isinstance(predictions, np.ndarray)

    predictions_torch = model.predict(samples_torch)
    expected_shape = tuple((22, 1))
    assert predictions_torch.shape == expected_shape
    assert torch.is_tensor(predictions_torch)

    # Test single probabilistic predictions
    single_predict_proba = model.predict_proba(single_sample)
    expected_shape = tuple((1, 2))
    assert single_predict_proba.shape == expected_shape
    assert isinstance(single_predict_proba, np.ndarray)

    single_predict_proba_torch = model.predict_proba(single_sample_torch)
    expected_shape = tuple((1, 2))
    assert single_predict_proba_torch.shape == expected_shape
    assert torch.is_tensor(single_predict_proba_torch)

    # bulk probabilistic predictions
    predictions_proba = model.predict_proba(samples)
    expected_shape = tuple((22, 2))
    assert predictions_proba.shape == expected_shape
    assert isinstance(single_predict_proba, np.ndarray)

    predictions_proba_torch = model.predict_proba(samples_torch)
    expected_shape = tuple((22, 2))
    assert predictions_proba_torch.shape == expected_shape
    assert torch.is_tensor(predictions_proba_torch)
