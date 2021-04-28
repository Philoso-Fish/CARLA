from carla.data.catalog import DataCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
from carla.recourse_methods.catalog.actionable_recourse import Actionable_Recourse
from carla.recourse_methods.catalog.dice import Dice


def test_dice_get_counterfactuals():
    # Build data and mlmodel
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

    model_tf = MLModelCatalog(data, "ann", feature_input_order)
    # get factuals
    factuals = predict_negative_instances(model_tf, data)
    test_factual = factuals.iloc[:22]

    cfs = Dice(model_tf, data).get_counterfactuals(
        factuals=test_factual, num_of_cf=1, desired_class=1
    )

    assert test_factual.shape[0] == cfs.shape[0]


def test_ar_get_counterfactual():
    # Build data and mlmodel
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
    model_tf = MLModelCatalog(data, "ann", feature_input_order)

    # get factuals
    factuals = predict_negative_instances(model_tf, data)
    test_factual = factuals.iloc[:22]

    cfs = Actionable_Recourse(data, model_tf).get_counterfactuals(test_factual)
    # TODO: number of CFS and factuals should fit
    print(cfs)

    pass
