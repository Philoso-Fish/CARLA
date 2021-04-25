def prepare_dataset_for_dice(data, mlmodel):
    feature_dict = dict()
    feature_order = mlmodel.feature_input_order

    for feature in feature_order:
        if feature in data.continous:
            min = data.raw[feature].min()
            max = data.raw[feature].max()
            feature_dict[feature] = [min, max]
        else:
            for cat in data.categoricals:
                if cat in feature:
                    if cat not in feature_dict.keys():
                        feature_dict[cat] = []
                    feature_dict[cat].append(feature.replace(cat + "_", ""))
                    break


def feature_order_for_raw(data, mlmodel):
    feature_order_encoded = []
    feature_order = mlmodel.feature_input_order

    for feature in feature_order:
        if feature in data.continous:
            feature_order_encoded.append(feature)
        else:
            for cat in data.categoricals:
                if (cat in feature) and (cat not in feature_order_encoded):
                    feature_order_encoded.append(cat)
                    break

    return feature_order_encoded
