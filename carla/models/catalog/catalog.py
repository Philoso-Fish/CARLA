import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing

from ..api import MLModel
from .load_model import load_model


class MLModelCatalog(MLModel):
    def __init__(
        self, data, data_name, model_type, ext="h5", cache=True, models_home=None, **kws
    ):
        """
        Constructing the ML model

        Parameters
        ----------
        model_type : str
            Name of the model ``{name}.{ext}`` on https://github.com/indyfree/cf-models.
        data_name : str
            Name of the dataset the model has been trained on.
        cache : boolean, optional
            If True, try to load from the local cache first, and save to the cache
            if a download is required.
        models_home : string, optional
            The directory in which to cache data; see :func:`get_models_home`.
        kws : keys and values, optional
            Additional keyword arguments are passed to passed through to the read model function
        data : data.api.Data Class
            Correct dataset for ML model
        ext : String
            File extension of saved ML model file
        """
        # check if dataset fits to ML model
        assert data.name == data_name

        self._continuous = data.continous
        self._categoricals = data.categoricals

        self._data_name = data_name
        self._model = load_model(model_type, data_name, ext, cache, models_home, **kws)

        if ext == "pt":
            self._backend = "pytorch"
        elif ext == "h5":
            self._backend = "tensorflow"
        else:
            raise Exception("Model type not in catalog")

        self._name = model_type + "_" + data_name

        # Preparing pipeline components
        self._scaler = preprocessing.MinMaxScaler().fit(data.raw[self._continuous])

        if data_name == "adult":  # TODO: Hier Lösung für Hardgecodeten quatsch finden
            if (
                self._backend == "tensorflow"
            ):  # TODO: Only for test, look for better implementation
                self._feature_input_order = [
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
                self._encodings = [  # Encodings should be built in the get_dummy way: {column}_{value}
                    "workclass_Private",
                    "marital-status_Non-Married",
                    "occupation_Other",
                    "relationship_Non-Husband",
                    "race_White",
                    "sex_Male",
                    "native-country_US",
                ]
            else:
                self._feature_input_order = [  # TODO: Make a yaml for those thing, similar to adult_catalog.yaml
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
                self._encodings = [  # Encodings should be built in the get_dummy way: {column}_{value}
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

        else:
            raise Exception("Model for dataset not in catalog")

    def pipeline(self, df):
        """
        Transforms input for prediction into correct form.
        Only possible for DataFrames without preprocessing steps.

        Recommended to use to keep correct encodings, normalization and input order

        Parameters
        ----------
        df : pd.DataFrame
            Contains unnormalized and

        Returns
        -------
        output : pd.DataFrame
            Prediction input in correct order, normalized and encoded

        """
        output = df.copy()

        # Normalization
        output[self._continuous] = self._scaler.transform(output[self._continuous])

        # Encoding
        output[self._encodings] = 0
        for encoding in self._encodings:
            for cat in self._categoricals:
                if cat in encoding:
                    value = encoding.split(cat + "_")[-1]
                    output.loc[output[cat] == value, encoding] = 1
                    break

        # Get correct order
        output = output[self._feature_input_order]

        return output

    def need_pipeline(self, x):
        """
        Checks if ML model input needs pipelining.
        Only DataFrames can be used to pipeline input.

        Parameters
        ----------
        x : pd.DataFrame or np.Array

        Returns
        -------
        bool : Boolean
            True if no pipelining process is already taken
        """
        if not isinstance(x, pd.DataFrame):
            return False

        if x.select_dtypes(exclude=[np.number]).empty:
            return False

        return True

    @property
    def name(self):
        """
        Contains meta information about the model, like name, dataset, structure, etc.

        Returns
        -------
        name : str
            Individual name of the ml model
        """
        return self._name

    @property
    def feature_input_order(self):
        """
        Saves the required order of feature as list.

        Prevents confusion about correct order of input features in evaluation

        Returns
        -------
        ordered_features : list of str
            Correct order of input features for ml model
        """
        return self._feature_input_order

    @property
    def backend(self):
        """
        Describes the type of backend which is used for the ml model.

        E.g., tensorflow, pytorch, sklearn, ...

        Returns
        -------
        backend : str
            Used framework
        """
        return self._backend

    @property
    def raw_model(self):
        """
        Returns the raw ml model built on its framework

        Returns
        -------
        ml_model : tensorflow, pytorch, sklearn model type
            Loaded model
        """
        return self._model

    def predict(self, x):
        """
        One-dimensional prediction of ml model for an output interval of [0, 1]

        Shape of input dimension has to be always two-dimensional (e.g., (1, m), (n, m))

        Parameters
        ----------
        x : np.Array, pd.DataFrame, or backend specific (tensorflow or pytorch tensor)
            Tabular data of shape N x M (N number of instances, M number of features)

        Returns
        -------
        output : np.ndarray, or backend specific (tensorflow or pytorch tensor)
            Ml model prediction for interval [0, 1] with shape N x 1
        """

        assert len(x.shape) == 2

        input = self.pipeline(x) if self.need_pipeline(x) else x

        if self._backend == "pytorch":
            # Pytorch model needs torch.Tensor as input
            if torch.is_tensor(input):
                device = "cuda" if input.is_cuda else "cpu"
                self._model = self._model.to(
                    device
                )  # Keep model and input on the same device
                output = self._model(
                    input
                )  # If input is a tensor, the prediction will be a tensor too.
            else:
                # Convert ndArray input into torch tensor
                if isinstance(input, pd.DataFrame):
                    input = input.values
                input = torch.Tensor(input)

                self._model = self._model.to("cpu")
                output = self._model(input)

                # Convert output back to ndarray
                output = output.detach().cpu().numpy()

        elif self._backend == "tensorflow":
            output = self._model.predict(input)[:, 1].reshape(
                (-1, 1)
            )  # keep output in shape N x 1

        return output

    def predict_proba(self, x):
        """
        Two-dimensional softmax prediction of ml model

        Shape of input dimension has to be always two-dimensional (e.g., (1, m), (n, m))

        Parameters
        ----------
        x : np.Array or pd.DataFrame
            Tabular data of shape N x M (N number of instances, M number of features)

        Returns
        -------
        output : np.ndarray, or backend specific (tensorflow or pytorch tensor)
            Ml model prediction with shape N x 2
        """

        assert len(x.shape) == 2

        input = self.pipeline(x) if self.need_pipeline(x) else x

        if self._backend == "pytorch":
            class_1 = 1 - self.predict(input)
            class_2 = self.predict(input)

            if torch.is_tensor(class_1):
                output = torch.cat((class_1, class_2), dim=1)
            else:
                output = np.array(list(zip(class_1, class_2))).reshape((-1, 2))

        elif self._backend == "tensorflow":
            output = self._model.predict(input)

        return output
