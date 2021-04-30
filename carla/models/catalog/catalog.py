import numpy as np
import pandas as pd
from sklearn import preprocessing

from carla.models.pipelining import encoder, order_data, scaler

from ..api import MLModel
from .load_model import load_model


class MLModelCatalog(MLModel):
    def __init__(
        self,
        data,
        model_type,
        feature_input_order,
        backend="tensorflow",
        cache=True,
        models_home=None,
        pipeline=None,
        encode_normalize_data=False,
        **kws
    ):
        """
        Constructor for pretrained ML models from the catalog.

        Possible backends are currently "pytorch" and "tensorflow".
        Possible models are corrently "ann".

        Parameters
        ----------
        data : data.api.Data Class
            Correct dataset for ML model
        model_type : str
            Architecture [ann]
        feature_input_order : list
            List containing all features in correct order for ML prediction
        backend : str
            Specifies the used framework [tensorflow, pytorch]
        cache : boolean, optional
            If True, try to load from the local cache first, and save to the cache
            if a download is required.
        models_home : string, optional
            The directory in which to cache data; see :func:`get_models_home`.
        kws : keys and values, optional
            Additional keyword arguments are passed to passed through to the read model function
        pipeline : list, optional
            List of (name, transform) tuples that are chained in the order in which they are preformed.
        encode_normalize_data : bool, optional
            If true, the model pipeline is used to build data.encoded, data.normalized and data.encoded_normalizd
        """
        self._backend = backend

        if self._backend == "pytorch":
            ext = "pt"
        elif self._backend == "tensorflow":
            ext = "h5"
        else:
            raise Exception("Model type not in catalog")

        self._model = load_model(model_type, data.name, ext, cache, models_home, **kws)

        self._continuous = data.continous
        self._categoricals = data.categoricals

        self._feature_input_order = feature_input_order

        # Preparing pipeline components
        self._pipeline = self.__init_pipeline(pipeline, data)

        if encode_normalize_data:
            data.set_encoded_normalized(self)

    def __init_pipeline(self, pipeline, data):
        if pipeline is None:
            self._scaler = preprocessing.MinMaxScaler().fit(data.raw[self._continuous])
            self._encoder = preprocessing.OneHotEncoder(
                handle_unknown="ignore", sparse=False
            ).fit(data.raw[self._categoricals])
            return [
                ("scaler", lambda x: scaler(self._scaler, self._continuous, x)),
                ("encoder", lambda x: encoder(self._encoder, self._categoricals, x)),
                ("order", lambda x: order_data(self._feature_input_order, x)),
            ]
        else:
            return pipeline

    def set_pipeline(self, steps):
        """
        Sets sequentially list of data transformations to perform on input before predictions.

        Parameters
        ----------
        steps : list
            List of (name, transform) tuples that are chained in the order in which they are preformed.

        Returns
        -------
        None
        """
        self._pipeline = steps

    def get_pipeline_element(self, key):
        """
        Returns a specific element of the pipeline

        Parameters
        ----------
        key : str
            Element of the pipeline we want to return

        Returns
        -------
        Pipeline element
        """
        key_idx = list(zip(*self._pipeline))[0].index(key)  # find key in pipeline
        return self._pipeline[key_idx][1]

    @property
    def pipeline(self):
        """
        Returns transformations steps for input before predictions.

        Returns
        -------
        pipeline : list
            List of (name, transform) tuples that are chained in the order in which they are preformed.
        """
        return self._pipeline

    def perform_pipeline(self, df):
        """
        Transforms input for prediction into correct form.
        Only possible for DataFrames without preprocessing steps.

        Recommended to use to keep correct encodings, normalization and input order

        Parameters
        ----------
        df : pd.DataFrame
            Contains unnormalized and not encoded data.

        Returns
        -------
        output : pd.DataFrame
            Prediction input in correct order, normalized and encoded

        """
        output = df.copy()

        for trans_name, trans_function in self._pipeline:
            output = trans_function(output)

        return output

    def need_pipeline(self, x):
        """
        Checks if ML model input needs pipelining.
        Only DataFrames can be used to perform_pipeline input.

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
    def feature_input_order(self):
        """
        Saves the required order of features as list.

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
        x : np.Array or pd.DataFrame
            Tabular data of shape N x M (N number of instances, M number of features)

        Returns
        -------
        output : np.ndarray
            Ml model prediction for interval [0, 1] with shape N x 1
        """

        if len(x.shape) != 2:
            raise ValueError("Input shape has to be two-dimensional")

        input = self.perform_pipeline(x) if self.need_pipeline(x) else x

        if self._backend == "pytorch":
            return self._model.predict(input)
        elif self._backend == "tensorflow":
            return self._model.predict(input)[:, 1]
        else:
            raise ValueError(
                'Uncorrect backend value. Please use only "pytorch" or "tensorflow".'
            )

    def predict_proba(self, x):
        """
        Two-dimensional probability prediction of ml model

        Shape of input dimension has to be always two-dimensional (e.g., (1, m), (n, m))

        Parameters
        ----------
        x : np.Array or pd.DataFrame
            Tabular data of shape N x M (N number of instances, M number of features)

        Returns
        -------
        output : float
            Ml model prediction with shape N x 2
        """

        if len(x.shape) != 2:
            raise ValueError("Input shape has to be two-dimensional")

        input = self.perform_pipeline(x) if self.need_pipeline(x) else x

        if self._backend == "pytorch":
            class_1 = 1 - self._model.forward(input).detach().numpy().squeeze()
            class_2 = self._model.forward(input).detach().numpy().squeeze()

            # For single prob prediction it happens, that class_1 is casted into float after 1 - prediction
            # Additionally class_1 and class_2 have to be at least shape 1
            if not isinstance(class_1, np.ndarray):
                class_1 = np.array(class_1).reshape(1)
                class_2 = class_2.reshape(1)

            return np.array(list(zip(class_1, class_2)))

        elif self._backend == "tensorflow":
            return self._model.predict(input)
        else:
            raise ValueError(
                'Uncorrect backend value. Please use only "pytorch" or "tensorflow".'
            )
