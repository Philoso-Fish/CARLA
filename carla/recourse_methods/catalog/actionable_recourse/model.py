import numpy as np
import recourse as rs
from lime.lime_tabular import LimeTabularExplainer

from ...api import Recourse_Method


class Actionable_Recourse(Recourse_Method):
    def __init__(self, data, mlmodel, coeffs=None, intercepts=None):
        """

        Restrictions
        ------------
        -   Actionable Recourse (AR) supports only binary categorical features.
            See implementation at https://github.com/ustunb/actionable-recourse/blob/master/examples/ex_01_quickstart.ipynb
        -   AR is only defined on linear models. To make it work for arbitrary non-linear networks
            we need to find coefficients for every instance, for example with lime.

        Parameters
        ----------
        data
        mlmodel
        coeffs : np.ndArray
            Coefficients
        intercepts
        """
        self._data = data
        self._mlmodel = mlmodel
        # Build ActionSet
        self._action_set = rs.ActionSet(
            X=self._data.encoded_normalized[self._mlmodel.feature_input_order]
        )

        # transform immutable feature names into encoded feature names of self._data.encoded_normalized
        self._immutables = []
        # TODO: Maybe find a more elegant way to find encoded immutable feature names
        for feature in self._data.immutables:
            if feature in self._mlmodel.feature_input_order:
                self._immutables.append(feature)
            else:
                for cat in self._mlmodel.feature_input_order:
                    if cat not in self._immutables:
                        if feature in cat:
                            self._immutables.append(cat)
                            break

        for feature in self._immutables:
            self._action_set[feature].mutable = False
            self._action_set[feature].actionable = False

        self._coeffs, self._intercepts = coeffs, intercepts

    def get_lime_coefficients(self, factuals):
        """
        Actionable Recourse is only defined on linear models. To make it work for arbitrary non-linear networks
        we need to find the lime coefficients for every instance.

        Parameters
        ----------
        factuals : pd.DataFrame
            Instances we want to get lime coefficients

        Returns
        -------
        coeffs : np.ndArray
        intercepts : np.ndArray

        """
        coeffs = np.zeros(factuals.shape)
        intercepts = []
        lime_data = self._data.encoded_normalized[self._mlmodel.feature_input_order]
        lime_label = self._data.encoded_normalized[self._data.target]

        lime_exp = LimeTabularExplainer(
            training_data=lime_data.values,
            training_labels=lime_label,
            feature_names=self._mlmodel.feature_input_order,
            discretize_continuous=False,
            sample_around_instance=True,
            categorical_names=[
                cat
                for cat in self._mlmodel.feature_input_order
                if cat not in self._data.continous
            ]
            # self._data.encoded_normalized's categorical features contain feature name and value, separated by '_'
            # while self._data.categoricals do not contain those additional values.
        )

        for i in range(factuals.shape[0]):
            factual = factuals.iloc[i].T.values
            explanations = lime_exp.explain_instance(
                factual,
                self._mlmodel.predict_proba,
                num_features=len(self._mlmodel.feature_input_order),
            )
            intercepts.append(explanations.intercept[1])

            for tpl in explanations.local_exp[1]:
                coeffs[i][tpl[0]] = tpl[1]

        return coeffs, intercepts

    def get_counterfactuals(self, factuals):
        cfs = []

        # Prepare factuals
        querry_instances = factuals.copy()

        # check if querry_instances are not empty
        assert querry_instances.shape[0] >= 1

        # preprocessing for lime
        factuals_enc_norm = self._mlmodel.pipeline(querry_instances)

        # Check if we need lime to build coefficients
        if (self._coeffs is None) and (self._intercepts is None):
            print("Start generating LIME coefficients")
            self._coeffs, self._intercepts = self.get_lime_coefficients(
                factuals_enc_norm
            )
            print("Finished generating LIME coefficients")

        # generate counterfactuals
        for i in range(factuals_enc_norm.shape[0]):
            factual_enc_norm = factuals_enc_norm.iloc[i].T.values
            coeff = self._coeffs[i]
            intercept = self._intercepts[i]

            # Align action set to coefficients
            self._action_set.set_alignment(coefficients=coeff)

            # Build AR flipset
            fs = rs.Flipset(
                x=factual_enc_norm,
                action_set=self._action_set,
                coefficients=coeff,
                intercept=intercept,
            )
            fs_pop = fs.populate(total_items=100)

            # Get actions to flip predictions
            actions = fs_pop.actions

            for action in actions:
                candidate_cf = (factual_enc_norm + action).reshape(
                    (1, -1)
                )  # Reshape to keep two-dim. input
                # Check if candidate counterfactual really flipps the prediction of ML model
                pred_cf = np.argmax(self._mlmodel.predict_proba(candidate_cf))
                pred_f = np.argmax(
                    self._mlmodel.predict_proba(factual_enc_norm.reshape((1, -1)))
                )
                if pred_cf != pred_f:
                    cfs.append(candidate_cf)
                    break

        cfs = np.array(cfs).squeeze()
        # TODO: counterfactuals an der Stellen noch zurücktransformieren und anschließend als df ausgeben

        return cfs
