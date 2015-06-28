from __future__ import division

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing._weights import _balance_weights as balance_weights
from sklearn.cross_validation import ShuffleSplit
from sklearn import cross_validation

from bamboo.helpers import NUMERIC_TYPES
import bamboo.plotting


class ModelingData():

    """
    A class that stores a set of features and
    targets to be used for modeling.
    Provides functionality to do simple operations
    and manipulations of that data, removing
    boilerplate for ML code.
    """

    def __init__(self, features, targets, weights=None):
        self.features = features
        self.targets = targets
        self.weights = weights

    @staticmethod
    def from_dataframe(df, target, features=None):

        target_data = df[target]

        if features is None:
            features = [feature for feature in df.columns
                        if feature != target]

        feature_data = df[features]

        return ModelingData(feature_data, target_data)

    def __len__(self):
        assert(len(self.features) == len(self.targets))
        return len(self.targets)

    def __str__(self):
        return "ModelingData({})".format(self.features.shape)

    def shape(self):
        return self.features.shape

    def num_classes(self):
        return len(self.targets.value_counts())

    def get_grouped_targets(self):
        return self.targets.groupby(self.targets)

    def is_orthogonal(self, other):
        indices = set(self.features.index)
        return not any(idx in indices for idx in other.features.index)

    def filter(self, filter_function):
        keep = self.features.apply(filter_function, axis=1)
        return ModelingData(self.features[keep], self.targets[keep])

    def subset_features(self, features):
        return ModelingData(self.features[features], self.targets)

    def fit(self, clf, *args, **kwargs):
        return clf.fit(self.features, self.targets, *args, **kwargs)

    def train_test_split(self, *args, **kwargs):
        """
        Based on: sklearn.cross_validation.train_test_split
        Returns two ModelingData instances, the first representing
        training data and the scond representing testing data
        """
        test_size = kwargs.pop('test_size', None)
        train_size = kwargs.pop('train_size', None)
        random_state = kwargs.pop('random_state', None)

        if test_size is None and train_size is None:
            test_size = 0.25

        n_samples = len(self.targets)

        cv = ShuffleSplit(n_samples, test_size=test_size,
                          train_size=train_size,
                          random_state=random_state)

        train, test = next(iter(cv))

        X_train = self.features.ix[train]
        y_train = self.targets.ix[train]

        X_test = self.features.ix[test]
        y_test = self.targets.ix[test]

        return ModelingData(X_train, y_train), ModelingData(X_test, y_test)

    def _balance_by_truncation(self):
        """
        Take a modeling_data instance and return
        a new instance with the state variable
        balanced
        """

        group_size = self.targets.value_counts().min()
        grouped = self.features.groupby(self.targets)
        indices = []

        for name, group in grouped:
            indices.extend(group[:group_size].index)

        return ModelingData(self.features.ix[indices], self.targets.ix[indices])

    def _balance_by_sample_with_replace(self, size=None, exact=False):
        if size is None:
            size = len(self)

        approx_num_per_class = size / self.num_classes()

        indices = []

        for target, group in self.get_grouped_targets():
            indices.extend(np.random.choice(group.index.values, approx_num_per_class, replace=True))

        return ModelingData(self.features.ix[indices], self.targets.ix[indices])

    def get_balance_weights(self):
        return balance_weights(self.targets)

    def get_balanced(self, how='sample'):
        """
        Return a ModelingData derived from this instance
        but with balanced data (balanced according to
        the supplied options)
        """

        if how == 'truncate':
            return self._balance_by_truncation()
        elif how == 'sample':
            return self._balance_by_sample_with_replace()
        else:
            raise AttributeError()

    def hist(self, var_name, **kwargs):
        grouped = self.features[var_name].groupby(self.targets)
        return bamboo.plotting._series_hist(grouped, **kwargs)

    def hist_all(self, shape=None, binning_map=None, figsize=None, **kwargs):

        fig = plt.figure(figsize=figsize)

        if shape is None:
            x = 3
            y = math.ceil(len(self.features.columns) / x)
        else:
            (x, y) = shape

        for i, feature in enumerate(self.features.columns):
            plt.subplot(x, y, i + 1)
            if binning_map and feature in binning_map:
                self.hist(feature, bins=bins, **kwargs)
            else:
                self.hist(feature, autobin=True, **kwargs)
            plt.xlabel(feature)
        plt.tight_layout()

    def stack(self, var_name, **kwargs):
        grouped = self.features[var_name].groupby(self.targets)
        return bamboo.plotting._draw_stacked_plot(grouped, **kwargs)

    def numeric_features(self):
        """
        Return a copy of thos ModelData that only contains
        numeric features
        """

        dtypes = self.features.dtypes
        numeric_dtypes = dtypes[dtypes.map(lambda x: x in NUMERIC_TYPES)]
        numeric_feature_names = list(numeric_dtypes.index.values)

        return ModelingData(self.features[numeric_feature_names], self.targets)

    def plot_auc_surve(self, clf):
        return plotting.plot_auc_curve(clf, self.features, self.targets)

    def predict_proba(self, clf):

        # The order of the targets in the classification predict_proba
        # matrix is based on the natural ordering of the input targets
        # So, we have to follow that natural ordering here
        ordered_targets = sorted(set(self.targets.values))

        scores = []

        for idx, row in self.features.iterrows():
            probabilities = clf.predict_proba(row)[0]

            res = {'index': idx}

            for target, proba in zip(ordered_targets, probabilities):
                res['proba_{}'.format(target)] = proba

            res['target'] = self.targets[idx]

            scores.append(res)

        return pd.DataFrame(scores)

    def predict(self, reg):

        scores = []

        for idx, row in self.features.iterrows():
            res = {'index': idx}
            res['predict'] = reg.predict(row)[0]
            res['target'] = self.targets[idx]
            scores.append(res)

        return pd.DataFrame(scores)

    def plot_proba(self, clf, target, **kwargs):
        probabilities = self.predict_proba(clf)
        target_name = 'proba_{}'.format(target)
        reduced = probabilities[[target_name, 'target']]
        return bamboo.plotting._series_hist(reduced.groupby('target')[target_name], **kwargs)

    def _cross_validate_score(self, clf, fit=False, **kwargs):
        return cross_validation.cross_val - score(clf, self.features, self.targets, **kwargs)

    def get_classifier_performance_summary(self, clf, target, thresholds=np.arange(0.0, 1.01, 0.01), **kwargs):
        """
        Take a classifier and a target
        and return a DataFrame listing the
        performance result as various thresholds
        for that classifier
        """
        probas = self.predict_proba(clf)
        threshold_summaries = [
            ModelingData.get_threshold_summary(
                probas,
                target,
                threshold) for threshold in thresholds]
        threshold_df = pd.DataFrame(threshold_summaries)
        threshold_df = threshold_df.set_index('threshold')
        return probas, threshold_df

    def get_classifier_score_and_threshold_summary(self, clf, target, thresholds=np.arange(0.0, 1.01, 0.01), **kwargs):
        pass

    @staticmethod
    def get_threshold_summary(probabilities, target, threshold=0.5, **kwargs):
        """
        Takes a probability summary, a target we're
        trying to predict (that is represented in the
        probability summary) and the threshold for
        that probability and returns a summary

        probabolity_summary = [{'proba_A': x, 'target': A}, {'proba_A': y, 'target': B}]
        """

        probability_label = "proba_{}".format(target)

        probability_summary = pd.DataFrame(probabilities)

        positives = probability_summary[probability_summary[probability_label] >= threshold]
        true_positives = positives[positives['target'] == target]
        false_positives = positives[positives['target'] != target]

        negatives = probability_summary[probability_summary[probability_label] < threshold]
        true_negatives = negatives[negatives['target'] != target]
        false_negatives = negatives[negatives['target'] == target]

        num = len(probability_summary)

        num_positives = len(positives)
        num_negatives = len(negatives)

        num_true_positives = len(true_positives)
        num_false_positives = len(false_positives)
        num_true_negatives = len(true_negatives)
        num_false_negatives = len(false_negatives)

        precision = num_true_positives / \
            (num_true_positives + num_false_positives) if num_true_positives + num_false_positives > 0 else 1.0
        recall = num_true_positives / \
            (num_true_positives + num_false_negatives) if num_true_positives + num_false_negatives > 0 else 1.0

        sensitivity = recall
        specificity = num_true_negatives / \
            (num_false_positives + num_true_negatives) if num_false_positives + num_true_negatives > 0 else 1.0

        true_positive_rate = sensitivity
        false_positive_rate = (1.0 - specificity)

        accuracy = (num_true_positives + num_true_negatives) / num
        f1 = 2 * num_true_positives / (2 * num_true_positives + num_false_positives +
                                       num_false_negatives) if 2 * num_true_positives + num_false_positives + num_false_negatives else 0.0

        return {'threshold': threshold,
                'target': target,
                'true_positives': num_true_positives,
                'false_positives': num_false_positives,
                'true_negatives': num_true_negatives,
                'false_negatives': num_false_negatives,
                'precision': precision,
                'recall': recall,
                'sensiticity': sensitivity,
                'specificity': specificity,
                'accuracy': accuracy,
                'f1': f1,
                'false_positive_rate': false_positive_rate,
                'true_positive_rate': true_positive_rate}
