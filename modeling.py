
import numpy as np

import pandas as pd
from pandas import DataFrame

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score, fbeta_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.externals.six import StringIO
import pydot
from sklearn import tree

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc

from random import sample
from plotting import hist

from data import NUMERIC_TYPES

import matplotlib.pyplot as plt


def plot_roc_curve(y_true, y_scores):

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")


def plot_precision_recall_curve(y_true, y_scores):

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    # Plot ROC curve
    plt.plot(recall, precision, label='P-R curve (area = %0.2f)' % pr_auc)
    plt.plot([0, 1], [1, 0], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower left")


def plot_distribution(clf, features, targets, fit_params=None, **kwargs):
    score_groups = get_scores(clf, features, targets,
                              StratifiedKFold(targets, n_folds=2), fit_params)
    hist(score_groups[0].groupby('targets'), var='predict_proba_1', **kwargs)


def balance_dataset(grouped, shrink=True, random=False):

    df = grouped.obj
    return df.ix[balanced_indices(grouped, shrink=shrink, random=random)]


def balanced_indices(grouped, shrink=False, random=False):

    lengths = [len(group) for idx, group in grouped.groups.iteritems()]

    if shrink:
        size = min(lengths)
    else:
        size = max(lengths)

    indices = []

    for group_key, group_indices in grouped.groups.iteritems():
        if random:
            sub_indices = np.random.choice(group_indices, size=size)
        else:
            sub_indices = group_indices[:size]
        indices.extend(sub_indices)

    return indices

'''
def balanced_indices(srs):
    """
    Take a set of indices 
    Return (row based) indices determining
    a list of rows that compromise a balanced
    dataset of the input series.
    To be used as: srs.iloc[indices]
    """
    values = set(srs.values)

    num = min(srs.value_counts())

    all_indices = []

    for val in values:
        reduced = srs[srs==val]
        all_indices.extend(sample(reduced.index, num))

    return get_index_rows(srs, all_indices)
'''



def get_prediction(classifier, features, targets=None, retain_columns=None):
    """
    Takes a trained classifier  as well as a set
    of features to test on and the true classes for those features.
    Returns a DataFrame containing the predicted class,
    the predicted class probabilities, and the true
    class (if available)
    """
    predictions = classifier.predict(features)

    df_dict = {'predict':predictions}
    if targets is not None:
        df_dict[targets.name] = targets
        df_dict['targets'] = targets

    scores = classifier.predict_proba(features).T
    for idx, scores in enumerate(scores):
        df_dict['predict_proba_%s' % idx] = scores

    prediction = pd.DataFrame(df_dict)
    prediction = prediction.set_index(features.index)

    if retain_columns is not None:
        prediction = prediction.join(features[retain_columns])

    return prediction


def get_scores(clf, features, targets, cv, balance=True, retain_columns=None):
    """
    Take an (untrained) classifier and a set of targets
    and features.  Split the targets and features into
    training and testing subsets using the input
    cross validation.  The classifier is trained on the
    training sets and is then scored on the testing set.
    This is repeated for every cross-validation set.
    A list of the results for each cross-validation set
    is returned.
    """

    groups = []

    for i, (train, test) in enumerate(cv):

        training_targets = targets.iloc[train]
        training_features = features.iloc[train]

        if balance:
            # Get the indices of a balanced dataset and use
            # on the training features and targets
            train_balanced = balanced_indices(training_targets.groupby(training_targets))
            training_targets = training_targets.ix[train_balanced]
            training_features = training_features.ix[train_balanced]

        classifier = clf.fit(training_features, training_targets)

        testing_targets = targets.iloc[test]
        testing_features = features.iloc[test]

        grp = get_prediction(classifier, testing_features, testing_targets, retain_columns)
        groups.append(grp)

    return groups


def feature_selection_trees(features, labels):
    """
    Returns a list of the names of the best features (as strings)
    """
    clf = ExtraTreesClassifier(n_estimators=100).fit(features, labels)
    #clf = clf.fit(features, labels)#.transform(features)
    importances = [x for x in zip(features.columns, clf.feature_importances_)]
    return sorted(importances, key=lambda x: x[1])


def get_best_features(features, labels, max_to_return=20):
    return [feature for (feature, importance) in feature_selection_trees(features, labels)[:max_to_return]]


def plot_tree(clf, file_name, **kwargs):
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data, **kwargs)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(file_name)


def get_scores_cv(classifier_class, features, labels, fit_params=None):
    """
    Return a dataframe where half of the features are scored
    """
    training_features = features[::2]
    training_labels = labels[::2]

    testing_features = features[1::2]
    testing_labels = labels[1::2]

    classifier = classifier_class.fit(training_features, training_labels, fit_params)

    grp = pd.DataFrame({'targets' : testing_labels.map(str),
                        'score' : [x[1] for x in classifier.predict_proba(testing_features)]}).groupby('targets')
    return grp


def get_floating_feature_names(df):

    float_feature_names = []

    for feature in df.columns:
        if df[feature].dtype in NUMERIC_TYPES:
            float_feature_names.append(feature)

    return float_feature_names


def get_floating_point_features(df, remove_na=True):

    float_feature_names = get_Floating_feature_names(df)
    float_features = df[float_feature_names]

    if remove_na:
        float_features = float_features.fillna(float_features.mean()).dropna(axis=1)

    return float_features


def check_nulls(df):
    nans = pandas.isnull(features).sum()
    nans.sort()
    return nans


def arff_to_df(arff):
    rows = []
    for row in arff[0]:
        rows.append(list(row))
    attributes = [x for x in arff[1]]

    # Create the DataFrame
    return pd.DataFrame(rows, columns=attributes)


def get_nominal_integer_dict(nominal_vals):
    d = {}
    for val in nominal_vals:
        if val not in d:
            current_max = max(d.values()) if len(d) > 0 else -1
            d[val] = current_max+1
    return d


def convert_to_integer(srs):
    d = get_nominal_integer_dict(srs)
    return srs.map(lambda x: d[x])


def convert_strings_to_integer(df):
    ret = pd.DataFrame()
    for column_name in df:
        column = df[column_name]
        if column.dtype=='string' or column.dtype=='object':
            ret[column_name] = convert_to_integer(column)
        else:
            ret[column_name] = column
    return ret


def get_index_rows(srs, indices):
    """
    Given a dataframe and a list of indices,
    return a list of row indices corresponding
    to the supplied indices (maintaining order)
    """
    rows = []
    for i, (index, row) in enumerate(srs.iteritems()): #.iterrows()):
        if index in indices:
            rows.append(i)
    return rows


def feature_importances(importances, features):
    return pd.DataFrame(sorted(zip(features.columns, importances), key=lambda x: -x[1]),
                        columns=['feature', 'value'])

def get_importances(features, targets):
    fit = RandomForestClassifier(n_estimators=100).fit(features, targets)
    return feature_importances(fit.feature_importances_, features)


class ScoreFeature():
    """
    A score created by a classifier using a
    specific set of features.
    Handles grabbing that set of features
    from a pandas dataset and
    scoring on those features.

    See examples in:
    /Users/george/Work/risk/IdFeatures.arff
    """

    def __init__(self, clf, names):
        self.clf = clf
        self.names = names
        self.fitted = None

    def get_features(self, df):
        return df[self.names]

    def score(self, features, class_index=1):
        scores = self.clf.predict_proba(self.get_features(features))
        class_scores = [score[class_index] for score in scores]
        return pd.Series(class_scores)
