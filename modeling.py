import numpy as np

import pandas as pd
from pandas import DataFrame

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score, fbeta_score

from sklearn.externals.six import StringIO
import pydot
from sklearn import tree

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc

from random import sample
from plotting import hist

import matplotlib.pyplot as plt


def get_prediction(classifier, testing_features, testing_targets=None, retain_columns=None):
    """
    A better version of get_scores
    Takes a classifier and a
    """
    predictions = classifier.predict(testing_features)

    df_dict = {'predict':predictions}
    if testing_targets is not None:
        df_dict[testing_targets.name] = testing_targets

    scores = classifier.predict_proba(testing_features).T
    for idx, scores in enumerate(scores):
        df_dict['predict_proba_%s' % idx] = scores

    prediction = pd.DataFrame(df_dict)
    prediction = prediction.set_index(testing_features.index)

    if retain_columns is not None:
        prediction = prediction.join(testing_features[retain_columns])

    return prediction


def print_roc_curve(y_true, y_scores):

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


def print_precision_recall_curve(y_true, y_scores):

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


def print_distribution(clf, features, targets, fit_params=None, **kwargs):
    score_groups = get_scores(targets, features, clf,
                              StratifiedKFold(targets, n_folds=2), fit_params)
    hist(score_groups[0].groupby('targets'), var='score', **kwargs)


def balance_dataset(df, var='state', shrink=False, random_seed=None):
    grouped = df.groupby(var)

    N0 = len(grouped.groups[0])
    N1 = len(grouped.groups[1])

    print 'Balance dataset:'
    print '   Group 0: %d' % N0
    print '   Group 1: %d' % N1

    idx0 = grouped.groups[0]
    idx1 = grouped.groups[1]

    if shrink:
        if N1 > N0:
            print '   Shrinking group 1.'
            idx1 = idx1[:N0]
        else:
            print '   Shrinking group 0.'
            idx0 = idx0[:N1]

        df = pd.concat([df.ix[idx0],df.ix[idx1]])
        return df

    if N1 > N0:
        Ndraw = (N1-N0)
        print '   Growing group 0.'
        print '   Drawing %d random samples (with replacement).' % Ndraw
        idx = grouped.groups[0]
    else:
        Ndraw = (N0-N1)
        print '   Growing group 1.'
        print '   Drawing %d random samples (with replacement).' % Ndraw
        idx = grouped.groups[1]

    if random_seed is not None:
        np.random.seed(random_seed)
    ridx = np.random.choice(idx, size=Ndraw)
    df = pd.concat([df,df.ix[ridx]])

    return df


def balanced_indices(srs):
    values = set(srs.values)

    num = min(srs.value_counts())

    all_indices = []

    for val in values:
        reduced = srs[srs==val]
        all_indices.extend(sample(reduced.index, num))

    return all_indices


def feature_selection_trees(features, labels):
    """
    Returns a list of the names of the best features (as strings)
    """
    clf = ExtraTreesClassifier(n_estimators=100)
    X_new = clf.fit(features, labels).transform(features)
    importances = [x for x in zip(features.columns, clf.feature_importances_)]
    return sorted(importances, key=lambda x: x[1])


def get_best_features(features, labels, max_to_return=20):
    return [feature for (feature, importance) in feature_selection_trees(features, labels)[:max_to_return]]


def print_tree(clf, file_name, **kwargs):
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data, **kwargs)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(file_name)


# Return a dataframe where half of the features are scored
def get_scores_cv(classifier_class, features, labels, fit_params=None):
    training_features = features[::2]
    training_labels = labels[::2]

    testing_features = features[1::2]
    testing_labels = labels[1::2]

    classifier = classifier_class.fit(training_features, training_labels, fit_params)

    grp = pd.DataFrame({'targets' : testing_labels.map(str),
                        'score' : [x[1] for x in classifier.predict_proba(testing_features)]}).groupby('targets')
    return grp


def get_floating_point_features(df, remove_na=True):
    # Get floating point features
    float_feature_names = []

    for feature in df.columns:
        if df[feature].dtype == 'float64':
            float_feature_names.append(feature)

    float_features = df[float_feature_names]

    if remove_na:
        float_features = float_features.fillna(float_features.mean()).dropna(axis=1)

    return float_features


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


def balanced_indices(srs):
    """
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


def get_scores(targets, features, clf, cv, balance=True):
    """
    Return a list of DataFrames consisting
    of target values and scores
    """

    groups = []

    for i, (train, test) in enumerate(cv):

        training_targets = targets.iloc[train]
        training_features = features.iloc[train]

        if balance:
            train_balanced = balanced_indices(training_targets)
            training_targets = training_targets.iloc[train_balanced]
            training_features = training_features.iloc[train_balanced]

        classifier = clf.fit(training_features, training_targets)

        testing_features = features.iloc[test]
        scores = [x[1] for x in classifier.predict_proba(testing_features)]

        grp = pd.DataFrame({'targets' : targets.iloc[test],
                            'score' : scores})
        grp['targets'] = grp['targets'].map(str)
        groups.append(grp)

    return groups


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
