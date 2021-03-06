
import numpy as np
import pandas as pd

from bamboo import frames

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.externals.six import StringIO
import pydot
from sklearn import tree

from sklearn.cross_validation import StratifiedKFold

from bamboo.helpers import NUMERIC_TYPES

import matplotlib.pyplot as plt


def plot_roc_curve(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
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
    frames.hist(score_groups[0].groupby('targets'), var='predict_proba_1', **kwargs)


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


def get_prediction(classifier, features, targets=None, retain_columns=None):
    """
    Takes a trained classifier  as well as a set
    of features to test on and the true classes for those features.
    Returns a DataFrame containing the predicted class,
    the predicted class probabilities, and the true
    class (if available)
    """
    predictions = classifier.predict(features)

    df_dict = {'predict': predictions}
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
    # clf = clf.fit(features, labels)#.transform(features)
    importances = [x for x in zip(features.columns, clf.feature_importances_)]
    return sorted(importances, key=lambda x: x[1])


def get_best_features(features, labels, max_to_return=20):
    return [feature for (feature, importance) in feature_selection_trees(features, labels)[:max_to_return]]


def feature_importances(importances, features):
    return pd.DataFrame(sorted(zip(features.columns, importances), key=lambda x: -x[1]),
                        columns=['feature', 'value'])


def get_importances(features, targets):
    fit = RandomForestClassifier(n_estimators=100).fit(features, targets)
    return feature_importances(fit.feature_importances_, features)


def plot_tree(clf, file_name, **kwargs):
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data, **kwargs)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(file_name)


def get_floating_feature_names(df):

    float_feature_names = []

    for feature in df.columns:
        if df[feature].dtype in NUMERIC_TYPES:
            float_feature_names.append(feature)

    return float_feature_names


def get_floating_point_features(df, remove_na=True):

    float_feature_names = get_floating_feature_names(df)
    float_features = df[float_feature_names]

    if remove_na:
        float_features = float_features.fillna(float_features.mean()).dropna(axis=1)

    return float_features


def get_nominal_integer_dict(nominal_vals):
    d = {}
    for val in nominal_vals:
        if val not in d:
            current_max = max(d.values()) if len(d) > 0 else -1
            d[val] = current_max + 1
    return d


def convert_to_integer(srs):
    d = get_nominal_integer_dict(srs)
    return srs.map(lambda x: d[x])


def convert_strings_to_integer(df):
    ret = pd.DataFrame()
    for column_name in df:
        column = df[column_name]
        if column.dtype == 'string' or column.dtype == 'object':
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
    for i, (index, row) in enumerate(srs.iteritems()):
        if index in indices:
            rows.append(i)
    return rows


def score_summary(classifier, features, targets, scoring, **kwargs):
    """
    Run cross-validation using the input (untrained) classifier
    with the input features and targets.
    Apply the requested cross-validation techniques
    and print a summary
    """

    for cv in scoring:
        scores = cross_validation.cross_val_score(classifier, features, targets, scoring=cv, **kwargs)
        print '----- %s -----' % cv
        print scores
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print '\n'
