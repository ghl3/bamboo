

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


def print_distribution(clf, features, targets):
    score_groups = get_scores(targets, features, clf, StratifiedKFold(targets, n_folds=2))
    bins = np.arange(0, 1.0, .05)
    hist(score_groups[0].groupby('targets'), var='score', normed=True, alpha=0.5, bins=bins)


def feature_importances(importances, features):
    return pd.DataFrame(sorted(zip(features.columns, importances), key=lambda x: -x[1]),
                        columns=['feature', 'value'])


def get_importances(features, targets):
    fit = RandomForestClassifier(n_estimators=100).fit(features, targets)
    return feature_importances(fit.feature_importances_, features)

